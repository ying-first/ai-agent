"""plugins/repo_index.py — 仓库符号索引（Python AST + 多语言正则）。

目标：让 Agent 找定义/找调用点时不必反复 read_file 全文。
索引落盘到 WORKSPACE_DIR/.agent_index/index.json，按 mtime+size 增量更新。

支持语言：
- Python：AST 精确提取类/函数/方法/调用关系；
- JS/TS、C#、C/C++、Go、Rust、Java、Ruby、PHP、Shell、PowerShell：正则启发式。
"""
import ast
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

from safety import WORKSPACE_DIR, CONFIG, tool
import sandbox as _sandbox

try:
    from tree_sitter import Language, Parser
except Exception:                 # tree-sitter 是可选依赖
    Language = Parser = None

_INDEX_VERSION = 3
_INDEX_DIR = os.path.join(WORKSPACE_DIR, ".agent_index")
_INDEX_PATH = os.path.join(_INDEX_DIR, "index.json")
_INDEX_LOCK = threading.RLock()
_CACHE = {"mtime": None, "data": None}
_OUT_MAX = 6000
_NON_SYMBOL_NAMES = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "new",
    "delete", "else", "do", "case", "using", "namespace", "get", "set",
}

_EXT_LANG = {
    ".py": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "tsx", ".mts": "ts", ".cts": "ts",
    ".cs": "csharp",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell", ".bash": "shell",
    ".ps1": "powershell",
}

# (regex, kind, name_group)
_REGEX_SPECS = {
    "js": [
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"), "function", 1),
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)"), "class", 1),
        (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("), "function", 1),
        (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"), "interface", 1),
        (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*="), "type", 1),
    ],
    "ts": [
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"), "function", 1),
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class", 1),
        (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"), "interface", 1),
        (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*="), "type", 1),
        (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?\("), "function", 1),
    ],
    "csharp": [
        (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|sealed|abstract|partial)\s+)*(?:class|record|struct|interface|enum)\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*namespace\s+([\w\.]+)"), "namespace", 1),
        (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|async|virtual|override|sealed|partial|extern|unsafe|new)\s+)+[\w<>\[\],\.\?]+\s+(\w+)\s*\("), "method", 1),
    ],
    "c": [
        (re.compile(r"^\s*(?:typedef\s+)?(?:struct|union|enum)\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*[\w\*\s]+?\b(\w+)\s*\([^;]*\)\s*\{?\s*$"), "function", 1),
    ],
    "cpp": [
        (re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct|union|enum)\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*namespace\s+(\w+)"), "namespace", 1),
        (re.compile(r"^\s*(?:[\w:\*&<>\[\],]+\s+)+(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$"), "function", 1),
    ],
    "go": [
        (re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)"), "function", 1),
        (re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)"), "type", 1),
    ],
    "rust": [
        (re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)"), "type", 1),
    ],
    "java": [
        (re.compile(r"^\s*(?:(?:public|private|protected|static|final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)+[\w<>\[\],\.\?]+\s+(\w+)\s*\("), "method", 1),
    ],
    "ruby": [
        (re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*)"), "type", 1),
        (re.compile(r"^\s*def\s+(?:self\.)?(\w+[?!=]?)"), "function", 1),
    ],
    "php": [
        (re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+(\w+)"), "function", 1),
    ],
    "shell": [
        (re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*\(\)\s*\{"), "function", 1),
    ],
    "powershell": [
        (re.compile(r"^\s*function\s+([\w\-]+)"), "function", 1),
    ],
}


def _cfg() -> dict:
    return CONFIG.get("repo_index", {}) or {}


def _clip(text: str, limit: int = _OUT_MAX) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...[输出已截断，原长 {len(text)} 字符]"


def _rel(path: str) -> str:
    return os.path.relpath(os.path.realpath(path), WORKSPACE_DIR).replace("\\", "/")


def _scope_prefix(path: str) -> str:
    if not path or path in (".", "./"):
        return ""
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(WORKSPACE_DIR, path))
    root = os.path.realpath(WORKSPACE_DIR)
    if not (full == root or full.startswith(root + os.sep)):
        return None
    rel = os.path.relpath(full, root).replace("\\", "/")
    return "" if rel == "." else rel.rstrip("/")


def _excluded_dir(name: str) -> bool:
    excludes = set(_cfg().get("exclude_dirs") or [])
    return name in excludes or name.startswith(".")


def _iter_files(root: str):
    max_files = int(_cfg().get("max_files", 5000) or 5000)
    max_kb = int(_cfg().get("max_file_kb", 1024) or 1024)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _excluded_dir(d))
        for fn in sorted(filenames):
            if (fn.startswith((".", "_")) or ".bak_" in fn or "副本" in fn):
                continue
            ext = os.path.splitext(fn)[1].lower()
            lang = _EXT_LANG.get(ext)
            if not lang:
                continue
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            if st.st_size > max_kb * 1024:
                continue
            yield full, lang, st, count
            count += 1
            if count >= max_files:
                return


# ================= Python AST =================
def _fmt_args(node) -> str:
    parts = [a.arg for a in getattr(node.args, "posonlyargs", [])]
    parts += [a.arg for a in node.args.args]
    if node.args.vararg:
        parts.append("*" + node.args.vararg.arg)
    parts += [a.arg for a in node.args.kwonlyargs]
    if node.args.kwarg:
        parts.append("**" + node.args.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


# ================= tree-sitter 多语言解析（可选） =================
_TS_MODULES = {
    "js": ("tree_sitter_javascript", "language"),
    "ts": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
}
_TS_LANGS = {}
_TS_FAILED = set()

_TS_DEF_TYPES = {
    "function_definition", "function_declaration", "function_item",
    "method_definition", "method_declaration", "constructor_declaration",
    "class_definition", "class_declaration",
    "interface_declaration", "struct_definition", "struct_item",
    "struct_declaration", "enum_item", "enum_declaration",
    "trait_item", "impl_item", "type_declaration", "type_alias_declaration",
    "record_declaration", "delegate_declaration", "namespace_declaration",
    "module_declaration",
}
_TS_CALL_TYPES = {
    "call_expression", "call", "method_invocation", "invocation_expression",
    "method_call", "function_call_expression",
}


def _ts_language(lang: str):
    if lang in _TS_LANGS:
        return _TS_LANGS[lang]
    if lang in _TS_FAILED or Language is None:
        return None
    info = _TS_MODULES.get(lang)
    if not info:
        return None
    mod_name, fn_name = info
    try:
        mod = importlib.import_module(mod_name)
        ptr = getattr(mod, fn_name)()
        try:
            language = Language(ptr)
        except Exception:
            language = Language(ptr, lang)
        _TS_LANGS[lang] = language
        return language
    except Exception:
        _TS_FAILED.add(lang)
        return None


def _ts_parser(lang: str):
    language = _ts_language(lang)
    if language is None:
        return None
    try:
        return Parser(language)
    except Exception:
        return None


def _node_text(node, src: bytes) -> str:
    try:
        return src[node.start_byte:node.end_byte].decode("utf-8", "replace")
    except Exception:
        return ""


def _ts_name(node, src: bytes) -> str:
    try:
        n = node.child_by_field_name("name")
        if n is not None:
            t = _node_text(n, src).strip()
            if t:
                return t
    except Exception:
        pass
    for ch in getattr(node, "children", []) or []:
        if ch.type in ("identifier", "type_identifier", "field_identifier",
                       "property_identifier", "name"):
            t = _node_text(ch, src).strip()
            if t:
                return t
    return ""


def _ts_kind(ntype: str) -> str:
    t = ntype.lower()
    if "constructor" in t:
        return "method"
    if "method" in t:
        return "method"
    if "function" in t or t.endswith("_func"):
        return "function"
    if "class" in t:
        return "class"
    if "interface" in t:
        return "interface"
    return "type"


def _ts_qual(node, src: bytes, name: str) -> str:
    parts = []
    p = node.parent
    while p is not None:
        if p.type in _TS_DEF_TYPES and any(x in p.type for x in
                                          ("class", "interface", "impl",
                                           "namespace", "module", "struct")):
            pn = _ts_name(p, src)
            if pn:
                parts.append(pn)
        p = p.parent
    return ".".join(list(reversed(parts)) + [name])


def _ts_extract(text: str, lang: str):
    parser = _ts_parser(lang)
    if parser is None:
        return [], [], []
    src = text.encode("utf-8")
    try:
        tree = parser.parse(src)
    except Exception:
        return [], [], []
    symbols, calls, imports = [], [], []
    seen = set()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        try:
            ntype = node.type
        except Exception:
            continue

        # 变量声明中的箭头函数：const foo = () => ...
        if ntype in ("variable_declarator", "init_declarator"):
            kids = getattr(node, "children", []) or []
            if any(k.type in ("arrow_function", "function_expression", "function")
                   for k in kids):
                name = _ts_name(node, src)
                if name:
                    line = node.start_point[0] + 1
                    if (name, line) not in seen:
                        seen.add((name, line))
                        symbols.append({
                            "name": name, "qual": name, "kind": "function",
                            "line": line, "end": node.end_point[0] + 1,
                            "sig": _node_text(node, src).splitlines()[0][:160],
                            "doc": "",
                        })
        elif ntype in _TS_DEF_TYPES:
            name = _ts_name(node, src)
            if name:
                line = node.start_point[0] + 1
                if (name, line) not in seen:
                    seen.add((name, line))
                    doc = ""
                    try:
                        prev = node.prev_sibling
                        if prev is not None and "comment" in prev.type:
                            doc = _node_text(prev, src).strip().splitlines()[-1][:160]
                    except Exception:
                        pass
                    first = (_node_text(node, src).splitlines() or [""])[0].strip()
                    symbols.append({
                        "name": name, "qual": _ts_qual(node, src, name),
                        "kind": _ts_kind(ntype), "line": line,
                        "end": node.end_point[0] + 1,
                        "sig": first[:160], "doc": doc,
                    })

        if ntype in _TS_CALL_TYPES:
            fn = None
            for field in ("function", "method", "name"):
                try:
                    fn = node.child_by_field_name(field)
                except Exception:
                    fn = None
                if fn is not None:
                    break
            if fn is not None:
                callee = _node_text(fn, src).strip()[:120]
                caller = "<module>"
                p = node.parent
                while p is not None:
                    if p.type in _TS_DEF_TYPES:
                        pn = _ts_name(p, src)
                        if pn:
                            caller = _ts_qual(p, src, pn)
                            break
                    p = p.parent
                if callee:
                    calls.append({"caller": caller, "callee": callee,
                                  "line": node.start_point[0] + 1})

        if "import" in ntype or ntype in ("use_declaration", "using_directive",
                                           "include_directive"):
            line = _node_text(node, src).strip().splitlines()[0][:200]
            if line:
                imports.append(line)

        try:
            stack.extend(reversed(node.children))
        except Exception:
            pass
    return symbols, calls, imports


def _callee_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        cur = node.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


class _PyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.symbols = []
        self.calls = []
        self.imports = []

    def _add(self, node, kind):
        qual = ".".join(self.stack + [node.name])
        doc = (ast.get_docstring(node) or "").strip().splitlines()
        doc = doc[0][:160] if doc else ""
        if kind == "class":
            sig = f"class {node.name}"
        elif kind == "method":
            sig = f"def {node.name}{_fmt_args(node)}"
        else:
            sig = f"def {node.name}{_fmt_args(node)}"
        self.symbols.append({
            "name": node.name, "qual": qual, "kind": kind,
            "line": getattr(node, "lineno", 0), "sig": sig, "doc": doc,
            "end": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
        })

    def visit_Import(self, node):
        for a in node.names:
            self.imports.append(a.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        if mod:
            self.imports.append(mod + ("." + node.names[0].name if node.names else ""))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._add(node, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        kind = "method" if self.stack else "function"
        self._add(node, kind)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        callee = _callee_name(node.func)
        if callee:
            self.calls.append({
                "caller": ".".join(self.stack) or "<module>",
                "callee": callee,
                "line": getattr(node, "lineno", 0),
            })
        self.generic_visit(node)


def _extract_python(text: str):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _extract_regex(text, "python")
    v = _PyVisitor()
    v.visit(tree)
    # 去重（同一 qual 只保留首次）
    seen, symbols = set(), []
    for s in v.symbols:
        key = (s["qual"], s["line"])
        if key not in seen:
            seen.add(key)
            symbols.append(s)
    return symbols, v.calls, v.imports


def _extract_regex(text: str, lang: str):
    specs = list(_REGEX_SPECS.get(lang)
                 or (_REGEX_SPECS.get("ts", []) if lang == "tsx" else []))
    if lang == "python":   # Python 语法损坏时的兜底
        specs = [
            (re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"), "function", 1),
            (re.compile(r"^\s*class\s+(\w+)"), "class", 1),
        ]
    symbols = []
    seen = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        for rx, kind, group in specs:
            m = rx.search(line)
            if not m:
                continue
            name = m.group(group)
            if (not name or name.lower() in _NON_SYMBOL_NAMES
                    or (name, lineno) in seen):
                continue
            seen.add((name, lineno))
            symbols.append({
                "name": name, "qual": name, "kind": kind, "line": lineno,
                "sig": line.strip()[:160], "doc": "",
                "end": 0,
            })
    return symbols, [], []


def _extract(path: str, lang: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if lang == "python":
        return _extract_python(text)
    if lang in _TS_MODULES:
        try:
            syms, calls, imports = _ts_extract(text, lang)
            if syms:
                return syms, calls, imports
        except Exception:
            pass
    return _extract_regex(text, lang)


# ================= 索引持久化 =================
def _load_index_from_disk():
    try:
        mtime = os.path.getmtime(_INDEX_PATH)
    except OSError:
        return None
    if _CACHE["data"] is not None and _CACHE["mtime"] == mtime:
        return _CACHE["data"]
    try:
        with open(_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("version") != _INDEX_VERSION:
        return None
    _CACHE["data"] = data
    _CACHE["mtime"] = mtime
    return data


def _save_index(data: dict):
    os.makedirs(_INDEX_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_INDEX_DIR, prefix="index.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _INDEX_PATH)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
    _CACHE["data"] = data
    try:
        _CACHE["mtime"] = os.path.getmtime(_INDEX_PATH)
    except OSError:
        _CACHE["mtime"] = None


def _ensure_index(force: bool = False) -> dict:
    """增量构建/刷新索引，返回索引数据。"""
    with _INDEX_LOCK:
        old = None if force else _load_index_from_disk()
        if old is None and not force and os.path.exists(_INDEX_PATH):
            # 版本不符或损坏
            old = None
        data = old or {"version": _INDEX_VERSION, "generated": 0, "files": {}}
        if not old:
            data["files"] = {}
        files = data["files"]
        changed = removed = symbols = 0
        seen = set()
        t0 = time.time()
        for full, lang, st, _ in _iter_files(WORKSPACE_DIR):
            rel = _rel(full)
            seen.add(rel)
            prev = files.get(rel)
            if (prev and prev.get("mtime") == st.st_mtime
                    and prev.get("size") == st.st_size and prev.get("lang") == lang):
                symbols += len(prev.get("symbols") or [])
                continue
            try:
                syms, calls, imports = _extract(full, lang)
            except Exception:
                syms, calls, imports = [], [], []
            files[rel] = {
                "mtime": st.st_mtime, "size": st.st_size, "lang": lang,
                "symbols": syms, "calls": calls, "imports": imports,
            }
            changed += 1
            symbols += len(syms)
        for rel in list(files.keys()):
            if rel not in seen:
                files.pop(rel, None)
                removed += 1
        data["generated"] = time.time()
        data["version"] = _INDEX_VERSION
        _save_index(data)
        return {
            "data": data, "files": len(files), "symbols": symbols,
            "changed": changed, "removed": removed,
            "elapsed": round(time.time() - t0, 2),
        }


def _get_index():
    cfg = _cfg()
    if not cfg.get("enabled", True):
        return None
    # 5 秒内复用，避免一次任务里每个查询都扫盘
    try:
        fresh = time.time() - os.path.getmtime(_INDEX_PATH) < 5
    except OSError:
        fresh = False
    if fresh and _load_index_from_disk() is not None:
        return _CACHE["data"]
    return _ensure_index(force=False)["data"]


def _iter_symbols(data, prefix: str):
    for rel, meta in data.get("files", {}).items():
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        for s in meta.get("symbols") or []:
            yield rel, s


# ================= 工具 =================
@tool(
    "【仓库索引】增量构建/刷新 WORKSPACE_DIR 的符号索引（.agent_index/index.json）。"
    "首次使用或代码改动后调用；search_symbol/repo_map 会自动建索引。",
    path="可选：只提示范围（索引始终覆盖整个工作区）",
    force="true=全量重建",
    _tier="read", _read_only=False,
)
def index_repo(path: str = "", force: bool = False) -> str:
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    stats = _ensure_index(force=bool(force))
    return (f"✅ 索引完成：文件 {stats['files']}，符号 {stats['symbols']}，"
            f"更新 {stats['changed']}，删除 {stats['removed']}，"
            f"耗时 {stats['elapsed']}s\n索引位置: {_INDEX_PATH}")


@tool(
    "【仓库索引】查看仓库结构地图（按文件列出类/函数签名）。"
    "修改/审查代码前优先用它建立全局认识，不要逐个 read_file 全文。",
    path="限定目录或文件（默认整个工作区）",
    max_symbols="最多返回的符号数（默认 150）",
    _tier="read", _read_only=True,
)
def repo_map(path: str = "", max_symbols: int = 150) -> str:
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        max_symbols = max(1, int(max_symbols))
    except Exception:
        max_symbols = 150
    groups = {}
    total = 0
    for rel, s in _iter_symbols(data, prefix):
        groups.setdefault(rel, []).append(s)
        total += 1
    if not groups:
        return "索引为空：工作区没有可识别源码，或路径过滤后无结果。"
    lines = [f"[repo map] 文件 {len(groups)}，符号 {total}（最多显示 {max_symbols}）"]
    shown = 0
    for rel in sorted(groups):
        lines.append(f"\n{rel}")
        for s in sorted(groups[rel], key=lambda x: (x.get("line", 0), x.get("qual", ""))):
            if shown >= max_symbols:
                break
            indent = "  " * max(0, s.get("qual", "").count("."))
            lines.append(f"  L{s.get('line', 0)} {indent}{s.get('sig', s.get('name', ''))}")
            if s.get("doc"):
                lines.append(f"      # {s['doc']}")
            shown += 1
        if shown >= max_symbols:
            break
    if shown < total:
        lines.append(f"\n... 还有 {total - shown} 个符号，可用 search_symbol 精确定位")
    return _clip("\n".join(lines))


@tool(
    "【仓库索引】按名称搜索符号（类/函数/方法）。返回 文件:行号 + 签名 + 文档首行。",
    query="符号名或片段（如 run_agent、Parser）",
    kind="可选类型过滤：class/function/method/interface/type/...",
    path="限定目录（默认整个工作区）",
    limit="最多返回条数（默认 30）",
    _tier="read", _read_only=True,
)
def search_symbol(query: str, kind: str = "", path: str = "", limit: int = 30) -> str:
    q = (query or "").strip()
    if not q:
        return "query 不能为空。"
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        limit = max(1, min(200, int(limit)))
    except Exception:
        limit = 30
    ql = q.lower()
    kind = (kind or "").strip().lower()
    hits = []
    for rel, s in _iter_symbols(data, prefix):
        if kind and s.get("kind", "").lower() != kind:
            continue
        name = s.get("name", "")
        qual = s.get("qual", "")
        if ql in name.lower() or ql in qual.lower():
            hits.append((rel, s))
    if not hits:
        return (f"未找到符号 '{q}'。可先用 repo_map(path=...) 看结构，"
                f"或 index_repo(force=true) 刷新索引。")
    hits.sort(key=lambda x: (x[1].get("name", "").lower() != ql, x[0], x[1].get("line", 0)))
    lines = [f"[search_symbol] '{q}' 命中 {len(hits)} 个（显示前 {min(limit, len(hits))}）"]
    for rel, s in hits[:limit]:
        doc = f"  # {s['doc']}" if s.get("doc") else ""
        lines.append(f"{rel}:{s.get('line', 0)}  [{s.get('kind', '')}] "
                     f"{s.get('qual', s.get('name', ''))} {s.get('sig', '')}{doc}")
    return _clip("\n".join(lines))


def _caller_location(data, rel, caller):
    meta = data.get("files", {}).get(rel, {})
    for s in meta.get("symbols") or []:
        if s.get("qual") == caller:
            return s.get("line", 0)
    return 0


@tool(
    "【仓库索引】查找调用点。Python 用 AST 调用图精确匹配；其它语言退化为文本搜索。",
    name="被调用的函数/方法名（可带限定名，如 safety.trim_history）",
    path="限定目录（默认整个工作区）",
    limit="最多返回条数（默认 40）",
    _tier="read", _read_only=True,
)
def find_callers(name: str, path: str = "", limit: int = 40) -> str:
    n = (name or "").strip()
    if not n:
        return "name 不能为空。"
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        limit = max(1, min(200, int(limit)))
    except Exception:
        limit = 40

    hits = []
    for rel, meta in data.get("files", {}).items():
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        for call in meta.get("calls") or []:
            callee = call.get("callee", "")
            if callee == n or callee.endswith("." + n):
                hits.append((rel, _caller_location(data, rel, call.get("caller", "")),
                             call.get("caller", ""), call.get("line", 0)))
    if hits:
        lines = [f"[find_callers] '{n}' 命中 {len(hits)} 处（显示前 {min(limit, len(hits))}）"]
        seen = set()
        for rel, sym_line, caller, call_line in hits:
            key = (rel, sym_line, caller, call_line)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{rel}:{call_line}  in {caller} (L{sym_line})")
            if len(seen) >= limit:
                break
        return _clip("\n".join(lines))

    # 非 Python 或调用图无结果：文本兜底
    rx = re.compile(r"(?<![\w.])" + re.escape(n.split(".")[-1]) + r"\s*\(")
    def_rx = re.compile(r"^\s*(?:async\s+)?(?:def|function|class|func|fn|void|public|private|protected|static)\b")
    found = []
    for rel, meta in data.get("files", {}).items():
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        full = os.path.join(WORKSPACE_DIR, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if rx.search(line) and not def_rx.search(line):
                        found.append((rel, lineno, line.strip()[:160]))
                        if len(found) >= limit:
                            break
        except OSError:
            continue
        if len(found) >= limit:
            break
    if not found:
        return f"未找到 '{n}' 的调用点。可尝试只传函数名，或先 index_repo(force=true)。"
    lines = [f"[find_callers] '{n}' 文本兜底命中 {len(found)} 处"]
    lines += [f"{rel}:{ln}  {code}" for rel, ln, code in found[:limit]]
    return _clip("\n".join(lines))


def _read_file_lines(rel: str):
    try:
        with open(os.path.join(WORKSPACE_DIR, rel), "r",
                  encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


@tool(
    "【仓库索引】取符号完整实现（类/函数/方法源码块）。"
    "比 read_file 全文更精准，适合修改前确认代码。",
    name="符号名（可带限定名，如 run_agent 或 RunAgent._finalize）",
    path="限定目录（默认整个工作区）",
    limit="最多返回匹配数（默认 3）",
    _tier="read", _read_only=True,
)
def get_symbol(name: str, path: str = "", limit: int = 3) -> str:
    q = (name or "").strip()
    if not q:
        return "name 不能为空。"
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        limit = max(1, min(10, int(limit)))
    except Exception:
        limit = 3
    matches = []
    ql = q.lower()
    for rel, s in _iter_symbols(data, prefix):
        if s.get("name", "").lower() == ql or s.get("qual", "").lower() == ql:
            matches.append((rel, s))
    if not matches:
        # 退化为包含匹配，给出候选
        hits = [(rel, s) for rel, s in _iter_symbols(data, prefix)
                if ql in s.get("name", "").lower()]
        if not hits:
            return f"未找到符号 '{q}'。可用 search_symbol 模糊搜索。"
        cand = "\n".join(f"{r}:{s.get('line', 0)} {s.get('qual', '')}"
                         for r, s in hits[:10])
        return f"未找到精确符号 '{q}'，候选：\n{cand}\n请用完整符号名重试。"

    out = [f"[get_symbol] '{q}' 命中 {len(matches)} 个（显示前 {min(limit, len(matches))}）"]
    for rel, s in matches[:limit]:
        lines = _read_file_lines(rel)
        start = int(s.get("line") or 1)
        end = int(s.get("end") or 0)
        if end <= start:
            # 正则提取的符号没有 end：取下一个符号行之前，最多 160 行
            meta = data.get("files", {}).get(rel, {})
            nxt = [x.get("line", 0) for x in meta.get("symbols") or []
                   if x.get("line", 0) > start]
            end = (min(nxt) - 1) if nxt else start + 160
            end = min(end, start + 160)
        end = min(end, len(lines), start + 400)
        block = "\n".join(f"{i:5d}| {lines[i-1]}" for i in range(start, end + 1)
                          if 0 < i <= len(lines))
        out.append(f"\n=== {rel}:{start}-{end} [{s.get('kind', '')}] "
                   f"{s.get('qual', s.get('name', ''))} ===\n{block}")
    return _clip("\n".join(out))


@tool(
    "【仓库索引】查找标识符的所有引用位置（定义行除外）。"
    "find_callers 看调用关系，find_references 看变量/类/函数的所有使用点。",
    name="标识符名（如 human_approve、CONFIG）",
    path="限定目录（默认整个工作区）",
    limit="最多返回条数（默认 50）",
    _tier="read", _read_only=True,
)
def find_references(name: str, path: str = "", limit: int = 50) -> str:
    n = (name or "").strip()
    if not n:
        return "name 不能为空。"
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        limit = max(1, min(300, int(limit)))
    except Exception:
        limit = 50
    short = n.split(".")[-1]
    rx = re.compile(r"(?<![\w.])" + re.escape(short) + r"(?![\w])")
    def_lines = {}
    for rel, s in _iter_symbols(data, prefix):
        if s.get("name") == short or s.get("qual") == n:
            def_lines.setdefault(rel, set()).add(int(s.get("line") or 0))

    hits = []
    for rel in sorted(data.get("files", {})):
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        for lineno, line in enumerate(_read_file_lines(rel), 1):
            if not rx.search(line):
                continue
            if lineno in def_lines.get(rel, set()):
                continue
            hits.append((rel, lineno, line.strip()[:180]))
            if len(hits) >= limit:
                break
        if len(hits) >= limit:
            break
    if not hits:
        return f"未找到 '{n}' 的引用（定义行已排除）。"
    lines = [f"[find_references] '{n}' 命中 {len(hits)} 处（显示前 {limit}）"]
    lines += [f"{rel}:{ln}  {code}" for rel, ln, code in hits[:limit]]
    return _clip("\n".join(lines))


@tool(
    "【仓库索引】影响面分析：沿调用图反向查找调用者，默认向上 3 层。"
    "修改函数前用它判断会影响哪些上层逻辑。",
    name="被分析函数名（可带限定名）",
    depth="向上追溯层数（默认 3，最大 6）",
    limit="最多返回调用者数（默认 60）",
    _tier="read", _read_only=True,
)
def impact_analysis(name: str, depth: int = 3, limit: int = 60) -> str:
    n = (name or "").strip()
    if not n:
        return "name 不能为空。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        depth = max(1, min(6, int(depth)))
        limit = max(1, min(300, int(limit)))
    except Exception:
        depth, limit = 3, 60
    short = n.split(".")[-1]

    def _loc(rel, qual):
        meta = data.get("files", {}).get(rel, {})
        for s in meta.get("symbols") or []:
            if s.get("qual") == qual:
                return s.get("line", 0)
        return 0

    levels = {0: {n, short}}
    seen = {n, short}
    parent = {}
    frontier = set(levels[0])
    edge = []  # (callee_short, caller_qual, rel, call_line)
    for rel, meta in data.get("files", {}).items():
        for call in meta.get("calls") or []:
            callee = str(call.get("callee") or "").split(".")[-1]
            edge.append((callee, call.get("caller") or "", rel,
                         call.get("line", 0)))

    for d in range(1, depth + 1):
        nxt = set()
        for callee, caller, rel, call_line in edge:
            if callee not in frontier:
                continue
            if caller in seen or not caller or caller == "<module>":
                continue
            nxt.add(caller)
            parent[caller] = (callee, rel, call_line)
            if len(seen) >= limit + len(levels[0]):
                break
        if not nxt:
            break
        levels[d] = nxt
        seen |= nxt
        frontier = nxt
        if len(seen) > limit:
            break

    if len(levels) <= 1:
        refs = find_references(n, limit=min(limit, 40))
        return f"[impact_analysis] 未发现调用者，退回文本引用：\n{refs}"

    lines = [f"[impact_analysis] '{n}' 向上 {len(levels)-1} 层，"
             f"调用者 {len(seen)-2} 个"]
    for d in sorted(levels):
        if d == 0:
            continue
        for caller in sorted(levels[d]):
            callee, rel, call_line = parent.get(caller, ("?", "?", 0))
            sym_line = _loc(rel, caller)
            lines.append(f"L{d}: {rel}:{sym_line} {caller} "
                         f"-> {callee} (call@L{call_line})")
            if len(lines) > limit + 5:
                lines.append("... [影响面结果已截断]")
                return _clip("\n".join(lines))
    return _clip("\n".join(lines))


@tool(
    "【仓库索引】静态诊断：Python 编译 + JSON 语法 + ruff 静态检查（若已安装 ruff）。",
    path="限定目录或文件（默认整个工作区）",
    limit="最多返回错误数（默认 50）",
    lint="是否运行 ruff 静态检查（默认 true）",
    _tier="read", _read_only=True,
)
def diagnostics(path: str = "", limit: int = 50, lint: bool = True) -> str:
    prefix = _scope_prefix(path)
    if prefix is None:
        return "安全拦截：路径不在工作区内。"
    data = _get_index()
    if data is None:
        return "仓库索引已禁用（config.repo_index.enabled=false）。"
    try:
        limit = max(1, min(500, int(limit)))
    except Exception:
        limit = 50
    errors, checked = [], 0
    for rel, meta in sorted(data.get("files", {}).items()):
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        lang = meta.get("lang")
        full = os.path.join(WORKSPACE_DIR, rel)
        checked += 1
        if lang == "python":
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    compile(f.read(), rel, "exec")
            except SyntaxError as e:
                errors.append(f"{rel}:{e.lineno}  SyntaxError: {e.msg}")
            except Exception as e:
                errors.append(f"{rel}:  {type(e).__name__}: {e}")
        elif rel.endswith(".json"):
            try:
                with open(full, "r", encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                errors.append(f"{rel}:  JSON 错误: {e}")
        if len(errors) >= limit:
            break

    ruff_used = False
    if lint and len(errors) < limit:
        ruff = shutil.which("ruff")
        if not ruff:
            cand = os.path.join(os.path.dirname(sys.executable), "Scripts", "ruff.exe")
            ruff = cand if os.path.isfile(cand) else ""
        if ruff:
            ruff_used = True
            target = (os.path.join(WORKSPACE_DIR, prefix)
                      if prefix else WORKSPACE_DIR)
            res = _sandbox.run(
                [ruff, "check", target, "--output-format=json", "--no-cache"],
                CONFIG.get("sandbox"), timeout=30, cwd=WORKSPACE_DIR,
                purpose="ruff", network="deny")
            if res.get("error"):
                errors.append(f"[ruff] 运行失败: {res['error']}")
            else:
                try:
                    items = json.loads(res.get("stdout") or "[]")
                except Exception:
                    items = []
                for item in items if isinstance(items, list) else []:
                    fn = str(item.get("filename") or "")
                    try:
                        fn = os.path.relpath(fn, WORKSPACE_DIR).replace("\\", "/")
                    except Exception:
                        pass
                    if prefix and not (fn == prefix or fn.startswith(prefix + "/")):
                        continue
                    loc = item.get("location") or {}
                    code = item.get("code") or ""
                    msg = str(item.get("message") or "")[:200]
                    errors.append(f"{fn}:{loc.get('row', 0)}:{loc.get('column', 0)}  "
                                  f"[ruff {code}] {msg}")
                    if len(errors) >= limit:
                        break
                if res.get("returncode") not in (0, 1) and not items:
                    errors.append(f"[ruff] 退出码 {res.get('returncode')}: "
                                  f"{(res.get('stderr') or '')[-300:]}")

    if not errors:
        extra = " + ruff" if ruff_used else ""
        return f"✅ 诊断通过：检查 {checked} 个文件（compile/JSON{extra}），未发现问题。"
    extra = "，含 ruff" if ruff_used else ""
    return _clip(f"❌ 诊断发现 {len(errors)} 个问题（检查 {checked} 个文件{extra}）：\n"
                 + "\n".join(errors[:limit]))
