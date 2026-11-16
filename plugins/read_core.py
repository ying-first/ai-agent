"""plugins/read_core.py — 核心文件只读工具（含 API 签名提取与防御）。"""
import os
import ast
from safety import tool, WORKSPACE_DIR

CORE_FILES = {"agent": "agent.py", "safety": "safety.py", "config": "config.json"}
DEFAULT_PREVIEW_CHARS = 1500
MAX_OUTPUT_CHARS = 4000

def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[输出已截断，原长 {len(text)} 字符]"

def _resolve_core_path(name):
    if not isinstance(name, str): return None, "name 必须是字符串"
    key = name.strip().lower()
    rel = CORE_FILES.get(key)
    if rel is None: return None, f"不支持的文件: {name}"
    root = os.path.realpath(WORKSPACE_DIR)
    full = os.path.join(WORKSPACE_DIR, rel)
    real = os.path.realpath(full)
    if real != os.path.join(root, rel):
        return None, "安全拦截：路径越界"
    return real, None

@tool(
    "【只读】读取核心文件。🚨写代码前查阅接口请务必传 mode='api'，禁止用 mode='src' 盲扫源码！\n"
    "🚨 如果遇到安全拦截（如 AST 报错 subprocess），说明系统严禁此操作，请**立刻停止翻源码尝试绕过**，直接向人类报告请求修改白名单！\n"
    "mode='api': 返回文件里所有类/函数的签名和注释（轻量，强烈推荐）\n"
    "mode='src': 返回源码片段（必须配合 find 使用，否则极大概率失败）",
    name="核心文件名：agent | safety | config",
    mode="模式：api(查签名/推荐) | src(读源码)",
    find="搜索关键词（强烈建议在 src 模式下使用）",
)
def read_core_file(name: str, mode: str = "api", find: str = "") -> str:
    path, err = _resolve_core_path(name)
    if err: return err
    if not os.path.isfile(path): return f"文件不存在: {os.path.basename(path)}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return f"读取失败: {e}"

    rel = CORE_FILES[name.strip().lower()]
    
    # ===== 模式：API 签名提取 =====
    if mode == "api":
        if not rel.endswith(".py"):
            return "config 文件不支持 api 模式，请用 src 模式并搭配 find 读取。"
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return "AST 解析失败，文件可能有语法错误。"
        
        out = [f"[{rel} | API 签名清单（只读）]"]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                sig = f"def {node.name}({', '.join(args)})"
                doc = ast.get_docstring(node) or ""
                doc_preview = doc.strip().split("\n")[0][:100] if doc else ""
                out.append(f"- L{node.lineno}: {sig}  {doc_preview}")
        return _clip("\n".join(out))

    # ===== 模式：源码搜索 =====
    total = len(text)
    lines = text.splitlines()

    if find:
        kw = find.strip()
        hits = []
        for i, ln in enumerate(lines):
            if kw in ln:
                lo, hi = max(0, i - 3), min(len(lines), i + 4)
                hits.append((i + 1, "\n".join(f"{j+1:5d}| {lines[j]}" for j in range(lo, hi))))
                if len(hits) >= 20: break
        if not hits: 
            return f"[{rel}] 未找到 '{kw}'。请更换更短、更核心的关键词重试，不要盲目翻页！"
        
        out = [f"[{rel} | 搜索 '{kw}' | 命中 {len(hits)} 处]"]
        for lineno, block in hits:
            out.append(f"\n--- 命中于 L{lineno} ---\n{block}")
        return _clip("\n".join(out))

    # 未传 find 时，默认返回预览（不支持分块翻页，请用 find 精准定位）
    return f"[{rel} | 总长 {total} 字符 | 请使用 mode='api' 或 find='关键词' 检索]\n{text[:DEFAULT_PREVIEW_CHARS]}\n...(已截断)"