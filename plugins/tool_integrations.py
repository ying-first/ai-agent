"""plugins/tool_integrations.py — 外部工具链集成。

集成：grep 搜索、git 只读查询、测试运行、HTTP 抓取、工具链体检。
所有子进程统一走 safety 的 sandbox（超时/内存/进程数限制 + 密钥剥离）。
"""
import fnmatch
import html
import importlib.util
import ipaddress
import os
import re
import shutil
import socket
import sys
import urllib.parse
import urllib.request

from safety import WORKSPACE_DIR, CONFIG, tool, is_safe_path
import sandbox as _sandbox

_EXCLUDE_DIRS = {
    ".git", "__pycache__", ".agent_index", ".agent_checkpoints", ".experiences",
    ".traces", ".venv", "venv", "node_modules", "dist", "build", "bin", "obj",
    ".vs", ".claude",
}
_MAX_OUT = 6000


def _clip(text, limit=_MAX_OUT):
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + f"\n...[输出已截断，原长 {len(s)} 字符]"


def _which(name):
    return shutil.which(name) or ""


def _version(exe, args=None, timeout=8):
    if not exe:
        return ""
    res = _sandbox.run([exe] + (args or ["--version"]),
                       CONFIG.get("sandbox"), timeout=timeout,
                       cwd=WORKSPACE_DIR, network="deny",
                       purpose="version")
    return ((res.get("stdout") or "") + (res.get("stderr") or "")).strip().splitlines()[0:1]


@tool(
    "【工具集成】查看已检测到的外部工具链：git、ruff、node、tree-sitter、tiktoken、"
    "AppContainer 沙箱、MCP 等。",
    _tier="read", _read_only=True,
)
def toolchain_status() -> str:
    lines = [f"Python {sys.version.split()[0]} @ {sys.executable}"]
    for name, exe in (("git", _which("git")), ("ruff", _which("ruff")),
                      ("node", _which("node")), ("npm", _which("npm")),
                      ("rg", _which("rg")), ("docker", _which("docker")),
                      ("wsl", _which("wsl"))):
        if not exe:
            lines.append(f"- {name}: 未安装")
            continue
        ver = _version(exe)
        lines.append(f"- {name}: {exe}" + (f" | {ver[0][:100]}" if ver else ""))
    try:
        import tiktoken
        lines.append(f"- tiktoken: 已安装（cl100k_base 估算）")
    except Exception:
        lines.append("- tiktoken: 未安装（回退启发式 token）")
    try:
        from tree_sitter import Language  # noqa
        langs = []
        for mod in ("tree_sitter_python", "tree_sitter_javascript",
                    "tree_sitter_typescript", "tree_sitter_c_sharp",
                    "tree_sitter_go", "tree_sitter_rust", "tree_sitter_java",
                    "tree_sitter_c", "tree_sitter_cpp"):
            try:
                importlib.import_module(mod)
                langs.append(mod.replace("tree_sitter_", ""))
            except Exception:
                pass
        lines.append(f"- tree-sitter: 已安装（{', '.join(langs) or '无语言包'}）")
    except Exception:
        lines.append("- tree-sitter: 未安装（回退 AST/正则）")
    st = _sandbox.appcontainer_status(dict(CONFIG.get("sandbox") or {},
                                            _workspace=WORKSPACE_DIR))
    lines.append(f"- AppContainer: {'可用' if st.get('ok') else '不可用'}"
                 f" profile={st.get('profile')}"
                 + (f" error={st.get('error')}" if st.get("error") else ""))
    lines.append(f"- 沙箱后端: run_python_file=AppContainer(镜像+断网), "
                 f"execute_system_command=Job Object")
    try:
        mcp = CONFIG.get("mcp", {}) or {}
        lines.append(f"- MCP: enabled={mcp.get('enabled')} "
                     f"servers={len(mcp.get('servers') or {})}")
    except Exception:
        pass
    return "\n".join(lines)


@tool(
    "【工具集成】全文搜索（优先 rg，否则 Python 扫描），返回 文件:行号:内容。",
    pattern="搜索模式",
    path="限定目录/文件（默认工作区）",
    glob="文件名通配符，如 *.py",
    regex="true=正则，false=子串",
    limit="最多返回条数（默认 50）",
    _tier="read", _read_only=True,
)
def grep_code(pattern: str, path: str = "", glob: str = "",
              regex: bool = True, limit: int = 50) -> str:
    pat = (pattern or "").strip()
    if not pat:
        return "pattern 不能为空。"
    root = os.path.realpath(path or WORKSPACE_DIR)
    ws = os.path.realpath(WORKSPACE_DIR)
    if not (root == ws or root.startswith(ws + os.sep)):
        return "安全拦截：搜索路径不在工作区内。"
    try:
        limit = max(1, min(300, int(limit)))
    except Exception:
        limit = 50
    try:
        rx = re.compile(pat if regex else re.escape(pat))
    except re.error as e:
        return f"正则错误: {e}"

    rg = _which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never",
               "--max-count", str(limit)]
        if glob:
            cmd += ["--glob", glob]
        if not regex:
            cmd += ["--fixed-strings"]
        cmd += [pat, root]
        res = _sandbox.run(cmd, CONFIG.get("sandbox"), timeout=20,
                           cwd=WORKSPACE_DIR, network="deny", purpose="rg")
        if not res.get("error"):
            body = (res.get("stdout") or "").strip()
            return _clip(body or "未找到匹配。")

    hits = []
    if os.path.isfile(root):
        files = [root]
    else:
        files = []
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in _EXCLUDE_DIRS
                      and not d.startswith(".")]
            for fn in fns:
                if glob and not fnmatch.fnmatch(fn, glob):
                    continue
                if fn.startswith((".", "_")) or ".bak_" in fn:
                    continue
                files.append(os.path.join(dp, fn))
    for full in files:
        try:
            if os.path.getsize(full) > 2 * 1024 * 1024:
                continue
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if rx.search(line):
                        rel = os.path.relpath(full, WORKSPACE_DIR).replace("\\", "/")
                        hits.append(f"{rel}:{lineno}: {line.rstrip()[:240]}")
                        if len(hits) >= limit:
                            break
        except OSError:
            continue
        if len(hits) >= limit:
            break
    if not hits:
        return f"未找到匹配 '{pat}'。"
    return _clip("\n".join(hits))


@tool(
    "【工具集成】git 只读查询：status / diff / log / branch。",
    action="status | diff | log | branch",
    limit="log 返回条数（默认 20）",
    _tier="read", _read_only=True,
)
def git_query(action: str = "status", limit: int = 20) -> str:
    git = _which("git")
    if not git:
        return "未检测到 git。"
    act = (action or "status").strip().lower()
    try:
        limit = max(1, min(200, int(limit)))
    except Exception:
        limit = 20
    cache = os.path.join(WORKSPACE_DIR, ".git", "index")
    if act == "status":
        cmd = [git, "status", "--short", "--branch"]
    elif act == "diff":
        cmd = [git, "diff", "--stat"]
    elif act == "log":
        cmd = [git, "log", "--oneline", "-n", str(limit)]
    elif act == "branch":
        cmd = [git, "branch", "--show-current"]
    else:
        return "action 仅支持 status / diff / log / branch。"
    if not os.path.isdir(os.path.join(WORKSPACE_DIR, ".git")):
        return "当前工作区不是 git 仓库。"
    res = _sandbox.run(cmd, CONFIG.get("sandbox"), timeout=20,
                       cwd=WORKSPACE_DIR, network="deny", purpose="git")
    if res.get("error"):
        return f"git 执行失败: {res['error']}"
    body = (res.get("stdout") or "").strip()
    return _clip(body or f"（git {act} 无输出）")


@tool(
    "【工具集成】运行测试：优先 pytest，否则 unittest discover。",
    target="测试文件/目录（默认当前工作区）",
    timeout="超时秒数（默认 60）",
    _tier="execute", _read_only=False,
)
def run_tests(target: str = "", timeout: int = 60) -> str:
    try:
        timeout = max(5, min(300, int(timeout)))
    except Exception:
        timeout = 60
    target = (target or "").strip()
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        cmd.append(target or ".")
    elif target.endswith(".py") and os.path.isfile(target):
        cmd = [sys.executable, target]
    else:
        cmd = [sys.executable, "-m", "unittest", "discover",
               "-s", target or ".", "-p", "test_*.py"]
    res = _sandbox.run(cmd, CONFIG.get("sandbox"), timeout=timeout,
                       cwd=WORKSPACE_DIR, purpose="run_tests",
                       network="allow")
    if res.get("timeout"):
        return f"❌ 测试超时（{timeout}s）。"
    if res.get("error"):
        return f"❌ 测试执行失败: {res['error']}"
    body = (res.get("stdout") or "") + (res.get("stderr") or "")
    if res.get("returncode") == 0:
        return _clip("✅ 测试通过：\n" + body[-5000:])
    return _clip("❌ 测试失败：\n" + body[-5000:])


def _safe_public_url(url: str):
    try:
        p = urllib.parse.urlsplit(url)
    except Exception:
        return "URL 解析失败"
    if p.scheme not in ("http", "https"):
        return "仅支持 http/https"
    host = p.hostname or ""
    if not host or host.lower() in ("localhost", "localhost.localdomain"):
        return "禁止访问本机地址"
    try:
        infos = socket.getaddrinfo(host, p.port or 80)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return f"禁止访问内网/保留地址: {ip}"
    except Exception as e:
        return f"DNS 解析失败: {e}"
    return ""


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


@tool(
    "【工具集成】HTTP GET 抓取网页文本（需 network 权限；禁止内网/本机地址）。",
    url="http(s) URL",
    max_chars="返回最大字符数（默认 4000）",
    _tier="network", _read_only=False,
)
def web_fetch(url: str, max_chars: int = 4000) -> str:
    err = _safe_public_url(url)
    if err:
        return f"⛔ {err}"
    try:
        max_chars = max(200, min(20000, int(max_chars)))
    except Exception:
        max_chars = 4000
    req = urllib.request.Request(url, headers={
        "User-Agent": "AI-Agent/14 (+local tool integration)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(300 * 1024)
    except Exception as e:
        return f"❌ 抓取失败: {type(e).__name__}: {e}"
    text = raw.decode("utf-8", "replace")
    if "html" in ctype.lower() or text.lstrip().lower().startswith("<!doctype"):
        text = _strip_html(text)
    return _clip(text, max_chars)
