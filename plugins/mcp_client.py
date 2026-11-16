"""plugins/mcp_client.py — MCP (Model Context Protocol) stdio 客户端。

特性：
- 纯标准库实现 JSON-RPC 2.0 over stdio，同时兼容换行分隔 JSON 与 LSP 风格
  Content-Length 两种传输；
- 启动时按 config.mcp.servers 自动连接、tools/list 发现工具并注册为
  mcp_<server>_<tool>，走统一权限分级（server.tier，默认 execute）；
- mcp_status / mcp_reload 管理工具；
- 断线、超时、单个 server 启动失败都不会影响主 Agent。

config.json 示例：
  "mcp": {
    "enabled": true,
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "tier": "read",
        "enabled": true
      }
    }
  }
"""
import atexit
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time

from safety import (CONFIG, WORKSPACE_DIR, tool, register_dynamic_tool,
                    unregister_dynamic_tool)

_PLUGIN_PATH = os.path.realpath(__file__)
_SERVERS = {}
_LOCK = threading.RLock()
_MAX_TEXT = 6000


def _mcp_cfg() -> dict:
    return CONFIG.get("mcp", {}) or {}


def _clip(text, limit: int = _MAX_TEXT) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + f"\n...[MCP 输出已截断，原长 {len(s)} 字符]"


class MCPError(RuntimeError):
    pass


class StdioMCPClient:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg or {}
        self.proc = None
        self._pending = {}
        self._id = 0
        self._lock = threading.RLock()
        self._stderr_tail = []
        self._threads = []
        self._closed = False

    # ---------- 进程 / 传输 ----------
    def _build_command(self):
        raw = self.cfg.get("command")
        if not raw:
            raise MCPError("缺少 command")
        if isinstance(raw, (list, tuple)):
            cmd = [str(x) for x in raw]
        else:
            cmd = shlex.split(str(raw), posix=(os.name != "nt"))
        if not cmd:
            raise MCPError("command 为空")
        args = self.cfg.get("args") or []
        if isinstance(args, str):
            args = shlex.split(args, posix=(os.name != "nt"))
        cmd += [str(a) for a in args]

        resolved = shutil.which(cmd[0])
        if resolved:
            cmd[0] = resolved
        if os.name == "nt" and cmd[0].lower().endswith((".cmd", ".bat")):
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            cmd = [comspec, "/c"] + cmd
        return cmd

    def _build_env(self):
        env = dict(os.environ)
        for k, v in (self.cfg.get("env") or {}).items():
            env[str(k)] = os.path.expandvars(str(v))
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def start(self, timeout: float = 20.0):
        cmd = self._build_command()
        cwd = self.cfg.get("cwd") or WORKSPACE_DIR
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1, cwd=cwd,
            env=self._build_env(), creationflags=flags,
        )
        t_out = threading.Thread(target=self._read_stdout, daemon=True,
                                 name=f"mcp-{self.name}-out")
        t_err = threading.Thread(target=self._read_stderr, daemon=True,
                                 name=f"mcp-{self.name}-err")
        t_out.start()
        t_err.start()
        self._threads = [t_out, t_err]

        protocol = str(self.cfg.get("protocol_version")
                       or _mcp_cfg().get("protocol_version")
                       or "2025-06-18")
        result = self.request("initialize", {
            "protocolVersion": protocol,
            "capabilities": {},
            "clientInfo": {"name": "ai-agent", "version": "14"},
        }, timeout=timeout)
        if not isinstance(result, dict):
            raise MCPError("initialize 返回格式异常")
        self.notify("notifications/initialized")

    def _send(self, obj: dict):
        if not self.proc or self.proc.poll() is not None:
            raise MCPError("MCP server 进程已退出")
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()

    def _dispatch(self, raw: str):
        try:
            obj = json.loads(raw)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        # server -> client 请求：只支持 ping，其它返回 method not found
        if "method" in obj and "id" in obj:
            if obj.get("method") == "ping":
                self._send({"jsonrpc": "2.0", "id": obj["id"], "result": {}})
            else:
                self._send({"jsonrpc": "2.0", "id": obj["id"],
                            "error": {"code": -32601, "message": "method not found"}})
            return
        rid = obj.get("id")
        with self._lock:
            slot = self._pending.get(rid)
        if slot is not None:
            slot["msg"] = obj
            slot["event"].set()

    def _read_stdout(self):
        stream = self.proc.stdout
        try:
            while not self._closed:
                line = stream.readline()
                if not line:
                    break
                low = line.strip().lower()
                if low.startswith("content-length:"):
                    try:
                        n = int(line.split(":", 1)[1].strip())
                    except Exception:
                        continue
                    # 吃掉剩余 headers 直到空行
                    while True:
                        h = stream.readline()
                        if not h or not h.strip():
                            break
                    self._dispatch(stream.read(n))
                else:
                    s = line.strip()
                    if s:
                        self._dispatch(s)
        except Exception:
            pass

    def _read_stderr(self):
        try:
            for line in self.proc.stderr:
                with self._lock:
                    self._stderr_tail.append(line.rstrip()[:300])
                    if len(self._stderr_tail) > 20:
                        self._stderr_tail.pop(0)
        except Exception:
            pass

    def _wait_proc_exit(self):
        if not self.proc:
            return
        try:
            rc = self.proc.poll()
            if rc is None:
                rc = self.proc.wait(timeout=2)
            return rc
        except Exception:
            return None

    # ---------- JSON-RPC ----------
    def request(self, method: str, params: dict = None, timeout: float = 30.0):
        with self._lock:
            self._id += 1
            rid = self._id
            slot = {"event": threading.Event(), "msg": None}
            self._pending[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                        "params": params or {}})
            if not slot["event"].wait(timeout):
                raise MCPError(f"{method} 超时（{timeout}s）")
            msg = slot["msg"] or {}
            if "error" in msg:
                err = msg.get("error") or {}
                raise MCPError(err.get("message") or str(err))
            return msg.get("result")
        finally:
            with self._lock:
                self._pending.pop(rid, None)

    def notify(self, method: str, params: dict = None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def list_tools(self, timeout: float = 30.0):
        tools, cursor = [], None
        while True:
            params = {"cursor": cursor} if cursor else {}
            res = self.request("tools/list", params, timeout=timeout)
            res = res or {}
            tools.extend(res.get("tools") or [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict, timeout: float = 60.0):
        return self.request("tools/call",
                            {"name": name, "arguments": arguments or {}},
                            timeout=timeout)

    def close(self):
        self._closed = True
        if not self.proc:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass
        for t in self._threads:
            try:
                t.join(timeout=1)
            except Exception:
                pass

    def stderr_tail(self) -> str:
        with self._lock:
            return "\n".join(self._stderr_tail[-8:])


# ================= 工具注册 =================
def _safe_tool_name(server: str, tool_name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", f"mcp_{server}_{tool_name}")
    if len(base) > 64:
        h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        base = base[:55] + "_" + h
    return base


def _normalize_schema(schema) -> dict:
    if not isinstance(schema, dict):
        schema = {}
    out = dict(schema)
    if out.get("type") != "object":
        out["type"] = "object"
    if not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    out.pop("$schema", None)
    return out


def _format_result(result) -> str:
    result = result or {}
    texts = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            texts.append(str(item))
            continue
        if item.get("type") == "text":
            texts.append(str(item.get("text", "")))
        elif item.get("type") == "resource":
            res = item.get("resource") or {}
            texts.append(f"[resource {res.get('uri', '')}]\n{res.get('text', '')}")
        elif item.get("type") == "image":
            texts.append(f"[image {item.get('mimeType', '')}，已省略二进制内容]")
        else:
            texts.append(json.dumps(item, ensure_ascii=False))
    if result.get("structuredContent") is not None:
        texts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    body = "\n".join(t for t in texts if t) or "(MCP 无文本返回)"
    if result.get("isError"):
        return _clip("❌ MCP 工具返回错误:\n" + body)
    return _clip(body)


def _contains_denied_path(value, deny_parts) -> str:
    """递归检查 MCP 参数中的路径片段；只用于 filesystem 类 server。"""
    if not deny_parts:
        return ""
    if isinstance(value, str):
        comps = [p for p in re.split(r"[\\/]+", value.lower()) if p]
        for comp in comps:
            if comp in deny_parts:
                return comp
        return ""
    if isinstance(value, dict):
        for v in value.values():
            hit = _contains_denied_path(v, deny_parts)
            if hit:
                return hit
    elif isinstance(value, (list, tuple)):
        for v in value:
            hit = _contains_denied_path(v, deny_parts)
            if hit:
                return hit
    return ""


def _register_proxy(server_name: str, client: StdioMCPClient, tool_def: dict,
                    tier: str, deny_parts=None) -> str:
    raw_name = str(tool_def.get("name") or "").strip()
    if not raw_name:
        return ""
    full_name = _safe_tool_name(server_name, raw_name)
    desc = str(tool_def.get("description") or "")
    desc = f"[MCP:{server_name}] {desc}"[:1500]
    schema = _normalize_schema(tool_def.get("inputSchema"))
    timeout = float(_mcp_cfg().get("call_timeout_s", 60) or 60)
    deny = {str(x).lower() for x in (deny_parts or [])}

    def _proxy(**kwargs):
        hit = _contains_denied_path(kwargs, deny)
        if hit:
            return (f"⛔ MCP 安全拦截：参数包含受保护路径片段 `{hit}`，"
                    f"禁止通过 [{server_name}/{raw_name}] 访问。")
        try:
            return _format_result(client.call_tool(raw_name, kwargs, timeout=timeout))
        except Exception as e:
            return f"❌ MCP 调用失败 [{server_name}/{raw_name}]: {e}"

    _proxy.__name__ = full_name
    ok = register_dynamic_tool(
        full_name, desc, schema, _proxy,
        tier=tier, read_only=(tier == "read"), plugin_path=_PLUGIN_PATH,
    )
    return full_name if ok else ""


def _remove_server_entry(name: str):
    entry = _SERVERS.pop(name, None)
    if not entry:
        return
    for full_name in entry.get("registered") or []:
        unregister_dynamic_tool(full_name)
    client = entry.get("client")
    if client:
        try:
            client.close()
        except Exception:
            pass


def _start_server(name: str, cfg: dict) -> dict:
    tier = str(cfg.get("tier") or "execute").lower()
    entry = {"client": None, "registered": [], "error": "", "tier": tier,
             "config": dict(cfg)}
    client = None
    try:
        client = StdioMCPClient(name, cfg)
        client.start(timeout=float(_mcp_cfg().get("startup_timeout_s", 20) or 20))
        tools = client.list_tools(timeout=float(
            _mcp_cfg().get("startup_timeout_s", 20) or 20))
        entry["client"] = client
        entry["tools"] = [t.get("name") for t in tools if isinstance(t, dict)]
        allowed = cfg.get("allowed_tools") or cfg.get("tool_allowlist")
        allow_set = ({str(x) for x in allowed} if allowed else None)
        denied = {str(x) for x in (cfg.get("deny_tools") or [])}
        deny_parts = cfg.get("deny_path_parts") or []
        for t in tools:
            if isinstance(t, dict):
                raw = str(t.get("name") or "")
                if allow_set is not None and raw not in allow_set:
                    continue
                if raw in denied:
                    continue
                full = _register_proxy(name, client, t, tier, deny_parts)
                if full:
                    entry["registered"].append(full)
        # 外部 server 的清单交给统一路由；工具立即加入本会话工作集
        try:
            from safety import activate_tools
            if entry["registered"]:
                activate_tools(entry["registered"])
        except Exception:
            pass
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        if client:
            try:
                client.close()
            except Exception:
                pass
    _SERVERS[name] = entry
    return entry


def _init_servers():
    cfg = _mcp_cfg()
    if not cfg.get("enabled", True):
        return
    # 沙箱冒烟/被沙箱执行的脚本里不再拉起外部 MCP 进程（避免嵌套进程与密钥暴露）
    if os.environ.get("AI_AGENT_SANDBOX") == "1":
        return
    servers = cfg.get("servers") or {}
    if not isinstance(servers, dict):
        return
    for name, scfg in servers.items():
        if not isinstance(scfg, dict) or scfg.get("enabled", True) is False:
            continue
        with _LOCK:
            if name in _SERVERS:
                continue
            _start_server(str(name), scfg)


def _reload_servers(server: str = "") -> str:
    cfg = _mcp_cfg()
    if not cfg.get("enabled", True):
        return "MCP 客户端已禁用（config.mcp.enabled=false）。"
    servers = cfg.get("servers") or {}
    if server:
        if server not in servers and server not in _SERVERS:
            return f"未知 MCP server: {server}"
        targets = [server]
    else:
        # 全量重载：既要重启配置里的，也要清理已经从配置中删掉的
        targets = sorted(set(servers.keys()) | set(_SERVERS.keys()))
    with _LOCK:
        for name in targets:
            _remove_server_entry(name)
        for name in targets:
            if name not in servers:
                continue
            scfg = servers.get(name) or {}
            if not isinstance(scfg, dict) or scfg.get("enabled", True) is False:
                continue
            _start_server(name, scfg)
    return _status_text()


def _status_text() -> str:
    cfg = _mcp_cfg()
    if not cfg.get("enabled", True):
        return "MCP 客户端已禁用（config.mcp.enabled=false）。"
    if not _SERVERS:
        return "MCP 已启用，但未配置/未成功启动任何 server（config.mcp.servers）。"
    lines = [f"MCP servers: {len(_SERVERS)}"]
    for name, entry in sorted(_SERVERS.items()):
        if entry.get("error"):
            lines.append(f"- {name} [❌ {entry['error']}]")
        else:
            lines.append(f"- {name} [✅ tier={entry.get('tier')} "
                         f"tools={len(entry.get('registered') or [])}]")
            if entry.get("registered"):
                lines.append(f"    {', '.join(entry['registered'])}")
        tail = ""
        try:
            tail = entry.get("client").stderr_tail() if entry.get("client") else ""
        except Exception:
            pass
        if tail:
            lines.append(f"    stderr: {tail.splitlines()[-1][:200]}")
    return _clip("\n".join(lines))


@tool(
    "【MCP】查看已配置 MCP server 的连接状态、权限级别、已注册工具与 stderr 尾部。",
    _tier="read", _read_only=True,
)
def mcp_status() -> str:
    return _status_text()


@tool(
    "【MCP】重新连接 MCP server 并重新发现工具。server 留空表示重载全部。",
    server="server 名称（留空=全部）",
    _tier="execute", _read_only=False,
)
def mcp_reload(server: str = "") -> str:
    return _reload_servers(server)


def _shutdown():
    with _LOCK:
        for name in list(_SERVERS.keys()):
            _remove_server_entry(name)


atexit.register(_shutdown)

# 插件加载时自动连接（config.mcp.servers 为空时零开销）
try:
    _init_servers()
except Exception as e:
    print(f"[mcp] 初始化失败: {e}")
