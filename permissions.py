"""permissions.py — 工具权限分级引擎（read / write / delete / execute / network）。

设计目标：
- 工具按能力分级，而不是"要不要授权"的非黑即白。
- read / write 默认放行（write 仍受 safety.is_safe_path 与写入审核约束）。
- delete / execute / network 默认需要人类批准；批准可按 tool / path / signature
  在会话内记住，避免每步重复弹窗。
- 已有内部审批的工具（safe_delete / execute_system_command / run_external_app）
  标记为 self_gated，避免二次弹窗。
"""
import json
import os
import threading

TIERS = ("read", "write", "delete", "execute", "network")

DEFAULT_TOOL_TIERS = {
    # 只读
    "read_file": "read",
    "read_core_file": "read",
    "list_tools": "read",
    "load_more_tools": "read",
    "repo_map": "read",
    "search_symbol": "read",
    "find_callers": "read",
    "index_repo": "read",          # 只写 .agent_index 缓存
    "mcp_status": "read",
    "parallel_execute": "read",    # 自身只负责编排；每个子步骤单独鉴权
    "experience_recall": "read",
    # 写
    "write_file": "write",
    "manage_experience": "write",
    "run_python_file": "write",    # AST + 沙箱 + 冒烟保护下按 write 处理
    # 删
    "safe_delete": "delete",
    # 执行
    "execute_system_command": "execute",
    "install_package": "execute",
    "run_external_app": "execute",
    "mcp_reload": "execute",
    # 网络
    "web_search": "network",
}

# 这些工具在实现内部已经调用了 human_approve，集中鉴权层不再重复弹窗
DEFAULT_SELF_GATED = frozenset({
    "safe_delete", "execute_system_command", "run_external_app",
})

_GRANTS = set()
_GRANT_LOCK = threading.RLock()


def reset_session_grants() -> None:
    with _GRANT_LOCK:
        _GRANTS.clear()


def get_tier(tool_name: str, cfg: dict = None, meta_tier: str = None) -> str:
    cfg = cfg or {}
    overrides = cfg.get("tool_tiers") or {}
    tier = overrides.get(tool_name) or meta_tier or DEFAULT_TOOL_TIERS.get(
        tool_name, cfg.get("default_tier", "write"))
    tier = str(tier).lower()
    return tier if tier in TIERS else "execute"


def _clip(v, n: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def describe_action(tool_name: str, args: dict) -> str:
    args = args or {}
    for key in ("filepath", "path", "script_path", "sln_path", "package_name",
                "query", "url", "cmd_list", "command"):
        if key in args and args[key] not in (None, "", []):
            return f"{key}={_clip(args[key], 160)}"
    return _clip(args, 160)


def _grant_key(tool_name: str, args: dict, tier: str, cfg: dict) -> str:
    scope = (cfg.get("remember_scope") or {}).get(tier, "tool")
    if scope == "signature":
        try:
            sig = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
        except Exception:
            sig = str(args)
        return f"{tier}|{tool_name}|{sig}"
    if scope == "path":
        raw = (args or {}).get("filepath") or (args or {}).get("path") or ""
        try:
            p = os.path.realpath(raw) if raw else ""
        except Exception:
            p = str(raw)
        return f"{tier}|{tool_name}|{p.lower()}"
    return f"{tier}|{tool_name}"


def check_tool_permission(tool_name: str, args: dict, cfg: dict, approve_fn,
                          meta_tier: str = None):
    """返回 (allowed: bool, message: str)。message 在拒绝时可直接作为工具结果。"""
    cfg = cfg or {}
    if not cfg.get("enabled", True):
        return True, ""

    tier = get_tier(tool_name, cfg, meta_tier)
    deny = {str(x).lower() for x in (cfg.get("deny") or [])}
    if tier in deny:
        return False, (f"⛔ 权限拒绝：{tier} 级工具 `{tool_name}` 已被策略禁用。"
                       f"请在 config.json → permissions.deny 中调整。")

    self_gated = {str(x) for x in (cfg.get("self_gated_tools") or DEFAULT_SELF_GATED)}
    if tool_name in self_gated:
        return True, ""   # 工具内部自行审批

    auto = {str(x).lower() for x in (cfg.get("auto_allow") or [])}
    if tier in auto:
        return True, ""

    key = _grant_key(tool_name, args, tier, cfg)
    with _GRANT_LOCK:
        if key in _GRANTS:
            return True, ""

    action = f"[{tier} 权限] 工具 {tool_name}：{describe_action(tool_name, args)}"
    try:
        approved = bool(approve_fn(action))
    except Exception:
        approved = False
    if not approved:
        return False, (f"已取消：用户或环境未批准 {tier} 级操作（{tool_name}）。"
                       f"请向用户说明需要该权限，不要反复重试。")

    if cfg.get("remember_session", True):
        with _GRANT_LOCK:
            _GRANTS.add(key)
    return True, ""


def status(cfg: dict = None) -> dict:
    cfg = cfg or {}
    with _GRANT_LOCK:
        grants = sorted(_GRANTS)
    return {
        "tiers": list(TIERS),
        "auto_allow": list(cfg.get("auto_allow") or []),
        "deny": list(cfg.get("deny") or []),
        "remember_session": bool(cfg.get("remember_session", True)),
        "session_grants": grants,
    }
