"""checkpoints.py — 任务检查点与回滚。

每次 run_agent 一步结束保存：
- chat_history 快照（超长时保留 system + 最近消息）；
- 当前执行计划；
- 本次会话写过的文件清单（含写前备份路径与写入 step）。

回滚时：
- 恢复快照中的 chat_history；
- 对「检查点之后」写过的文件，恢复其最早一次写前备份；
- 检查点之后新建的文件会被删除。

不依赖 safety（避免循环导入），工作区根取本文件所在目录。
"""
import json
import os
import tempfile
import threading
import time

BASE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CFG = {
    "enabled": True,
    "dir": ".agent_checkpoints",
    "max_per_session": 40,
    "max_history_chars": 200000,
    "max_files_per_session": 200,
}
_LOCK = threading.RLock()
_FILES = {}   # session -> [{"step","path","backup","created"}]


def _cfg(cfg: dict = None) -> dict:
    out = dict(DEFAULT_CFG)
    if isinstance(cfg, dict):
        out.update(cfg)
    return out


def _dir(cfg: dict = None) -> str:
    return os.path.join(BASE, _cfg(cfg).get("dir", ".agent_checkpoints"))


def _safe_id(cid: str) -> str:
    cid = str(cid or "")
    if not cid or any(ch in cid for ch in ("/", "\\", "..")):
        return ""
    return cid


def _shrink_history(history: list, max_chars: int) -> list:
    try:
        size = sum(len(str(m.get("content", ""))) for m in history)
    except Exception:
        return history
    if size <= max_chars or len(history) <= 2:
        return history
    head = [history[0]] if history and history[0].get("role") == "system" else []
    used = sum(len(str(m.get("content", ""))) for m in head)
    tail = []
    for m in reversed(history[len(head):]):
        n = len(str(m.get("content", "")))
        if used + n > max_chars:
            break
        tail.append(m)
        used += n
    tail.reverse()
    out = head + tail
    if head and len(out) == 1:
        out.append({"role": "system",
                    "content": "[检查点提示] 历史过长，仅保存了 system 消息。"})
    return out


def _journal_path(session_id: str, cfg: dict = None) -> str:
    return os.path.join(_dir(cfg), f"{session_id}.journal.json")


def _load_journal(session_id: str, cfg: dict = None) -> list:
    try:
        with open(_journal_path(session_id, cfg), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries") or []
    except Exception:
        return []


def _save_journal(session_id: str, entries: list, cfg: dict = None) -> None:
    try:
        d = _dir(cfg)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix="journal.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"session": session_id, "entries": entries}, f,
                      ensure_ascii=False)
        os.replace(tmp, _journal_path(session_id, cfg))
    except Exception:
        pass


def record_file(session_id: str, step: int, filepath: str,
                backup_path: str = "", created: bool = False,
                cfg: dict = None) -> None:
    """由 safety.process_write 在写入成功后调用，记录写前状态（含磁盘日志）。"""
    sid = str(session_id or "manual")
    entry = {"step": int(step or 0), "path": os.path.realpath(filepath),
             "backup": os.path.realpath(backup_path) if backup_path else "",
             "created": bool(created)}
    with _LOCK:
        arr = _load_journal(sid, cfg) or list(_FILES.get(sid, []))
        arr.append(entry)
        cap = int(_cfg(cfg).get("max_files_per_session", 200) or 200)
        if len(arr) > cap:
            del arr[:len(arr) - cap]
        _FILES[sid] = arr
        _save_journal(sid, arr, cfg)


def save(session_id: str, step: int, history: list, plan: dict = None,
         cfg: dict = None) -> str:
    """保存检查点，返回 checkpoint_id；失败返回 ""。"""
    c = _cfg(cfg)
    if not c.get("enabled", True) or not session_id or not history:
        return ""
    sid = str(session_id)
    hist = _shrink_history(history, int(c.get("max_history_chars", 200000) or 200000))
    with _LOCK:
        files = _load_journal(sid, c) or list(_FILES.get(sid, []))
    cid = f"{sid}__s{int(step):03d}_{time.time_ns()}"
    data = {
        "version": 1,
        "id": cid,
        "session": sid,
        "step": int(step),
        "ts": time.time(),
        "history": hist,
        "plan": plan,
        "files": files,
    }
    d = _dir(c)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="ckpt.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(d, cid + ".json"))
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return ""
    _retain(sid, d, int(c.get("max_per_session", 40) or 40))
    return cid


def _retain(session_id: str, d: str, max_count: int) -> None:
    try:
        items = []
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = None
            if info and info.get("id") and info.get("session") == session_id:
                items.append((info.get("ts", 0), fn))
        items.sort(reverse=True)
        for _, fn in items[max_count:]:
            try:
                os.remove(os.path.join(d, fn))
            except OSError:
                pass
    except OSError:
        pass


def _load_stem(stem: str, cfg: dict = None):
    path = os.path.join(_dir(cfg), stem + ".json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load(checkpoint_id: str, cfg: dict = None):
    cid = _safe_id(checkpoint_id)
    if not cid:
        return None
    return _load_stem(cid, cfg)


def list_checkpoints(session_id: str = "", limit: int = 20,
                     cfg: dict = None) -> list:
    d = _dir(cfg)
    if not os.path.isdir(d):
        return []
    out = []
    try:
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            info = _load_stem(fn[:-5], cfg)
            if not info or not info.get("id"):
                continue
            if session_id and info.get("session") != session_id:
                continue
            out.append({
                "id": info.get("id"),
                "session": info.get("session"),
                "step": info.get("step"),
                "ts": info.get("ts"),
                "plan_goal": ((info.get("plan") or {}).get("goal") or "")[:120],
                "files": len(info.get("files") or []),
                "messages": len(info.get("history") or []),
            })
    except OSError:
        return []
    out.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    return out[:max(1, int(limit or 20))]


def _inside_base(path: str) -> bool:
    if not path:
        return False
    real = os.path.realpath(path)
    return real == BASE or real.startswith(BASE + os.sep)


def rollback(checkpoint_id: str, restore_files: bool = True,
             cfg: dict = None) -> dict:
    """回滚到检查点。返回 {ok, history, files_restored, files_deleted, errors}。"""
    info = load(checkpoint_id)
    if not info:
        return {"ok": False, "error": f"检查点不存在或不可读: {checkpoint_id}",
                "history": None, "files_restored": [], "files_deleted": [], "errors": []}
    errors, restored, deleted = [], [], []
    journal = _load_journal(str(info.get("session") or ""), cfg)
    entries = journal or info.get("files") or []
    if restore_files:
        # 每个文件取“检查点之后最早的一次写入”的写前状态
        earliest = {}
        for e in entries:
            if int(e.get("step", 0)) <= int(info.get("step", 0)):
                continue
            p = e.get("path")
            if not _inside_base(p):
                errors.append(f"越界文件已跳过: {p}")
                continue
            if p not in earliest or int(e["step"]) < int(earliest[p]["step"]):
                earliest[p] = e
        for path, e in earliest.items():
            try:
                backup = e.get("backup") or ""
                if backup and _inside_base(backup) and os.path.isfile(backup):
                    import shutil
                    shutil.copy2(backup, path)
                    restored.append(path)
                elif e.get("created"):
                    if os.path.isfile(path):
                        os.remove(path)
                        deleted.append(path)
                else:
                    errors.append(f"备份缺失，无法恢复: {path}")
            except Exception as ex:
                errors.append(f"{path}: {ex}")
        # 已回滚，丢弃检查点之后的journal条目，避免二次回滚误恢复
        sid = str(info.get("session") or "")
        if sid:
            remain = [e for e in entries
                      if int(e.get("step", 0)) <= int(info.get("step", 0))]
            _FILES[sid] = remain
            _save_journal(sid, remain, cfg)
    return {
        "ok": True,
        "id": info.get("id"),
        "session": info.get("session"),
        "step": info.get("step"),
        "history": info.get("history"),
        "files_restored": restored,
        "files_deleted": deleted,
        "errors": errors,
    }
