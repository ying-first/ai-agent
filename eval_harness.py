"""eval_harness.py — Agent 任务评测与回归。

任务格式（evals/tasks.json）：
[
  {
    "id": "repo_symbol",
    "prompt": "用仓库索引找到 run_agent 定义位置并回答",
    "tags": ["repo"],
    "max_steps": 4,
    "allow_tools": ["search_symbol", "read_file"],
    "checks": {
      "answer_contains": ["agent.py"],
      "tools_used": ["search_symbol"],
      "tools_not_used": ["write_file"],
      "max_steps": 4
    },
    "cleanup": ["_eval_tmp.py"]
  }
]

首轮运行可 update_baseline=True 建立基线；后续运行会自动比较并报告回归。
"""
import glob
import json
import os
import threading
import time

BASE = os.path.dirname(os.path.realpath(__file__))
EVAL_DIR = os.path.join(BASE, "evals")
TASKS_PATH = os.path.join(EVAL_DIR, "tasks.json")
REPORT_DIR = os.path.join(EVAL_DIR, ".reports")
BASELINE_PATH = os.path.join(EVAL_DIR, "baseline.json")
_RUN_LOCK = threading.Lock()


def _read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_tasks(tags=None, limit=None, task_file=None):
    tasks = _read_json(task_file or TASKS_PATH, [])
    if not isinstance(tasks, list):
        return []
    if tags:
        want = {str(t) for t in tags}
        tasks = [t for t in tasks
                 if want & {str(x) for x in (t.get("tags") or [])}]
    if limit:
        tasks = tasks[:max(1, int(limit))]
    return tasks


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _check_answer(task, answer, events, steps, outcome):
    checks = task.get("checks") or {}
    failures = []
    answer = str(answer or "")
    tools = [e.get("tool") for e in events if e.get("tool")]

    for s in _as_list(checks.get("answer_contains")):
        if str(s) not in answer:
            failures.append(f"answer 缺少 {s!r}")
    for s in _as_list(checks.get("answer_not_contains")):
        if str(s) in answer:
            failures.append(f"answer 不应包含 {s!r}")
    for s in _as_list(checks.get("tools_used")):
        if s not in tools:
            failures.append(f"未调用工具 {s}")
    any_tools = _as_list(checks.get("tools_any"))
    if any_tools and not any(t in tools for t in any_tools):
        failures.append(f"未调用任一工具 {any_tools}")
    for s in _as_list(checks.get("tools_not_used")):
        if s in tools:
            failures.append(f"不应调用工具 {s}")
    exp_outcome = checks.get("outcome")
    if exp_outcome and outcome != exp_outcome:
        failures.append(f"outcome={outcome}，期望 {exp_outcome}")
    max_steps = checks.get("max_steps")
    if max_steps is not None and int(steps or 0) > int(max_steps):
        failures.append(f"steps={steps} 超过 {max_steps}")

    regex = checks.get("answer_regex")
    if regex:
        import re
        if not re.search(str(regex), answer, re.DOTALL):
            failures.append(f"answer 不匹配正则 {regex}")

    for p in _as_list(checks.get("files_exist")):
        if not os.path.exists(os.path.join(BASE, str(p))):
            failures.append(f"文件不存在: {p}")
    for p in _as_list(checks.get("files_not_exist")):
        if os.path.exists(os.path.join(BASE, str(p))):
            failures.append(f"文件应被删除: {p}")
    for item in _as_list(checks.get("files_contain")):
        if isinstance(item, str):
            path, needle = item, ""
        else:
            path, needle = item.get("path"), item.get("contains", "")
        try:
            with open(os.path.join(BASE, str(path)), "r", encoding="utf-8",
                      errors="replace") as f:
                text = f.read()
        except Exception as e:
            failures.append(f"读取 {path} 失败: {e}")
            continue
        if needle and str(needle) not in text:
            failures.append(f"{path} 缺少 {needle!r}")
    return failures


def _cleanup(task):
    for p in _as_list(task.get("cleanup")):
        full = os.path.join(BASE, str(p))
        try:
            if os.path.isfile(full):
                os.remove(full)
            for bak in glob.glob(full + ".bak_*"):
                try:
                    os.remove(bak)
                except OSError:
                    pass
        except OSError:
            pass


def _run_task(task):
    """在隔离历史 + 只读白名单中运行一个任务，返回结果 dict。"""
    import agent as agent_mod
    import safety

    prompt = str(task.get("prompt") or "")
    allow = task.get("allow_tools")
    if allow is None:
        allow = set((safety.CONFIG.get("subagents") or {}).get("read_only_tools") or [])
    else:
        allow = set(allow)
    max_steps = task.get("max_steps") or \
        int((safety.CONFIG.get("evals") or {}).get("default_max_steps", 6) or 6)

    captured = {"events": [], "outcome": "unknown", "steps": 0}
    old_save = agent_mod._save_trace
    old_exp = agent_mod._try_auto_save_experience
    old_approve = getattr(safety, "_HUMAN_APPROVE_CB", None)
    old_allow = getattr(agent_mod._TOOL_ALLOW_CTX, "allow", None)
    old_depth = getattr(agent_mod._SUBAGENT_DEPTH, "depth", 0)

    def _fake_trace(trace_id, user, events, outcome, steps):
        captured["events"] = list(events or [])
        captured["outcome"] = outcome
        captured["steps"] = steps

    agent_mod._save_trace = _fake_trace
    agent_mod._try_auto_save_experience = lambda *a, **k: None
    safety.set_human_approve(lambda action: False)   # 评测中禁止交互式授权
    hist = [{"role": "system", "content": agent_mod.SYSTEM_PROMPT}]
    t0 = time.time()
    error = ""
    answer = ""
    try:
        answer = agent_mod.run_agent(
            prompt, initial_max_steps=max_steps,
            on_text=lambda t: None, ask_continue=lambda add: False,
            history=hist, subagent=True, tool_allowlist=allow)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        agent_mod._save_trace = old_save
        agent_mod._try_auto_save_experience = old_exp
        safety.set_human_approve(old_approve)
        agent_mod._TOOL_ALLOW_CTX.allow = old_allow
        agent_mod._SUBAGENT_DEPTH.depth = old_depth

    failures = _check_answer(task, answer, captured["events"],
                             captured["steps"], captured["outcome"])
    if error:
        failures.append(error)
    _cleanup(task)
    events = captured["events"]
    tokens = sum(int(e.get("tokens") or 0) for e in events
                 if e.get("tool") == "__context__")
    tool_errors = sum(1 for e in events if e.get("ok") is False)
    return {
        "id": task.get("id") or prompt[:40],
        "tags": task.get("tags") or [],
        "ok": not failures,
        "failures": failures,
        "answer": str(answer)[:1000],
        "steps": captured["steps"],
        "tool_calls": len(events),
        "tool_errors": tool_errors,
        "tokens": tokens,
        "tools": [e.get("tool") for e in events if e.get("tool")],
        "outcome": captured["outcome"],
        "duration_s": round(time.time() - t0, 2),
    }


def run_evals(tags=None, limit=None, update_baseline=False, tasks=None,
              task_file=None):
    """运行评测。返回 {ok, report, regressions, fixes, baseline_updated, error}。"""
    if not _RUN_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "已有评测在运行，请稍后。",
                "report": None, "regressions": [], "fixes": []}
    try:
        if tasks is None:
            tasks = load_tasks(tags=tags, limit=limit, task_file=task_file)
        results = []
        t0 = time.time()
        for task in tasks:
            results.append(_run_task(task))
        passed = sum(1 for r in results if r["ok"])
        report = {
            "ts": time.strftime("%Y%m%d_%H%M%S"),
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / max(1, len(results)), 4),
            "duration_s": round(time.time() - t0, 2),
            "results": results,
        }
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            _write_json(os.path.join(REPORT_DIR,
                                     f"eval_{report['ts']}.json"), report)
        except Exception:
            pass

        baseline = _read_json(BASELINE_PATH, None)
        regressions, fixes = [], []
        if isinstance(baseline, dict):
            old = {r["id"]: r.get("ok") for r in baseline.get("results") or []}
            new = {r["id"]: r.get("ok") for r in results}
            for tid, was_ok in old.items():
                if was_ok and tid in new and not new[tid]:
                    regressions.append(tid)
                elif not was_ok and new.get(tid):
                    fixes.append(tid)

        if update_baseline:
            _write_json(BASELINE_PATH, report)

        return {
            "ok": passed == len(results) and not regressions,
            "report": report,
            "regressions": regressions,
            "fixes": fixes,
            "baseline_updated": bool(update_baseline),
            "previous_pass_rate": (baseline or {}).get("pass_rate"),
            "error": "",
        }
    finally:
        _RUN_LOCK.release()


def latest_report():
    if not os.path.isdir(REPORT_DIR):
        return None
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "eval_*.json")))
    if not files:
        return None
    return _read_json(files[-1], None)


def format_report(result) -> str:
    if not result or result.get("error"):
        return f"❌ 评测失败：{(result or {}).get('error')}"
    rep = result.get("report") or {}
    lines = [f"[evals] 通过 {rep.get('passed')}/{rep.get('total')} "
             f"({rep.get('pass_rate', 0):.0%})，耗时 {rep.get('duration_s')}s"]
    if result.get("baseline_updated"):
        lines.append("✅ 已更新基线 baseline.json")
    if result.get("previous_pass_rate") is not None:
        lines.append(f"上次通过率: {result['previous_pass_rate']:.0%}")
    if result.get("regressions"):
        lines.append(f"🔴 回归任务: {', '.join(result['regressions'])}")
    if result.get("fixes"):
        lines.append(f"🟢 新修复: {', '.join(result['fixes'])}")
    for r in rep.get("results") or []:
        mark = "✅" if r.get("ok") else "❌"
        lines.append(f"\n{mark} {r.get('id')}  steps={r.get('steps')} "
                     f"tools={r.get('tool_calls')}  {r.get('duration_s')}s")
        if r.get("failures"):
            for f in r["failures"]:
                lines.append(f"   - {f}")
        if r.get("answer"):
            lines.append(f"   answer: {r['answer'][:200]}")
    return "\n".join(lines)
