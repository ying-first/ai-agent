"""plugins/eval_tools.py — 评测与回归工具。"""
from safety import tool
import eval_harness


@tool(
    "【评测】运行 evals/tasks.json 中的任务并生成报告；默认与 baseline.json 比较回归。"
    "首次建立基线时传 update_baseline=true。",
    tag="只运行包含该 tag 的任务（留空=全部）",
    limit="最多运行任务数（0=全部）",
    update_baseline="true=把本次结果写为基线",
    _tier="execute", _read_only=False,
)
def run_eval_suite(tag: str = "", limit: int = 0, update_baseline: bool = False) -> str:
    tags = [tag] if tag else None
    try:
        lim = int(limit) if limit else None
    except Exception:
        lim = None
    res = eval_harness.run_evals(tags=tags, limit=lim,
                                 update_baseline=bool(update_baseline))
    return eval_harness.format_report(res)


@tool(
    "【评测】查看最近一次评测报告（通过率、失败任务、回归列表）。",
    _tier="read", _read_only=True,
)
def eval_report() -> str:
    rep = eval_harness.latest_report()
    if not rep:
        return "暂无评测报告。先运行 run_eval_suite。"
    lines = [f"最近评测 {rep.get('ts')}：通过 {rep.get('passed')}/{rep.get('total')} "
             f"({rep.get('pass_rate', 0):.0%})，耗时 {rep.get('duration_s')}s"]
    for r in rep.get("results") or []:
        mark = "✅" if r.get("ok") else "❌"
        lines.append(f"{mark} {r.get('id')} steps={r.get('steps')} "
                     f"tools={r.get('tool_calls')}")
        for f in r.get("failures") or []:
            lines.append(f"   - {f}")
    return "\n".join(lines)
