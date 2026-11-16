"""plugins/evolution_tools.py — 自进化提案 / 评测 / 晋升 / 回滚工具。"""
from safety import tool
import evolution


@tool(
    "【自进化】升级流水线。action: propose(goal) / evaluate(proposal_id) / "
    "shadow / promote(proposal_id) / canary / rollback(version) / undo(version) / "
    "baseline / list / status / sync_frozen / meta_eval。"
    "候选只写入 .evolution/candidates/<id>/；冻结文件改动会被评分门禁否决；"
    "promote 需要人类 approve，晋升后自动 canary，失败可自动撤销。",
    action="动作名",
    goal="action=propose 时必填：升级目标",
    proposal_id="action=evaluate/promote 时必填",
    version="action=rollback 时可选，默认当前版本",
    _tier="execute", _read_only=False,
)
def evolve(action: str, goal: str = "", proposal_id: str = "",
           version: str = "") -> str:
    return evolution.run_action(action, goal=goal,
                                proposal_id=proposal_id, version=version)


@tool(
    "【自进化】查看当前版本、基线、提案数量、最近评分卡与冻结任务集。",
    _tier="read", _read_only=True,
)
def evolution_status() -> str:
    return evolution.status_text()
