"""plugins/experience.py — 分层经验库工具（SQLite/FTS5 + 向量召回）。

对外保持 auto_save / manage_experience 兼容；底层由 experience_store 提供：
  episode → experience（指纹合并）→ pattern（聚类）→ playbook / principle（需审批）
"""
import hashlib
import json
import os
import time

import experience_store
from safety import tool, WORKSPACE_DIR, CONFIG, human_approve

EXP_ROOT = experience_store._root()
LEGACY_JSONL = os.path.join(WORKSPACE_DIR, ".experiences.jsonl")
ExperienceStore = experience_store.ExperienceStore
_store = experience_store.get_store()


def _exp_cfg():
    out = dict(experience_store.DEFAULT_CFG)
    out.update(CONFIG.get("experience", {}) or {})
    return out


def migrate_legacy():
    """把更早的 .experiences.jsonl 导入新库（一次性）。"""
    if not os.path.exists(LEGACY_JSONL):
        return
    if _store._meta_get("migrated_legacy_jsonl") == "1":
        return
    count = 0
    try:
        with open(LEGACY_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if isinstance(e, dict) and e.get("id"):
                        _store.import_legacy(e)
                        count += 1
                except Exception:
                    pass
        _store._meta_set("migrated_legacy_jsonl", "1")
        if count:
            print(f"[experience] 已导入旧 JSONL {count} 条")
    except Exception as e:
        print(f"[experience] 旧 JSONL 迁移失败: {e}")


migrate_legacy()


def _simple_summary(user: str, answer: str, tool_calls: list) -> dict:
    text = f"{user}\n{answer}"[:600].replace("\n", " ").strip()
    files = []
    for t in tool_calls or []:
        args = (t.get("args") or {}) if isinstance(t, dict) else {}
        for k in ("filepath", "path", "script_path", "sln_path"):
            if args.get(k) and args[k] not in files:
                files.append(args[k])
    return {"summary": text, "files_touched": files[:10], "lessons": [],
            "tags": []}


def make_experience(user: str, answer: str, tool_calls: list = None,
                    outcome: str = "success", steps: int = 0) -> dict:
    """兼容旧接口：只构造轻量 entry，不再同步调用 LLM。"""
    tool_calls = tool_calls or []
    compressed = _simple_summary(user, answer, tool_calls)
    return {
        "id": hashlib.md5(f"{time.time()}-{user[:200]}".encode("utf-8")).hexdigest()[:12],
        "ts": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user[:500],
        "outcome": outcome,
        "steps": int(steps),
        "tool_calls_count": len(tool_calls),
        **compressed,
    }


# ================= 自动保存钩子（供 agent.py 调用） =================

def auto_save(user: str, answer: str, tool_calls: list,
              steps: int = 0, outcome: str = "success",
              strategy: str = "") -> str:
    _store.refresh_cfg()
    cfg = _exp_cfg()
    if not cfg.get("auto_save", True):
        return "skip:auto_save_disabled"
    if not tool_calls and len((user or "").strip()) < 30 and not strategy:
        return "skip:too_short"
    try:
        res = _store.record_episode(user, answer, tool_calls,
                                    steps=steps, outcome=outcome,
                                    strategy=strategy)
        if not res.get("id"):
            return "skip:" + str(res.get("skipped") or "not_saved")
        return ("merged:" if res.get("merged") else "saved:") + res["id"]
    except Exception as e:
        return f"failed:{e}"


# ================= 工具接口 =================

def _format_cards(cards, budget=None) -> list:
    if not cards:
        return []
    out = []
    for c in cards:
        head = (f"[{c.get('kind', 'experience')}] id={c.get('id')} "
                f"置信={float(c.get('confidence') or 0):.2f} "
                f"证据={c.get('evidence', 0)} "
                f"成功/失败={c.get('success', 0)}/{c.get('fail', 0)}")
        body = [head]
        if c.get("trigger"):
            body.append(f"触发: {str(c['trigger'])[:240]}")
        if c.get("action"):
            body.append(f"做法: {str(c['action'])[:320]}")
        if c.get("strategy"):
            body.append(f"策略: {str(c['strategy'])[:320]}")
        if c.get("lesson"):
            body.append(f"经验: {str(c['lesson'])[:400]}")
        if c.get("scope"):
            body.append(f"范围: {str(c['scope'])[:200]}")
        if c.get("exceptions"):
            body.append(f"边界: {str(c['exceptions'])[:200]}")
        if c.get("steps"):
            try:
                steps = c["steps"][:6]
                body.append("步骤: " + "; ".join(
                    f"{i+1}){str(s.get('goal') or s)[:80]}"
                    + (f"[{s.get('tool')}]" if isinstance(s, dict) and s.get("tool") else "")
                    for i, s in enumerate(steps)))
            except Exception:
                pass
        out.append("\n".join(body))
    text = _store.format_cards(cards, token_budget=budget)
    return text.split("\n\n") if text else out


@tool(
    "【经验召回】只读检索分层记忆：返回紧凑经验卡片（触发/做法/教训/边界/置信度），"
    "命中相似经验、模式、已批准 playbook/principle。复杂任务的记忆摘要无法确定时使用；"
    "普通任务会由系统自动注入经验卡片，无需重复调用。",
    query="检索关键词或任务描述",
    k="返回条数，默认 3",
    _tier="read", _read_only=True,
)
def experience_recall(query: str, k: int = 3) -> str:
    _store.refresh_cfg()
    if not (query or "").strip():
        return "query 不能为空"
    r = _store.route(query, k=int(k or 3))
    cards = r.get("cards") or []
    if not cards:
        return "未找到相关经验。"
    text = r.get("text") or _store.format_cards(cards)
    ids = [f"{c.get('kind')}:{c.get('id')}" for c in cards]
    prefix = ""
    if r.get("mode") == "direct" and r.get("playbook"):
        pb = r["playbook"]
        prefix = (f"【可直行】命中已批准 playbook {pb.get('id')} "
                  f"（置信 {pb.get('confidence')}，证据 {pb.get('evidence')}）。"
                  "优先按步骤执行，参数按当前任务填写。\n\n")
    return prefix + text + "\n\n可用 id: " + ", ".join(ids)


@tool(
    "【经验库统一管理】分层经验记忆。action 支持：\n"
    "  save(user,answer) / recall(query) / get(exp_id) / list / forget(exp_id)\n"
    "  stats / merge(exp_id=逗号分隔id) / feedback(exp_id,used,outcome)\n"
    "  consolidate(limit) / rebuild(force_vectors) / export(path)\n"
    "  route(query) / correct(exp_id,lesson/action/...) / suspect(exp_id,reason)\n"
    "  pending(待审批) / approve(exp_id,used=是否批准) / gaps / resolve_gap(exp_id)\n"
    "  metrics(A/B/直行指标) / contradictions(矛盾经验)\n"
    "经验自动指纹合并；相似经验聚类为模式；playbook/principle 需人类批准后才会注入。",
    action="操作类型",
    user="action=save：用户原始需求",
    answer="action=save：最终回答",
    strategy="action=save/correct：可复用思考/执行策略（可空，后台也会自动提炼）",
    query="action=recall/route：检索词",
    exp_id="action=get/forget/feedback/approve/correct/suspect/resolve_gap：id；merge 逗号分隔",
    used="action=feedback：是否采纳；action=approve：是否批准（默认 true）",
    outcome="action=feedback：success/fail",
    limit="action=consolidate：本轮处理条数",
    force_vectors="action=rebuild：是否重建向量与聚类",
    path="action=export：导出路径",
    trigger="action=correct：新的触发条件",
    action_text="action=correct：新的做法",
    lesson="action=correct：新的经验/教训",
    scope="action=correct：适用范围",
    exceptions="action=correct：反例/边界",
    tags="action=correct：标签，逗号分隔",
    reason="action=correct/suspect：修正原因",
)
def manage_experience(action: str, user: str = "", answer: str = "",
                      query: str = "", exp_id: str = "", used: bool = True,
                      outcome: str = "success", limit: int = 5,
                      force_vectors: bool = False, path: str = "",
                      strategy: str = "", trigger: str = "",
                      action_text: str = "", lesson: str = "",
                      scope: str = "", exceptions: str = "",
                      tags: str = "", reason: str = "") -> str:
    action = (action or "").strip().lower()
    _store.refresh_cfg()
    try:
        if action == "save":
            if not user.strip():
                return "保存失败：user 不能为空"
            res = _store.record_episode(user, answer, [], outcome="success",
                                        force=True, strategy=strategy)
            return (f"✅ 经验已保存（id={res['id']}，"
                    f"{'已合并到既有经验' if res.get('merged') else '新经验'}；"
                    f"后台会压缩并聚类）\n摘要: {str(answer)[:100]}")
        if action in ("recall", "route"):
            if not query.strip():
                return "query 不能为空"
            r = _store.route(query, k=int(limit or 3) if limit else None)
            if not r.get("cards"):
                return "未找到相关经验。"
            text = r.get("text") or _store.format_cards(r["cards"])
            if action == "route":
                pb = r.get("playbook") or {}
                return (f"route={r.get('mode')} reason={r.get('reason')} "
                        f"playbook={pb.get('id', '-')}\n\n" + text)
            return text
        if action == "get":
            if not exp_id.strip():
                return "exp_id 不能为空"
            e = _store.get(exp_id)
            if not e:
                return f"未找到 id={exp_id}"
            kind = e.get("kind", "experience")
            lines = [f"=== {kind} {e.get('id')} ==="]
            for k in ("trigger", "action", "lesson", "strategy", "scope",
                      "exceptions", "text", "name", "summary", "status",
                      "confidence", "evidence", "support", "success", "fail",
                      "utility", "revision", "corrected_from",
                      "superseded_by", "last_fail_reason"):
                if e.get(k) not in (None, "", [], {}):
                    lines.append(f"{k}: {e[k]}")
            if e.get("steps"):
                lines.append("steps: " + json.dumps(e["steps"], ensure_ascii=False))
            if e.get("tags"):
                lines.append(f"tags: {e['tags']}")
            eps = e.get("episodes") or []
            if eps:
                lines.append(f"关联 episode: {len(eps)} 条（最近 {eps[0].get('id')}）")
            if e.get("contradictions"):
                lines.append(f"矛盾/边界: {len(e['contradictions'])} 条")
            return "\n".join(lines)
        if action == "list":
            rows = _store.list_recent(int(limit or 10))
            if not rows:
                return "经验库为空。"
            lines = [f"最近 {len(rows)} 条经验："]
            for e in rows:
                lines.append(f"- {e['id']} | {e.get('trigger', '')[:80]} | "
                             f"置信 {e.get('confidence', 0):.2f} | "
                             f"成功率 {e.get('success', 0)}/{e.get('fail', 0)}")
            return "\n".join(lines)
        if action == "forget":
            if not exp_id.strip():
                return "exp_id 不能为空"
            return (f"✅ 已删除经验 {exp_id}"
                    if _store.forget(exp_id) else f"未找到 id={exp_id}")
        if action == "stats":
            s = _store.stats()
            lines = [f"经验 {s['experiences']} 条 | episode {s['episodes']} 条 | "
                     f"模式 {s['patterns']} | playbook {s['playbooks']} | "
                     f"principle {s['principles']}",
                     f"未巩固 {s['unconsolidated']} | 召回记录 {s['recalls']} | "
                     f"矛盾 {s['contradictions']} | 可疑 {s['suspect']} | "
                     f"能力缺口 {s['gaps']}",
                     f"热门标签: {', '.join(f'{t}({c})' for t, c in s['top_tags']) or '(无)'}",
                     f"DB: {s['db']}"]
            return "\n".join(lines)
        if action == "merge":
            ids = [x.strip() for x in (exp_id or "").replace("，", ",").split(",")
                   if x.strip()]
            return _store.merge(ids)
        if action == "feedback":
            ids = [x.strip() for x in (exp_id or "").replace("，", ",").split(",")
                   if x.strip()]
            if not ids:
                return "exp_id 不能为空"
            return _store.feedback(ids, used=bool(used), outcome=outcome)
        if action == "correct":
            if not exp_id.strip():
                return "exp_id 不能为空"
            return _store.correct(exp_id, trigger=trigger, action=action_text,
                                  lesson=lesson, scope=scope,
                                  exceptions=exceptions, tags=tags,
                                  reason=reason, strategy=strategy)
        if action == "suspect":
            if not exp_id.strip():
                return "exp_id 不能为空"
            return _store.mark_suspect(exp_id, reason)
        if action == "gaps":
            return _store.format_gaps()
        if action == "resolve_gap":
            if not exp_id.strip():
                return "gap id 不能为空"
            return _store.resolve_gap(exp_id, resolved_by=reason)
        if action == "consolidate":
            res = _store.consolidate(limit=int(limit or 5), use_llm=True)
            return (f"✅ 已巩固 {res['consolidated']} 条，剩余 "
                    f"{res['remaining']} 条待巩固")
        if action == "rebuild":
            return _store.rebuild(force_vectors=bool(force_vectors))
        if action == "export":
            if path:
                real = os.path.realpath(path)
                ws = os.path.realpath(WORKSPACE_DIR)
                try:
                    inside = os.path.commonpath([real, ws]) == ws
                except ValueError:
                    inside = False
                if not inside:
                    return "导出被拦截：路径不在工作区内。"
            return _store.export(path)
        if action == "metrics":
            m = _store.metrics()
            lines = [f"A/B: {'启用' if m['ab_enabled'] else '未启用'}"
                     f"（每组至少 {m['ab_min_samples']} 样本）",
                     f"召回使用率: {m['recall']['use_rate']:.1%} "
                     f"({m['recall']['used']}/{m['recall']['offered']}) | "
                     f"平均效用 {m['avg_utility']:.2f}",
                     f"经验直行: {m['direct']['runs']} 次，成功率 "
                     f"{m['direct']['success_rate']:.1%}"]
            if not m["arms"]:
                lines.append("暂无 A/B 样本；开启 config.experience.ab_test.enabled 后运行任务。")
            for arm, a in sorted(m["arms"].items()):
                lines.append(f"- {arm}: n={a['n']} 成功率={a['success_rate']:.1%} "
                             f"平均步数={a['avg_steps']} 延迟={a['avg_latency_s']}s "
                             f"召回={a['recalled']}")
            return "\n".join(lines)
        if action == "contradictions":
            with _store._lock:
                rows = _store._q("SELECT a_id,b_id,reason,status,created "
                                 "FROM contradictions WHERE status='open' "
                                 "ORDER BY created DESC LIMIT 20")
            if not rows:
                return "暂无矛盾经验。"
            lines = [f"矛盾/边界对 {len(rows)} 组（相似做法结果相反）："]
            for r in rows:
                lines.append(f"- {r['a_id']} ↔ {r['b_id']} | {r['reason']}")
            return "\n".join(lines)
        if action == "pending":
            with _store._lock:
                plays = _store._q("SELECT id,name,trigger,confidence,evidence "
                                  "FROM playbooks WHERE status='candidate'")
                prins = _store._q("SELECT id,text,confidence,evidence "
                                  "FROM principles WHERE status='candidate'")
                clusters = _store._q("SELECT id,trigger,lesson,support,confidence "
                                     "FROM clusters WHERE status='active' "
                                     "AND lesson!='' ORDER BY support DESC LIMIT 10")
            lines = [f"待审批 playbook: {len(plays)} | principle: {len(prins)}",
                     f"已抽象模式: {len(clusters)}"]
            for r in plays:
                lines.append(f"- playbook {r['id']} | {r.get('name', '')[:80]} | "
                             f"证据 {r.get('evidence', 0)}")
            for r in prins:
                lines.append(f"- principle {r['id']} | "
                             f"{str(r.get('text', ''))[:100]} | "
                             f"证据 {r.get('evidence', 0)}")
            for r in clusters:
                lines.append(f"- pattern {r['id']} | "
                             f"{str(r.get('trigger', ''))[:60]} → "
                             f"{str(r.get('lesson', ''))[:80]} | 支持 {r.get('support', 0)}")
            if len(lines) == 2:
                lines.append("(暂无待审批项；需要更多相似经验形成证据)")
            return "\n".join(lines)
        if action == "approve":
            if not exp_id.strip():
                return "exp_id 不能为空"
            if used and not human_approve(
                    f"批准经验原则/流程 {exp_id} 进入自动注入？"):
                return "已取消：人类未批准，该条保持 candidate。"
            return _store.approve(exp_id, approve=bool(used))
        return f"未知操作: {action}"
    except Exception as e:
        return f"经验操作失败: {e}"
