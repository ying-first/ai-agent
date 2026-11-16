"""验证 400 修复：trim_history 裁剪后是否仍产出合法的 messages 结构。

400 (invalid_request_error) 最常见的两个成因：
  A. role="tool" 消息找不到前置 assistant.tool_calls（裁剪把头砍掉了）
  B. assistant.tool_calls 没有对应的 role="tool" 回应（裁剪把尾巴砍掉了）
另外：content 为 None、未知 role 也可能触发。
"""
import sys


def validate(out):
    """返回问题列表，空列表代表合法。"""
    problems = []
    declared = {}
    resp_ids = set()

    for i, m in enumerate(out):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            problems.append(f"[{i}] 非法 role={role!r}")

        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tid = tc.get("id")
                if tid in declared:
                    problems.append(f"[{i}] tool_call id={tid} 重复声明")
                declared[tid] = i

        if role == "tool":
            tid = m.get("tool_call_id")
            resp_ids.add(tid)
            if tid not in declared:
                problems.append(f"[{i}] A类: tool 消息 tool_call_id={tid} 无前置 tool_calls")
            elif declared[tid] > i:
                problems.append(f"[{i}] A类: tool 消息出现在其 tool_calls 之前")

        if role != "assistant" and m.get("content") is None:
            problems.append(f"[{i}] B类: role={role} content=None（部分供应商直接 400）")

    for tid, idx in declared.items():
        if tid not in resp_ids:
            if idx != len(out) - 1:
                problems.append(f"[{idx}] B类: tool_calls id={tid} 无 tool 回应且不在末尾")

    if out and out[0].get("role") == "tool":
        problems.append("[0] 首条即 tool 消息（几乎没有供应商接受）")

    return problems


def M(role, content="", **kw):
    d = {"role": role, "content": content}
    d.update(kw)
    return d


def build_rich(n_turns=12, tool_result_len=800):
    h = [M("system", "你是助手。" * 20)]
    for t in range(n_turns):
        h.append(M("user", f"第{t}轮问题 " + "描述" * 30))
        h.append(M("assistant", "",
                   tool_calls=[{"id": f"call_{t}", "type": "function",
                                "function": {"name": "read_local_file",
                                             "arguments": '{"filepath": "a.py"}'}}]))
        h.append(M("tool", "X" * tool_result_len, tool_call_id=f"call_{t}", name="read_local_file"))
        h.append(M("assistant", f"第{t}轮回答 " + "结论" * 20))
    return h


def build_cases():
    cases = []
    base = build_rich()
    for budget in (300, 600, 1200, 2500, 5000, 15000):
        cases.append((f"rich history / max_chars={budget}", base, budget))

    h = [M("system", "sys"),
         M("user", "读文件"),
         M("assistant", "", tool_calls=[{"id": "c1", "type": "function",
                                         "function": {"name": "read", "arguments": "{}"}}]),
         M("tool", "Y" * 20000, tool_call_id="c1"),
         M("assistant", "好了")]
    cases.append(("单条 20k 工具结果", h, 1000))

    cases.append(("中文/emoji 历史", build_rich(4, 200), 200))

    h = [M("system", "sys"),
         M("user", "q"),
         M("assistant", "", tool_calls=[{"id": "c9", "type": "function",
                                         "function": {"name": "f", "arguments": "{}"}}]),
         M("tool", "res", tool_call_id="c9")]
    cases.append(("仅一轮 tool 交互", h, 60))
    return cases, base


def main():
    """捕获全部输出后以异常形式抛出，便于在无 stdout 回显的环境中查看。"""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = _run()
    raise RuntimeError("EXIT=%d\n%s" % (code, buf.getvalue()))


def _run():
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from safety import trim_history

    cases, base = build_cases()
    failed = 0
    for name, hist, budget in cases:
        try:
            out = trim_history([dict(m) for m in hist], max_chars=budget)
            probs = validate(out)
        except Exception as e:  # noqa: BLE001
            probs = [f"抛异常: {type(e).__name__}: {e}"]
            out = []
        if probs:
            failed += 1
            print(f"[FAIL] {name}  ({len(hist)} -> {len(out)} 条)")
            for p in probs[:6]:
                print("        -", p)
        else:
            print(f"[ OK ] {name}  ({len(hist)} -> {len(out)} 条)")

    out1 = trim_history([dict(m) for m in base], max_chars=800)
    out2 = trim_history([dict(m) for m in out1], max_chars=800)
    p2 = validate(out2)
    if p2:
        failed += 1
        print("[FAIL] 二次裁剪幂等性")
        for p in p2[:6]:
            print("        -", p)
    else:
        print("[ OK ] 二次裁剪幂等性")

    print("\n==== 结果:", "全部通过，未复现 400 结构问题" if failed == 0
          else f"{failed} 个用例失败", "====")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
