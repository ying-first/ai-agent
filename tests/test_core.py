import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("AI_AGENT_DISABLE_EMBEDDINGS", "1")
os.environ.setdefault("AI_AGENT_NO_BG", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)

import safety
import experience_store


class ConfigTests(unittest.TestCase):
    def test_config_json_and_types(self):
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(safety._config_type_error(safety._CONFIG_DEFAULT, data), "")

    def test_evolution_status_task_uses_semantic_check(self):
        with open(os.path.join(ROOT, "evals", "tasks.json"), encoding="utf-8") as f:
            tasks = json.load(f)
        task = next(t for t in tasks if t["id"] == "evolution_status")
        self.assertIn("当前版本", task["checks"]["answer_contains"])
        self.assertIn("基线", task["checks"]["answer_contains"])


class WriteGateTests(unittest.TestCase):
    def setUp(self):
        self.old_approve = safety._HUMAN_APPROVE_CB
        self.old_reviewer = safety.call_reviewer_agent
        self.old_fail = safety._REVIEW_FAIL_COUNT
        self.old_degrade = safety._REVIEW_DEGRADE_UNTIL
        self.probe = os.path.join(safety.WORKSPACE_DIR, "_ci_gate_probe.txt")
        if os.path.exists(self.probe):
            os.remove(self.probe)

    def tearDown(self):
        safety.set_human_approve(self.old_approve)
        safety.call_reviewer_agent = self.old_reviewer
        safety._REVIEW_FAIL_COUNT = self.old_fail
        safety._REVIEW_DEGRADE_UNTIL = self.old_degrade
        if os.path.exists(self.probe):
            os.remove(self.probe)

    def test_non_frozen_workspace_write_is_not_blocked(self):
        safety.set_human_approve(lambda action: False)
        safety.call_reviewer_agent = lambda *a, **k: {"pass": True, "reason": ""}
        result = safety.process_write(self.probe, "CI_OK\n", "", "overwrite")
        self.assertIn("写入成功", result)
        with open(self.probe, encoding="utf-8") as f:
            self.assertEqual(f.read(), "CI_OK\n")

    def test_frozen_config_still_requires_approval(self):
        safety.set_human_approve(lambda action: False)
        safety.call_reviewer_agent = lambda *a, **k: {"pass": True, "reason": ""}
        path = os.path.join(ROOT, "config.json")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        result = safety.process_write(path, content, content, "noop")
        self.assertIn("需要人类批准", result)

    def test_reviewer_failure_is_fail_closed(self):
        class BadReviewer:
            def __getattr__(self, name):
                raise RuntimeError("reviewer down")

        old_client = safety.reviewer_client
        safety.reviewer_client = BadReviewer()
        safety._REVIEW_FAIL_COUNT = 0
        safety._REVIEW_DEGRADE_UNTIL = 0.0
        try:
            results = [safety.call_reviewer_agent("a=1\n", "a=2\n", "write", "x.py")
                       for _ in range(4)]
        finally:
            safety.reviewer_client = old_client
        self.assertTrue(all(r.get("pass") is False for r in results))
        self.assertTrue(results[3].get("degraded"))


class ExperienceTests(unittest.TestCase):
    def test_dedup_correct_route_and_gaps(self):
        root = tempfile.mkdtemp(prefix="ai_agent_exp_")
        old_root = os.environ.get("AI_AGENT_EXPERIENCE_ROOT")
        os.environ["AI_AGENT_EXPERIENCE_ROOT"] = root
        try:
            st = experience_store.ExperienceStore(root=root)
            first = st.record_episode(
                "检查项目状态", "读取配置和工具状态",
                [{"name": "read_file", "args": {"filepath": "config.json"}}],
                1, "success", strategy="先读配置")
            second = st.record_episode(
                "检查项目状态", "读取配置和工具状态",
                [{"name": "read_file", "args": {"filepath": "config.json"}}],
                1, "success", strategy="先读配置")
            third = st.record_episode(
                "检查项目状态", "读取配置和工具状态",
                [{"name": "read_file", "args": {"filepath": "config.json"}}],
                1, "success", strategy="先读配置")
            self.assertFalse(first["merged"])
            self.assertTrue(second["merged"])
            self.assertTrue(third["merged"])
            exp = st.get(first["id"])
            self.assertEqual(exp["evidence"], 3)
            self.assertEqual(exp["strategy"], "先读配置")

            route = st.route("检查项目状态", k=5)
            self.assertEqual(route["mode"], "direct")
            self.assertEqual(route["experience"]["id"], first["id"])

            corrected = st.correct(first["id"],
                                   action="读取配置、工具链并检查异常",
                                   lesson="先读配置再检查工具链", reason="补充步骤")
            self.assertIn("已修正", corrected)
            old = st.get(first["id"])
            self.assertEqual(old["status"], "superseded")
            new_id = old["superseded_by"]
            self.assertEqual(st.get(new_id)["status"], "active")

            gap = st.record_gap("需要 OCR", "缺少 OCR 工具", ["ocr"], "新增 OCR 能力")
            self.assertTrue(any(g["id"] == gap for g in st.pending_gaps()))
            st.resolve_gap(gap, "test")
            self.assertFalse(any(g["id"] == gap for g in st.pending_gaps()))
        finally:
            if old_root is None:
                os.environ.pop("AI_AGENT_EXPERIENCE_ROOT", None)
            else:
                os.environ["AI_AGENT_EXPERIENCE_ROOT"] = old_root
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
