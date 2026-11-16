"""experience_store.py — 分层经验记忆（Phase A/B/C 核心）。

存储：SQLite + FTS5，JSON 仅作为导出格式。
分层：
  episodes      原始任务记录（不轻易改写）
  experiences   结构化经验（指纹去重、成功率/置信度/效用）
  clusters      相似经验聚类后的模式（Pattern）
  playbooks     可复用步骤（Procedural memory）
  principles    经批准的原则（少量、高置信）
检索：FTS5 BM25 + 向量余弦 + 标签，RRF 融合 + MMR 去重 + 紧凑卡片。
嵌入：fastembed（默认 BAAI/bge-small-zh-v1.5），不可用时退回轻量哈希向量。
"""
import array
import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter

from safety import WORKSPACE_DIR, CONFIG

DEFAULT_CFG = {
    "db": "experience.db",
    "auto_save": True,
    "auto_recall": True,
    "auto_recall_min_chars": 6,
    "recall_top_k": 3,
    "recall_min_score": 0.0,
    "recall_token_budget": 700,
    "candidate_scan_limit": 50,
    "vector_candidates": 50,
    "max_entries": 500,
    "compress_timeout": 20,
    "embedding_backend": "fastembed",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "embedding_cache_dir": "~/.ai_agent_models",
    "hf_endpoint": "https://hf-mirror.com",
    "vector_dim": 512,
    "cluster_threshold": 0.82,
    "similarity_threshold": 0.90,
    "min_evidence_for_pattern": 3,
    "min_evidence_for_principle": 5,
    "consolidate_after": 10,
    "consolidate_on_save": True,
    "consolidate_limit": 5,
    "auto_feedback": True,
    "route_top_k": 5,
    "direct_enabled": True,
    "direct_min_confidence": 0.75,
    "direct_min_evidence": 3,
    "direct_min_success_rate": 0.7,
    "direct_min_score": 0.10,
    "direct_exp_min_score": 0.03,
    "suspect_utility": 0.35,
    "ab_test": {
        "enabled": False,
        "salt": "experience-v1",
        "treatment_ratio": 0.5,
        "min_samples": 10,
    },
}

_FE = {"tried": False, "failed": False, "model": None, "name": "",
       "dim": 0, "lock": threading.RLock()}
_BG = {"thread": None, "lock": threading.Lock()}


def _cfg() -> dict:
    out = dict(DEFAULT_CFG)
    user = CONFIG.get("experience", {}) or {}
    for k, v in user.items():
        out[k] = v
    return out


def _root() -> str:
    env = os.environ.get("AI_AGENT_EXPERIENCE_ROOT")
    return os.path.realpath(env) if env else os.path.join(WORKSPACE_DIR, ".experiences")


def _norm(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^\w一-鿿]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> list:
    s = (text or "").lower()
    out = re.findall(r"[a-z_][a-z0-9_]{1,}", s)
    for seg in re.findall(r"[一-鿿]+", s):
        if len(seg) == 1:
            out.append(seg)
        for i in range(len(seg) - 1):
            out.append(seg[i:i + 2])
    return out


def _search_text(*parts) -> str:
    toks = _tokens(" ".join(str(p or "") for p in parts))
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= 1200:
            break
    return " ".join(out)[:12000]


def _hash_vec(text: str, dim: int = 512) -> list:
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _cos(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _vec_bytes(vec) -> bytes:
    return array.array("f", [float(x) for x in vec]).tobytes()


def _vec_load(blob):
    if not blob:
        return []
    a = array.array("f")
    try:
        a.frombytes(blob)
    except Exception:
        return []
    return a.tolist()


def _jsonable(v):
    if isinstance(v, bytes):
        return {"__bytes_base64__": base64.b64encode(v).decode("ascii")}
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def task_hash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:16]


def _fingerprint(trigger: str, action: str, tools: list, outcome: str,
                 files: list = None) -> str:
    """相同任务意图 + 相同工具/文件 + 相同结果 → 同一经验，与措辞无关。"""
    tools = [str(x) for x in (tools or [])]
    files = [str(x) for x in (files or [])]
    parts = [_norm(trigger)[:500]]
    if tools or files:
        parts += [",".join(sorted(tools)), ",".join(sorted(files)), str(outcome)]
    else:
        parts += [_norm(action)[:300], str(outcome)]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _confidence(success: int, fail: int) -> float:
    return round((success + 1.0) / (success + fail + 2.0), 4)


def _language(files: list) -> str:
    exts = {os.path.splitext(str(f))[1].lower() for f in files or []}
    for ext, lang in ((".py", "python"), (".js", "js"), (".ts", "ts"),
                      (".tsx", "tsx"), (".cs", "csharp"), (".go", "go"),
                      (".rs", "rust"), (".java", "java"), (".c", "c"),
                      (".cpp", "cpp"), (".md", "markdown"), (".json", "json")):
        if ext in exts:
            return lang
    return ""


def _clip(text, n=300) -> str:
    s = str(text or "").replace("\n", " ").strip()
    return s[:n]


# ================= 向量后端 =================

def _fastembed_vectors(texts: list):
    cfg = _cfg()
    with _FE["lock"]:
        if _FE["failed"] or _FE["tried"] and _FE["model"] is None:
            return None
        if os.environ.get("AI_AGENT_DISABLE_EMBEDDINGS") == "1":
            _FE["name"] = "hash:" + str(cfg.get("vector_dim", 512))
            return None
        try:
            model_name = cfg.get("embedding_model") or "BAAI/bge-small-zh-v1.5"
            wanted = "fastembed:" + str(model_name)
            if _FE["model"] is not None and _FE["name"] != wanted:
                _FE["model"] = None
            if _FE["model"] is None:
                os.environ.setdefault("HF_ENDPOINT",
                                      cfg.get("hf_endpoint") or "https://hf-mirror.com")
                from fastembed import TextEmbedding
                cache = os.path.expanduser(cfg.get("embedding_cache_dir")
                                           or "~/.ai_agent_models")
                os.makedirs(cache, exist_ok=True)
                _FE["model"] = TextEmbedding(model_name=model_name, cache_dir=cache)
                _FE["name"] = wanted
                _FE["tried"] = True
            vecs = [list(map(float, v)) for v in
                    _FE["model"].embed([t[:1200] for t in texts])]
            _FE["dim"] = len(vecs[0]) if vecs else 0
            out = []
            for v in vecs:
                n = math.sqrt(sum(x * x for x in v))
                out.append([x / n for x in v] if n else v)
            return out, _FE["name"], _FE["dim"]
        except Exception as e:
            _FE["failed"] = True
            _FE["tried"] = True
            _FE["name"] = "hash:" + str(cfg.get("vector_dim", 512))
            print(f"[experience] fastembed 不可用，回退 hash 向量: {e}")
            return None


def embed_texts(texts: list):
    cfg = _cfg()
    texts = [str(t or "") for t in texts]
    backend = str(cfg.get("embedding_backend") or "fastembed").lower()
    if backend in ("fastembed", "auto"):
        got = _fastembed_vectors(texts)
        if got:
            return got
    dim = int(cfg.get("vector_dim", 512) or 512)
    _FE["name"] = f"hash:{dim}"
    return [_hash_vec(t, dim) for t in texts], _FE["name"], dim


# ================= Store =================

class ExperienceStore:
    def __init__(self, root: str = None, cfg: dict = None):
        self.cfg = dict(DEFAULT_CFG)
        self._merge_cfg()
        if cfg:
            self.cfg.update(cfg)
        self.root = os.path.realpath(root or _root())
        os.makedirs(self.root, exist_ok=True)
        db_name = os.path.basename(str(self.cfg.get("db") or "experience.db"))
        self.db_path = os.path.join(self.root, db_name)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                     timeout=30)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        self._fts_ok = True
        self._init_schema()
        self._migrate_json_once()

    def _merge_cfg(self):
        user = {k: v for k, v in (CONFIG.get("experience", {}) or {}).items()
                if not str(k).startswith("_")}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                base = dict(self.cfg[k])
                base.update({kk: vv for kk, vv in v.items()
                             if not str(kk).startswith("_")})
                self.cfg[k] = base
            else:
                self.cfg[k] = v
        # 旧版相关分是 0~N 的计分，新版是 RRF 小分；旧阈值 1.0 会误杀全部结果。
        try:
            if float(self.cfg.get("recall_min_score", 0) or 0) > 0.2:
                self.cfg["recall_min_score"] = 0.0
        except Exception:
            self.cfg["recall_min_score"] = 0.0

    def refresh_cfg(self):
        self._merge_cfg()

    # ---------- 基础 ----------
    def _q(self, sql, args=(), one=False):
        cur = self._conn.execute(sql, args)
        rows = cur.fetchall()
        return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]

    def _x(self, sql, args=()):
        cur = self._conn.execute(sql, args)
        return cur

    def _commit(self):
        self._conn.commit()

    def _meta_get(self, key, default=None):
        r = self._q("SELECT value FROM meta WHERE key=?", (key,), one=True)
        return r["value"] if r else default

    def _meta_set(self, key, value):
        with self._lock:
            self._x("INSERT INTO meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value)))
            self._commit()

    def _init_schema(self):
        with self._lock:
            c = self._conn
            c.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS episodes(
              id TEXT PRIMARY KEY, ts REAL, task_hash TEXT, user TEXT, answer TEXT,
              outcome TEXT, steps INTEGER, tokens INTEGER, tools TEXT, files TEXT,
              strategy TEXT, agent_version TEXT, trace_path TEXT);
            CREATE TABLE IF NOT EXISTS experiences(
              id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE, trigger TEXT,
              action TEXT, lesson TEXT, scope TEXT, exceptions TEXT, tags TEXT,
              files TEXT, tools TEXT, language TEXT, evidence INTEGER DEFAULT 1,
              success INTEGER DEFAULT 0, fail INTEGER DEFAULT 0,
              confidence REAL DEFAULT 0.5, utility REAL DEFAULT 0.5,
              cluster_id TEXT, status TEXT DEFAULT 'active',
              agent_version TEXT, created REAL, updated REAL,
              source_episode TEXT, search_text TEXT, vector BLOB,
              embed_model TEXT, embed_dim INTEGER, human_approved INTEGER DEFAULT 0,
              kind TEXT DEFAULT 'experience', consolidated INTEGER DEFAULT 0,
              strategy TEXT, superseded_by TEXT, corrected_from TEXT,
              revision INTEGER DEFAULT 0, last_fail_reason TEXT);
            CREATE TABLE IF NOT EXISTS clusters(
              id TEXT PRIMARY KEY, centroid BLOB, dim INTEGER, embed_model TEXT,
              trigger TEXT, action TEXT, scope TEXT, lesson TEXT, exceptions TEXT,
              tags TEXT, support INTEGER DEFAULT 0, success INTEGER DEFAULT 0,
              fail INTEGER DEFAULT 0, confidence REAL DEFAULT 0.5,
              status TEXT DEFAULT 'active', created REAL, updated REAL,
              search_text TEXT);
            CREATE TABLE IF NOT EXISTS principles(
              id TEXT PRIMARY KEY, text TEXT, scope TEXT, evidence INTEGER DEFAULT 0,
              success INTEGER DEFAULT 0, fail INTEGER DEFAULT 0,
              confidence REAL DEFAULT 0.5, status TEXT DEFAULT 'candidate',
              human_approved INTEGER DEFAULT 0, cluster_id TEXT,
              created REAL, updated REAL, search_text TEXT, vector BLOB,
              embed_model TEXT, embed_dim INTEGER);
            CREATE TABLE IF NOT EXISTS playbooks(
              id TEXT PRIMARY KEY, name TEXT, trigger TEXT, steps TEXT,
              tools TEXT, scope TEXT, evidence INTEGER DEFAULT 0,
              success INTEGER DEFAULT 0, fail INTEGER DEFAULT 0,
              confidence REAL DEFAULT 0.5, status TEXT DEFAULT 'candidate',
              human_approved INTEGER DEFAULT 0, cluster_id TEXT,
              created REAL, updated REAL, search_text TEXT, vector BLOB,
              embed_model TEXT, embed_dim INTEGER);
            CREATE TABLE IF NOT EXISTS recalls(
              id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id TEXT, task_hash TEXT,
              ts REAL, offered INTEGER DEFAULT 1, used INTEGER DEFAULT 0,
              outcome TEXT, rank REAL);
            CREATE TABLE IF NOT EXISTS episode_links(
              episode_id TEXT, exp_id TEXT, PRIMARY KEY(episode_id, exp_id));
            CREATE TABLE IF NOT EXISTS contradictions(
              a_id TEXT, b_id TEXT, reason TEXT, status TEXT DEFAULT 'open',
              created REAL, PRIMARY KEY(a_id, b_id));
            CREATE TABLE IF NOT EXISTS experiments(
              id INTEGER PRIMARY KEY AUTOINCREMENT, task_hash TEXT, arm TEXT,
              ts REAL, outcome TEXT, latency_s REAL, steps INTEGER,
              recalled INTEGER DEFAULT 0, used INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS route_runs(
              id INTEGER PRIMARY KEY AUTOINCREMENT, playbook_id TEXT,
              task_hash TEXT, ts REAL, outcome TEXT, steps INTEGER);
            CREATE TABLE IF NOT EXISTS capability_gaps(
              id TEXT PRIMARY KEY, ts REAL, task_hash TEXT, user TEXT,
              reason TEXT, missing_tools TEXT, suggested_goal TEXT,
              status TEXT DEFAULT 'open', resolved_by TEXT, updated REAL);
            CREATE INDEX IF NOT EXISTS idx_exp_fp ON experiences(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_exp_status ON experiences(status);
            CREATE INDEX IF NOT EXISTS idx_exp_cluster ON experiences(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_recalls_task ON recalls(task_hash);
            CREATE INDEX IF NOT EXISTS idx_recalls_exp ON recalls(exp_id);
            CREATE INDEX IF NOT EXISTS idx_gaps_status ON capability_gaps(status);
            """)
            # 旧库原地升级：SQLite 没有 CREATE TABLE 的列迁移语法
            for table, cols in (
                    ("episodes", {"strategy": "TEXT"}),
                    ("experiences", {
                        "strategy": "TEXT", "superseded_by": "TEXT",
                        "corrected_from": "TEXT", "revision": "INTEGER DEFAULT 0",
                        "last_fail_reason": "TEXT"}),):
                try:
                    have = {r["name"] for r in self._q(
                        f"PRAGMA table_info({table})")}
                except Exception:
                    have = set()
                for col, ddl in cols.items():
                    if col not in have:
                        try:
                            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                        except Exception as e:
                            print(f"[experience] 迁移列失败 {table}.{col}: {e}")
            try:
                c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS experience_fts "
                          "USING fts5(exp_id UNINDEXED, search_text, "
                          "tokenize='unicode61')")
            except Exception as e:
                self._fts_ok = False
                print(f"[experience] FTS5 不可用，退回 LIKE 检索: {e}")
            self._commit()

    # ---------- FTS ----------
    def _fts_upsert(self, exp_id: str, search_text: str):
        if not self._fts_ok:
            return
        try:
            with self._lock:
                self._x("DELETE FROM experience_fts WHERE exp_id=?", (exp_id,))
                self._x("INSERT INTO experience_fts(exp_id,search_text) VALUES(?,?)",
                        (exp_id, search_text))
                self._commit()
        except Exception:
            self._fts_ok = False

    def _fts_delete(self, exp_id: str):
        if not self._fts_ok:
            return
        try:
            with self._lock:
                self._x("DELETE FROM experience_fts WHERE exp_id=?", (exp_id,))
                self._commit()
        except Exception:
            self._fts_ok = False

    def _fts_ranks(self, query: str, limit: int) -> list:
        if not self._fts_ok:
            return []
        toks = list(dict.fromkeys(_tokens(query)))[:30]
        if not toks:
            return []
        match = " OR ".join('"' + t.replace('"', '') + '"' for t in toks)
        try:
            with self._lock:
                rows = self._q(
                    "SELECT exp_id, bm25(experience_fts) AS s "
                    "FROM experience_fts WHERE experience_fts MATCH ? "
                    "ORDER BY s LIMIT ?", (match, limit))
            return [(r["exp_id"], float(r["s"])) for r in rows]
        except Exception:
            return []

    # ---------- 向量 ----------
    def _embed_and_store(self, rows, limit=200):
        need = []
        for r in rows:
            if not r.get("vector") or r.get("embed_model") != _FE.get("name"):
                need.append(r)
            if len(need) >= limit:
                break
        if not need:
            return
        texts = [r.get("search_text") or r.get("trigger") or "" for r in need]
        vecs, model, dim = embed_texts(texts)
        with self._lock:
            for r, v in zip(need, vecs):
                self._x("UPDATE experiences SET vector=?, embed_model=?, embed_dim=? "
                        "WHERE id=?", (_vec_bytes(v), model, dim, r["id"]))
            self._commit()

    def _query_vector(self, text: str):
        vecs, model, dim = embed_texts([text])
        return vecs[0], model, dim

    # ---------- 迁移 ----------
    def _migrate_json_once(self):
        if self._meta_get("migrated_json_v1") == "1":
            return
        count = 0
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if not d.startswith(".")]
            for fn in fns:
                if not fn.endswith(".json") or fn.startswith("_"):
                    continue
                full = os.path.join(dp, fn)
                if full == self.db_path or full.endswith(".tmp"):
                    continue
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        e = json.load(f)
                except Exception:
                    continue
                if not isinstance(e, dict) or not e.get("id"):
                    continue
                try:
                    self._import_legacy(e)
                    count += 1
                except Exception:
                    pass
        self._meta_set("migrated_json_v1", "1")
        if count:
            print(f"[experience] 已迁移 {count} 条旧经验到 SQLite")

    def import_legacy(self, entry: dict) -> str:
        """兼容旧 JSONL / 旧插件调用。"""
        return self._import_legacy(entry)

    def _import_legacy(self, e: dict):
        ts = float(e.get("ts") or time.time())
        user = str(e.get("user") or "")
        summary = str(e.get("summary") or "")
        lessons = e.get("lessons") or []
        tags = [str(x) for x in (e.get("tags") or [])]
        files = [str(x) for x in (e.get("files_touched") or [])]
        tools = [str(x) for x in (e.get("tools") or [])]
        outcome = str(e.get("outcome") or "success")
        exp_id = "exp_" + hashlib.sha256(
            (str(e.get("id")) + user + summary).encode("utf-8")).hexdigest()[:16]
        trigger = user[:500]
        action = summary[:500]
        lesson = "\n".join(str(x) for x in lessons[:5])[:1000]
        strategy = str(e.get("strategy") or "")[:2000]
        scope = ", ".join(files[:10])
        fp = _fingerprint(trigger, action, tools, outcome, files)
        old = self._q("SELECT * FROM experiences WHERE fingerprint=?", (fp,), one=True)
        if old:
            self._merge_experience(old, outcome, tags, files, tools,
                                   source_ep=None, trigger=trigger, action=action,
                                   lesson=lesson, scope=scope, strategy=strategy)
            return old["id"]
        success = 1 if outcome == "success" else 0
        fail = 0 if outcome == "success" else 1
        search = _search_text(trigger, action, lesson, scope, strategy,
                              " ".join(tags), " ".join(files), " ".join(tools))
        now = time.time()
        with self._lock:
            self._x("""INSERT OR IGNORE INTO experiences(
                id,fingerprint,trigger,action,lesson,scope,exceptions,tags,files,
                tools,language,evidence,success,fail,confidence,utility,
                status,created,updated,search_text,embed_model,embed_dim,
                human_approved,kind,consolidated,strategy)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (exp_id, fp, trigger, action, lesson, scope, "", json.dumps(tags),
                 json.dumps(files), json.dumps(tools), _language(files), 1,
                 success, fail, _confidence(success, fail), 0.5, "active",
                 ts, now, search, "", 0, 1, "experience", 1, strategy))
            self._commit()
        self._fts_upsert(exp_id, search)
        return exp_id

    # ---------- 写入 ----------
    def _merge_experience(self, old: dict, outcome: str, tags, files, tools,
                          source_ep, trigger="", action="", lesson="",
                          scope="", strategy=""):
        exp_id = old["id"]
        success = int(old.get("success") or 0) + (1 if outcome == "success" else 0)
        fail = int(old.get("fail") or 0) + (0 if outcome == "success" else 1)
        evidence = int(old.get("evidence") or 1) + 1
        old_tags = json.loads(old.get("tags") or "[]")
        old_files = json.loads(old.get("files") or "[]")
        old_tools = json.loads(old.get("tools") or "[]")
        tags = list(dict.fromkeys([str(x) for x in old_tags + list(tags or [])]))[:20]
        files = list(dict.fromkeys([str(x) for x in old_files + list(files or [])]))[:20]
        tools = list(dict.fromkeys([str(x) for x in old_tools + list(tools or [])]))[:20]
        new_action = action if (action and not old.get("action")) else old.get("action")
        new_lesson = lesson if (lesson and not old.get("lesson")) else old.get("lesson")
        new_strategy = (strategy if (strategy and not old.get("strategy"))
                        else old.get("strategy") or "")
        search = _search_text(old.get("trigger"), new_action, new_lesson,
                              old.get("scope"), new_strategy, " ".join(tags),
                              " ".join(files), " ".join(tools))
        with self._lock:
            self._x("""UPDATE experiences SET evidence=?, success=?, fail=?,
                confidence=?, tags=?, files=?, tools=?, action=?, lesson=?,
                strategy=?, updated=?, source_episode=COALESCE(?,source_episode),
                search_text=?, vector=NULL, embed_model='', embed_dim=0
                WHERE id=?""",
                (evidence, success, fail, _confidence(success, fail),
                 json.dumps(tags), json.dumps(files), json.dumps(tools),
                 new_action, new_lesson, new_strategy, time.time(), source_ep,
                 search, exp_id))
            self._commit()
        self._fts_upsert(exp_id, search)
        return exp_id

    def record_episode(self, user: str, answer: str, tool_calls: list = None,
                       steps: int = 0, outcome: str = "success",
                       trace_id: str = "", tokens: int = 0,
                       agent_version: str = "14", force: bool = False,
                       strategy: str = "") -> dict:
        cfg = self.cfg
        if not force and not cfg.get("auto_save", True):
            return {"id": "", "merged": False, "skipped": "auto_save_disabled"}
        tool_calls = tool_calls or []
        tools, files = [], []
        for t in tool_calls:
            name = t.get("name") if isinstance(t, dict) else str(t)
            if name and name not in tools:
                tools.append(str(name))
            args = (t.get("args") or {}) if isinstance(t, dict) else {}
            for k in ("filepath", "path", "script_path", "sln_path"):
                v = args.get(k)
                if v and str(v) not in files:
                    files.append(str(v))
        trigger = str(user or "")[:2000]
        action = str(answer or "")[:2000]
        strategy = str(strategy or "")[:4000]
        tags = list(dict.fromkeys(
            tools + [outcome] + ([_language(files)] if _language(files) else [])))[:12]
        fp = _fingerprint(trigger, action, tools, outcome, files)
        now = time.time()
        eid = "ep_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        task = task_hash(trigger)
        search = _search_text(trigger, action, strategy, " ".join(tags),
                              " ".join(files), " ".join(tools))
        with self._lock:
            self._x("""INSERT INTO episodes(id,ts,task_hash,user,answer,outcome,
                steps,tokens,tools,files,strategy,agent_version,trace_path)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (eid, now, task, trigger, action, outcome, int(steps),
                 int(tokens), json.dumps(tools), json.dumps(files), strategy,
                 agent_version, trace_id))
            old = self._q("SELECT * FROM experiences WHERE fingerprint=?",
                          (fp,), one=True)
            self._commit()
        if old:
            exp_id = self._merge_experience(old, outcome, tags, files, tools,
                                            source_ep=eid, trigger=trigger,
                                            action=action, strategy=strategy)
            with self._lock:
                self._x("INSERT OR IGNORE INTO episode_links(episode_id,exp_id) "
                        "VALUES(?,?)", (eid, exp_id))
                self._commit()
            self._enforce_limit()
            return {"id": exp_id, "episode_id": eid, "merged": True}

        exp_id = "exp_" + fp[:16]
        success = 1 if outcome == "success" else 0
        fail = 0 if outcome == "success" else 1
        with self._lock:
            self._x("""INSERT OR IGNORE INTO experiences(
                id,fingerprint,trigger,action,lesson,scope,exceptions,tags,files,
                tools,language,evidence,success,fail,confidence,utility,status,
                created,updated,source_episode,search_text,embed_model,embed_dim,
                human_approved,kind,consolidated,strategy)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (exp_id, fp, trigger, action, "", ", ".join(files[:10]), "",
                 json.dumps(tags), json.dumps(files), json.dumps(tools),
                 _language(files), 1, success, fail,
                 _confidence(success, fail), 0.5, "active",
                 now, now, eid, search, "", 0, 0, "experience", 0, strategy))
            self._x("INSERT OR IGNORE INTO episode_links(episode_id,exp_id) "
                    "VALUES(?,?)", (eid, exp_id))
            self._commit()
        self._fts_upsert(exp_id, search)
        self._enforce_limit()
        self._schedule_consolidate()
        return {"id": exp_id, "episode_id": eid, "merged": False}

    # ---------- 反馈 ----------
    def task_hash(self, text: str) -> str:
        return task_hash(text)

    def record_recall(self, exp_ids: list, th: str, rank: float = 0.0,
                      used: bool = False):
        ids = [x for x in exp_ids if str(x).startswith("exp_")]
        if not ids:
            return
        now = time.time()
        with self._lock:
            for eid in ids:
                self._x("INSERT INTO recalls(exp_id,task_hash,ts,offered,used,"
                        "outcome,rank) VALUES(?,?,?,?,?,?,?)",
                        (eid, th, now, 1, 1 if used else 0, None, rank))
            self._commit()

    def record_recall_outcome(self, th: str, outcome: str):
        with self._lock:
            rows = self._q("SELECT DISTINCT exp_id FROM recalls WHERE task_hash=?",
                           (th,))
            self._x("UPDATE recalls SET outcome=? WHERE task_hash=? AND outcome IS NULL",
                    (outcome, th))
            self._commit()
        for r in rows:
            self._recompute_utility(r["exp_id"])

    def feedback(self, exp_ids: list, used: bool = True, outcome: str = "",
                 th: str = ""):
        exp_ids = [str(x) for x in exp_ids if str(x).startswith("exp_")]
        if not exp_ids:
            return "无有效 experience id"
        now = time.time()
        with self._lock:
            for eid in exp_ids:
                self._x("INSERT INTO recalls(exp_id,task_hash,ts,offered,used,"
                        "outcome,rank) VALUES(?,?,?,?,?,?,?)",
                        (eid, th or "manual", now, 1, 1 if used else 0,
                         outcome or None, 0))
            self._commit()
        for eid in exp_ids:
            self._recompute_utility(eid)
        return f"✅ 已记录 {len(exp_ids)} 条反馈"

    def experiment_arm(self, user: str) -> str:
        """确定性 A/B 分桶（同一任务描述稳定在同一臂）；未启用返回 ""。"""
        ab = self.cfg.get("ab_test") or {}
        if not ab.get("enabled", False):
            return ""
        try:
            ratio = float(ab.get("treatment_ratio", 0.5))
        except Exception:
            ratio = 0.5
        raw = f"{ab.get('salt', 'experience-v1')}|{task_hash(user)}"
        h = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16) % 10000
        return "treatment" if h < int(ratio * 10000) else "control"

    def record_experiment(self, th: str, arm: str, outcome: str,
                          latency_s: float = 0.0, steps: int = 0,
                          recalled: int = 0, used: int = 0):
        if not arm:
            return
        with self._lock:
            self._x("INSERT INTO experiments(task_hash,arm,ts,outcome,latency_s,"
                    "steps,recalled,used) VALUES(?,?,?,?,?,?,?,?)",
                    (th, arm, time.time(), outcome, float(latency_s),
                     int(steps), int(recalled), int(used)))
            self._commit()

    def record_playbook_outcome(self, playbook_id: str, outcome: str,
                                th: str = "", steps: int = 0):
        if not playbook_id:
            return
        with self._lock:
            row = self._q("SELECT * FROM playbooks WHERE id=?",
                          (playbook_id,), one=True)
            if not row:
                return
            succ = int(row.get("success") or 0) + (1 if outcome == "success" else 0)
            fail = int(row.get("fail") or 0) + (0 if outcome == "success" else 1)
            status = row.get("status") or "approved"
            if (status == "approved" and fail >= 2
                    and succ / max(1, succ + fail) < 0.6):
                status = "candidate"   # 连续失败：撤销自动直行，等人工复核
            self._x("UPDATE playbooks SET success=?,fail=?,confidence=?,status=?,"
                    "updated=? WHERE id=?",
                    (succ, fail, _confidence(succ, fail), status, time.time(),
                     playbook_id))
            self._x("INSERT INTO route_runs(playbook_id,task_hash,ts,outcome,steps)"
                    " VALUES(?,?,?,?,?)",
                    (playbook_id, th, time.time(), outcome, int(steps)))
            self._commit()

    def metrics(self) -> dict:
        with self._lock:
            exps = self._q("SELECT arm,outcome,latency_s,steps,recalled,used "
                           "FROM experiments")
            rec = self._q("SELECT outcome,used FROM recalls")
            routes = self._q("SELECT outcome,steps FROM route_runs")
            us = self._q("SELECT AVG(utility) a FROM experiences", one=True)
        arms = {}
        for r in exps:
            a = arms.setdefault(r["arm"], {"n": 0, "success": 0, "fail": 0,
                                           "latency": 0.0, "steps": 0,
                                           "recalled": 0, "used": 0})
            a["n"] += 1
            if r.get("outcome") == "success":
                a["success"] += 1
            else:
                a["fail"] += 1
            a["latency"] += float(r.get("latency_s") or 0)
            a["steps"] += int(r.get("steps") or 0)
            a["recalled"] += int(r.get("recalled") or 0)
            a["used"] += int(r.get("used") or 0)
        for a in arms.values():
            n = max(1, a["n"])
            a["success_rate"] = round(a["success"] / n, 4)
            a["avg_latency_s"] = round(a["latency"] / n, 3)
            a["avg_steps"] = round(a["steps"] / n, 2)
            a["used_rate"] = round(a["used"] / max(1, a["recalled"]), 4)
        offered = len(rec)
        used_n = sum(1 for r in rec if r.get("used"))
        direct_n = len(routes)
        direct_ok = sum(1 for r in routes if r.get("outcome") == "success")
        return {
            "arms": arms,
            "recall": {"offered": offered, "used": used_n,
                       "use_rate": round(used_n / offered, 4) if offered else 0.0},
            "direct": {"runs": direct_n, "success": direct_ok,
                       "success_rate": round(direct_ok / direct_n, 4) if direct_n else 0.0},
            "avg_utility": round(float((us or {}).get("a") or 0.5), 4),
            "ab_min_samples": int((self.cfg.get("ab_test") or {}).get("min_samples", 10) or 10),
            "ab_enabled": bool((self.cfg.get("ab_test") or {}).get("enabled", False)),
        }

    def _recompute_utility(self, exp_id: str):
        with self._lock:
            rows = self._q("SELECT outcome,used FROM recalls WHERE exp_id=? "
                           "AND outcome IS NOT NULL", (exp_id,))
            exp = self._q("SELECT status FROM experiences WHERE id=?",
                          (exp_id,), one=True)
            n = len(rows)
            s = sum(1 for r in rows if r["outcome"] == "success")
            used_n = sum(1 for r in rows if int(r.get("used") or 0))
            fails = sum(1 for r in rows if r["outcome"] != "success")
            util = (s + 1.0) / (n + 2.0) if n else 0.5
            status = (exp or {}).get("status") or "active"
            try:
                threshold = float(self.cfg.get("suspect_utility", 0.35))
            except Exception:
                threshold = 0.35
            if (status in ("active", "suspect") and n >= 2
                    and used_n > 0 and util <= threshold):
                status = "suspect"
            reason = "最近反馈失败" if fails else ""
            self._x("""UPDATE experiences SET utility=?, status=?,
                last_fail_reason=COALESCE(NULLIF(?, ''), last_fail_reason)
                WHERE id=?""", (util, status, reason, exp_id))
            self._commit()

    # ---------- 召回 ----------
    def _card_exp(self, row: dict, score: float = 0.0) -> dict:
        try:
            tags = json.loads(row.get("tags") or "[]")
        except Exception:
            tags = []
        return {
            "id": row.get("id"), "kind": "experience",
            "trigger": row.get("trigger") or "",
            "action": row.get("action") or "",
            "lesson": row.get("lesson") or "",
            "strategy": row.get("strategy") or "",
            "scope": row.get("scope") or "",
            "exceptions": row.get("exceptions") or "",
            "status": row.get("status") or "active",
            "revision": int(row.get("revision") or 0),
            "corrected_from": row.get("corrected_from") or "",
            "tags": tags, "confidence": float(row.get("confidence") or 0.5),
            "evidence": int(row.get("evidence") or 0),
            "success": int(row.get("success") or 0),
            "fail": int(row.get("fail") or 0),
            "utility": float(row.get("utility") or 0.5),
            "score": round(score, 5),
            "_vec": _vec_load(row.get("vector")),
            "_model": row.get("embed_model") or "",
        }

    def recall(self, query: str, k: int = None, min_score: float = None,
               token_budget: int = None, top_k: int = None) -> list:
        cfg = self.cfg
        q = str(query or "").strip()
        if not q:
            return []
        k = int(k or top_k or cfg.get("recall_top_k", 3) or 3)
        min_score = (float(cfg.get("recall_min_score", 0.0))
                     if min_score is None else float(min_score))
        limit = int(cfg.get("candidate_scan_limit", 50) or 50)
        vlimit = int(cfg.get("vector_candidates", 50) or 50)
        with self._lock:
            rows = self._q("SELECT * FROM experiences "
                           "WHERE status IN ('active','suspect')")

        # FTS
        fts = self._fts_ranks(q, limit)
        fts_rank = {eid: i + 1 for i, (eid, _) in enumerate(fts)}
        cand = set(fts_rank)

        # 向量
        qv, qm, qd = self._query_vector(q)
        by_id = {r["id"]: r for r in rows}
        self._embed_and_store(list(by_id.values()), limit=200)
        # _embed_and_store 会更新 DB；reload 才能拿到本轮刚写入的向量
        with self._lock:
            by_id = {r["id"]: r for r in self._q(
                "SELECT * FROM experiences "
                "WHERE status IN ('active','suspect')")}
        vec_scores = []
        for eid, r in by_id.items():
            if r.get("embed_model") != qm:
                continue
            v = _vec_load(r.get("vector"))
            if v:
                vec_scores.append((_cos(qv, v), eid))
        vec_scores.sort(reverse=True)
        vec_rank = {eid: i + 1 for i, (_, eid) in enumerate(vec_scores[:vlimit])}
        cand |= set(vec_rank)

        # 标签/搜索文本兜底
        tag_rank, tag_weight = {}, {}
        idf_sum = 1.0
        toks = list(dict.fromkeys(_tokens(q)))[:12]
        if toks:
            df = {t: 0 for t in toks}
            for r in by_id.values():
                text = (r.get("search_text") or "") + " " + (r.get("tags") or "")
                for t in toks:
                    if t in text:
                        df[t] += 1
            n_docs = max(1, len(by_id))
            idf = {t: math.log((n_docs + 1) / (df[t] + 1)) + 1.0 for t in toks}
            idf_sum = sum(idf.values()) or 1.0
            for eid, r in by_id.items():
                text = (r.get("search_text") or "") + " " + (r.get("tags") or "")
                w = sum(idf[t] for t in toks if t in text)
                if w:
                    tag_rank[eid] = -w
                    tag_weight[eid] = w
                    cand.add(eid)
        ranked_tags = {eid: i + 1 for i, (eid, _) in
                       enumerate(sorted(tag_rank.items(), key=lambda kv: kv[1]))}

        cards = []
        for eid in cand:
            r = by_id.get(eid)
            if not r:
                continue
            score = 0.0
            if eid in fts_rank:
                score += 0.55 / (60 + fts_rank[eid])
            if eid in vec_rank:
                score += 0.30 / (60 + vec_rank[eid])
            if eid in ranked_tags:
                score += 0.10 / (60 + ranked_tags[eid])
            if eid in tag_weight:
                score += 0.04 * tag_weight[eid] / idf_sum
            score += 0.03 * float(r.get("confidence") or 0.5)
            score += 0.02 * float(r.get("utility") or 0.5)
            if (r.get("status") or "active") == "suspect":
                score *= 0.55
            if score < min_score:
                continue
            cards.append(self._card_exp(r, score))

        # Pattern / Playbook / Principle
        cards.extend(self._recall_abstract(q, qv, qm, min_score))

        # MMR：相关性 + 多样性
        cards.sort(key=lambda c: c.get("score", 0), reverse=True)
        selected = []
        while cards and len(selected) < k:
            best, best_val = None, -1e9
            for c in cards:
                rel = float(c.get("score") or 0)
                sim = 0.0
                for s in selected:
                    if c.get("_vec") and s.get("_vec"):
                        sim = max(sim, _cos(c["_vec"], s["_vec"]))
                val = rel - 0.25 * sim
                if val > best_val:
                    best, best_val = c, val
            selected.append(best)
            cards.remove(best)
        for c in selected:
            c.pop("_vec", None)
            c.pop("_model", None)
        return selected

    def route(self, query: str, k: int = None, token_budget: int = None) -> dict:
        """返回 direct / assist / none；direct 仅限已批准的高置信 playbook。"""
        cfg = self.cfg
        token_budget = int(token_budget or cfg.get("recall_token_budget", 700) or 700)
        cards = self.recall(query, k=k or cfg.get("route_top_k", 5),
                            token_budget=token_budget)
        if not cards:
            return {"mode": "none", "cards": [], "text": "",
                    "playbook": None, "experience": None, "reason": "no_match"}
        best_play, best_exp = None, None
        if cfg.get("direct_enabled", True):
            try:
                min_conf = float(cfg.get("direct_min_confidence", 0.75))
                min_ev = int(cfg.get("direct_min_evidence", 3))
                min_rate = float(cfg.get("direct_min_success_rate", 0.7))
                min_score = float(cfg.get("direct_min_score", 0.10))
                exp_min_score = float(cfg.get("direct_exp_min_score", 0.03))
            except Exception:
                min_conf, min_ev, min_rate = 0.75, 3, 0.7
                min_score, exp_min_score = 0.10, 0.03
            for c in cards:
                if c.get("kind") != "playbook" or not c.get("steps"):
                    continue
                if str(c.get("status") or "approved") != "approved":
                    continue
                succ, fail = int(c.get("success") or 0), int(c.get("fail") or 0)
                rate = succ / max(1, succ + fail)
                if (float(c.get("confidence") or 0) >= min_conf
                        and int(c.get("evidence") or 0) >= min_ev
                        and rate >= min_rate
                        and float(c.get("score") or 0) >= min_score):
                    best_play = c
                    break
            if best_play is None:
                for c in cards:
                    if c.get("kind") != "experience":
                        continue
                    if str(c.get("status") or "active") != "active":
                        continue
                    succ, fail = int(c.get("success") or 0), int(c.get("fail") or 0)
                    rate = succ / max(1, succ + fail)
                    if (float(c.get("confidence") or 0) >= min_conf
                            and int(c.get("evidence") or 0) >= min_ev
                            and rate >= min_rate
                            and float(c.get("score") or 0) >= exp_min_score
                            and (c.get("lesson") or c.get("action")
                                 or c.get("strategy"))):
                        best_exp = c
                        break
        text = self.format_cards(cards, token_budget=token_budget)
        direct = best_play or best_exp
        if best_play:
            reason = "approved_playbook"
        elif best_exp:
            reason = "high_confidence_experience"
        else:
            reason = "advisory_cards"
        return {"mode": "direct" if direct else "assist", "cards": cards,
                "text": text, "playbook": best_play, "experience": best_exp,
                "reason": reason}

    def _ensure_abstract_vectors(self, model: str):
        """嵌入后端切换/升级后，重算模式与 playbook/principle 向量。"""
        jobs = []
        with self._lock:
            for table, vec_col, dim_col in (("clusters", "centroid", "dim"),
                                            ("playbooks", "vector", "embed_dim"),
                                            ("principles", "vector", "embed_dim")):
                rows = self._q(f"SELECT id,search_text,{vec_col} AS v,embed_model "
                               f"FROM {table} WHERE embed_model!=? OR {vec_col} IS NULL",
                               (model,))
                for r in rows:
                    text = r.get("search_text") or ""
                    if text:
                        jobs.append((table, vec_col, dim_col, r["id"], text))
        jobs = jobs[:120]
        for start in range(0, len(jobs), 32):
            batch = jobs[start:start + 32]
            vecs, m, dim = embed_texts([x[4] for x in batch])
            with self._lock:
                for (table, vec_col, dim_col, rid, _), vec in zip(batch, vecs):
                    self._x(f"UPDATE {table} SET {vec_col}=?,embed_model=?,"
                            f"{dim_col}=? WHERE id=?",
                            (_vec_bytes(vec), m, dim, rid))
                self._commit()

    def _recall_abstract(self, query: str, qv, qm, min_score=0.0) -> list:
        self._ensure_abstract_vectors(qm)
        q_tokens = set(_tokens(query))
        out = []

        def _lex(*parts):
            if not q_tokens:
                return 0.0
            tt = set(_tokens(" ".join(str(p or "") for p in parts)))
            return len(q_tokens & tt) / max(1, len(q_tokens))

        with self._lock:
            clusters = self._q("SELECT * FROM clusters WHERE status='active' "
                               "AND lesson!=''")
            plays = self._q("SELECT * FROM playbooks WHERE status='approved'")
            prins = self._q("SELECT * FROM principles WHERE status='approved'")
        for r in clusters:
            v = _vec_load(r.get("centroid"))
            if not v or r.get("embed_model") != qm:
                continue
            sim = _cos(qv, v)
            lex = _lex(r.get("trigger"), r.get("action"), r.get("lesson"),
                       r.get("scope"), r.get("exceptions"), r.get("tags"))
            if sim < 0.45 and lex < 0.25:
                continue
            out.append({
                "id": r["id"], "kind": "pattern", "trigger": r.get("trigger") or "",
                "action": r.get("action") or "", "lesson": r.get("lesson") or "",
                "scope": r.get("scope") or "", "exceptions": r.get("exceptions") or "",
                "confidence": float(r.get("confidence") or 0.5),
                "evidence": int(r.get("support") or 0),
                "success": int(r.get("success") or 0), "fail": int(r.get("fail") or 0),
                "score": round(0.12 + 0.10 * max(sim, lex), 5), "_vec": v,
            })
        for r in plays + prins:
            v = _vec_load(r.get("vector") or r.get("centroid"))
            if not v or r.get("embed_model") != qm:
                continue
            sim = _cos(qv, v)
            is_play = "name" in r
            lex = _lex(r.get("name"), r.get("trigger"), r.get("text"),
                       r.get("scope"), r.get("steps"))
            if sim < 0.45 and lex < 0.25:
                continue
            out.append({
                "id": r["id"], "kind": "playbook" if is_play else "principle",
                "name": r.get("name") or "",
                "trigger": (r.get("trigger") or "") if is_play else "",
                "action": "",
                "lesson": (r.get("text") or "") if not is_play else "",
                "scope": r.get("scope") or "", "exceptions": "",
                "steps": json.loads(r.get("steps") or "[]") if is_play else [],
                "confidence": float(r.get("confidence") or 0.5),
                "evidence": int(r.get("evidence") or 0),
                "success": int(r.get("success") or 0), "fail": int(r.get("fail") or 0),
                "score": round(0.14 + 0.10 * max(sim, lex), 5), "_vec": v,
            })
        return out

    def format_cards(self, cards: list, token_budget: int = None) -> str:
        cfg = self.cfg
        budget = int(token_budget or cfg.get("recall_token_budget", 700) or 700)
        try:
            from safety import estimate_tokens
        except Exception:
            def estimate_tokens(x):
                return max(1, len(str(x)) // 2)
        lines, used = [], 0
        for c in cards:
            block = [f"[{c.get('kind', 'experience')}] {c.get('id')} "
                     f"置信={c.get('confidence', 0):.2f} "
                     f"证据={c.get('evidence', 0)} "
                     f"成功/失败={c.get('success', 0)}/{c.get('fail', 0)}"
                     + (f" 状态={c['status']}" if c.get("status") == "suspect" else "")]
            if c.get("name"):
                block.append(f"名称: {_clip(c['name'], 160)}")
            if c.get("trigger"):
                block.append(f"触发: {_clip(c['trigger'], 240)}")
            if c.get("action"):
                block.append(f"做法: {_clip(c['action'], 320)}")
            if c.get("strategy"):
                block.append(f"策略: {_clip(c['strategy'], 320)}")
            if c.get("lesson"):
                block.append(f"经验: {_clip(c['lesson'], 400)}")
            if c.get("scope"):
                block.append(f"范围: {_clip(c['scope'], 200)}")
            if c.get("exceptions"):
                block.append(f"反例/边界: {_clip(c['exceptions'], 200)}")
            if c.get("steps"):
                try:
                    steps = c["steps"][:6]
                    block.append("步骤: " + "; ".join(
                        f"{i+1}){str(s.get('goal') or s)[:80]}"
                        + (f"[{s.get('tool')}]" if isinstance(s, dict) and s.get('tool') else "")
                        for i, s in enumerate(steps)))
                except Exception:
                    pass
            text = "\n".join(block)
            n = estimate_tokens(text) + 2
            if lines and used + n > budget:
                break
            lines.append(text)
            used += n
        return "\n\n".join(lines)

    # ---------- 查询/维护 ----------
    def get(self, exp_id: str) -> dict:
        with self._lock:
            r = self._q("SELECT * FROM experiences WHERE id=?", (exp_id,), one=True)
            if r:
                try:
                    r["tags"] = json.loads(r.get("tags") or "[]")
                    r["files"] = json.loads(r.get("files") or "[]")
                    r["tools"] = json.loads(r.get("tools") or "[]")
                except Exception:
                    pass
                r["episodes"] = self._q(
                    "SELECT e.* FROM episodes e JOIN episode_links l "
                    "ON e.id=l.episode_id WHERE l.exp_id=? ORDER BY e.ts DESC LIMIT 5",
                    (exp_id,))
                r["contradictions"] = self._q(
                    "SELECT a_id,b_id,reason,status FROM contradictions "
                    "WHERE status='open' AND (a_id=? OR b_id=?)", (exp_id, exp_id))
                r.pop("vector", None)
                return r
            for table, kind in (("clusters", "pattern"), ("playbooks", "playbook"),
                                ("principles", "principle")):
                r = self._q(f"SELECT * FROM {table} WHERE id=?", (exp_id,), one=True)
                if r:
                    r["kind"] = kind
                    r.pop("vector", None)
                    r.pop("centroid", None)
                    if r.get("steps"):
                        try:
                            r["steps"] = json.loads(r["steps"])
                        except Exception:
                            pass
                    return r
        return {}

    def add(self, entry: dict) -> str:
        """兼容旧插件：把一条旧式 entry 导入分层库。"""
        return self._import_legacy(entry)

    def delete(self, exp_id: str) -> bool:
        return self.forget(exp_id)

    def list_recent(self, n=10) -> list:
        with self._lock:
            return self._q("SELECT id,trigger,action,confidence,evidence,success,fail,"
                           "status,created,updated FROM experiences "
                           "ORDER BY updated DESC LIMIT ?", (int(n),))

    def stats(self) -> dict:
        with self._lock:
            exp = self._q("SELECT COUNT(*) c FROM experiences", one=True)["c"]
            eps = self._q("SELECT COUNT(*) c FROM episodes", one=True)["c"]
            cl = self._q("SELECT COUNT(*) c FROM clusters", one=True)["c"]
            pb = self._q("SELECT COUNT(*) c FROM playbooks", one=True)["c"]
            pr = self._q("SELECT COUNT(*) c FROM principles", one=True)["c"]
            rec = self._q("SELECT COUNT(*) c FROM recalls", one=True)["c"]
            con = self._q("SELECT COUNT(*) c FROM contradictions WHERE status='open'",
                          one=True)["c"]
            un = self._q("SELECT COUNT(*) c FROM experiences WHERE consolidated=0",
                         one=True)["c"]
            sus = self._q("SELECT COUNT(*) c FROM experiences "
                          "WHERE status='suspect'", one=True)["c"]
            gaps = self._q("SELECT COUNT(*) c FROM capability_gaps "
                           "WHERE status='open'", one=True)["c"]
            tags = Counter()
            for r in self._q("SELECT tags FROM experiences"):
                try:
                    tags.update(json.loads(r.get("tags") or "[]"))
                except Exception:
                    pass
            return {
                "experiences": exp, "episodes": eps, "patterns": cl,
                "playbooks": pb, "principles": pr, "recalls": rec,
                "contradictions": con, "unconsolidated": un,
                "suspect": sus, "gaps": gaps,
                "top_tags": tags.most_common(10), "db": self.db_path,
            }

    def forget(self, exp_id: str) -> bool:
        with self._lock:
            r = self._q("SELECT id FROM experiences WHERE id=?", (exp_id,), one=True)
            self._x("DELETE FROM experiences WHERE id=?", (exp_id,))
            self._x("DELETE FROM episode_links WHERE exp_id=?", (exp_id,))
            self._x("DELETE FROM recalls WHERE exp_id=?", (exp_id,))
            self._x("DELETE FROM contradictions WHERE a_id=? OR b_id=?",
                    (exp_id, exp_id))
            self._commit()
        self._fts_delete(exp_id)
        return bool(r)

    def _enforce_limit(self):
        try:
            max_n = int(self.cfg.get("max_entries", 500) or 500)
        except Exception:
            max_n = 500
        if max_n <= 0:
            return
        with self._lock:
            n = self._q("SELECT COUNT(*) c FROM experiences", one=True)["c"]
            if n <= max_n:
                return
            rows = self._q("SELECT id FROM experiences ORDER BY created ASC LIMIT ?",
                           (n - max_n,))
        for r in rows:
            self.forget(r["id"])

    def merge(self, ids: list) -> str:
        ids = [x for x in ids if x]
        if len(ids) < 2:
            return "至少需要两个 id"
        with self._lock:
            rows = [self._q("SELECT * FROM experiences WHERE id=?", (i,), one=True)
                    for i in ids]
        rows = [r for r in rows if r]
        if len(rows) < 2:
            return "有效经验不足两条"
        primary = rows[0]
        tags = set(json.loads(primary.get("tags") or "[]"))
        files = set(json.loads(primary.get("files") or "[]"))
        tools = set(json.loads(primary.get("tools") or "[]"))
        lessons = [primary.get("lesson") or ""]
        success = sum(int(r.get("success") or 0) for r in rows)
        fail = sum(int(r.get("fail") or 0) for r in rows)
        evidence = sum(int(r.get("evidence") or 0) for r in rows)
        for r in rows[1:]:
            tags |= set(json.loads(r.get("tags") or "[]"))
            files |= set(json.loads(r.get("files") or "[]"))
            tools |= set(json.loads(r.get("tools") or "[]"))
            lessons.append(r.get("lesson") or "")
        lesson = "\n".join(x for x in dict.fromkeys(lessons) if x)[:2000]
        search = _search_text(primary.get("trigger"), primary.get("action"),
                              lesson, primary.get("strategy"), " ".join(tags),
                              " ".join(files))
        with self._lock:
            self._x("""UPDATE experiences SET evidence=?,success=?,fail=?,
                confidence=?,tags=?,files=?,tools=?,lesson=?,search_text=?,updated=?
                WHERE id=?""",
                (evidence, success, fail, _confidence(success, fail),
                 json.dumps(sorted(tags)), json.dumps(sorted(files)),
                 json.dumps(sorted(tools)), lesson, search, time.time(),
                 primary["id"]))
            for r in rows[1:]:
                self._x("DELETE FROM experiences WHERE id=?", (r["id"],))
                self._x("UPDATE episode_links SET exp_id=? WHERE exp_id=?",
                        (primary["id"], r["id"]))
                self._x("DELETE FROM contradictions WHERE a_id=? OR b_id=?",
                        (r["id"], r["id"]))
            self._commit()
        for r in rows[1:]:
            self._fts_delete(r["id"])
        self._fts_upsert(primary["id"], search)
        return f"✅ 已合并 {len(rows)} 条到 {primary['id']}"

    def mark_suspect(self, exp_id: str, reason: str = "") -> str:
        with self._lock:
            r = self._q("SELECT id,status FROM experiences WHERE id=?",
                        (exp_id,), one=True)
            if not r:
                return f"未找到 {exp_id}"
            self._x("UPDATE experiences SET status='suspect', last_fail_reason=?, "
                    "updated=? WHERE id=?",
                    (str(reason or "人工判定不可靠")[:1000], time.time(), exp_id))
            self._commit()
        return f"✅ {exp_id} 已标记 suspect（召回降权，不再自动直行）"

    def correct(self, exp_id: str, trigger: str = "", action: str = "",
                lesson: str = "", scope: str = "", exceptions: str = "",
                tags=None, reason: str = "", strategy: str = "") -> str:
        """生成修正版并废弃旧经验：旧经验 status=superseded，新经验立即可召回。"""
        with self._lock:
            old = self._q("SELECT * FROM experiences WHERE id=?", (exp_id,), one=True)
        if not old:
            return f"未找到经验 {exp_id}"
        if (old.get("status") or "active") == "superseded":
            return f"{exp_id} 已被修正过（superseded_by={old.get('superseded_by')}）"
        new_trigger = (str(trigger).strip() or old.get("trigger") or "")[:2000]
        new_action = (str(action).strip() or old.get("action") or "")[:2000]
        new_lesson = (str(lesson).strip() or old.get("lesson") or "")[:2000]
        new_scope = (str(scope).strip() or old.get("scope") or "")[:1000]
        new_exc = (str(exceptions).strip() or old.get("exceptions") or "")[:2000]
        if reason:
            new_exc = (new_exc + "\n纠正原因: " + str(reason)[:500]).strip()
        old_tags = []
        try:
            old_tags = json.loads(old.get("tags") or "[]")
        except Exception:
            pass
        if tags:
            if isinstance(tags, str):
                new_tags = [x.strip() for x in tags.replace("，", ",").split(",")
                            if x.strip()]
            else:
                new_tags = [str(x) for x in tags]
            new_tags = list(dict.fromkeys(new_tags))[:20]
        else:
            new_tags = old_tags
        try:
            files = json.loads(old.get("files") or "[]")
            tools = json.loads(old.get("tools") or "[]")
        except Exception:
            files, tools = [], []
        strategy = (str(strategy).strip() or old.get("strategy") or "")[:2500]
        changed = (new_trigger != (old.get("trigger") or "")
                   or new_action != (old.get("action") or "")
                   or new_lesson != (old.get("lesson") or "")
                   or new_scope != (old.get("scope") or "")
                   or new_exc != (old.get("exceptions") or "")
                   or strategy != (old.get("strategy") or "")
                   or list(new_tags) != list(old_tags))
        if not changed and not reason:
            return "没有可修正的内容（可传 trigger/action/lesson/scope/exceptions/tags/reason）"
        fp = _fingerprint(new_trigger, new_action, tools, "corrected", files)
        now = time.time()
        search = _search_text(new_trigger, new_action, new_lesson, strategy,
                              new_scope, new_exc, " ".join(new_tags),
                              " ".join(files), " ".join(tools))
        with self._lock:
            existing = self._q("SELECT * FROM experiences WHERE fingerprint=?",
                               (fp,), one=True)
            if existing and existing["id"] != exp_id:
                new_id = existing["id"]
                self._x("""UPDATE experiences SET trigger=?,action=?,lesson=?,
                    scope=?,exceptions=?,tags=?,search_text=?,status='active',
                    corrected_from=?,revision=?,updated=?,
                    vector=NULL,embed_model='',embed_dim=0 WHERE id=?""",
                    (new_trigger, new_action, new_lesson, new_scope, new_exc,
                     json.dumps(new_tags), search, exp_id,
                     int(old.get("revision") or 0) + 1, now, new_id))
            else:
                new_id = "exp_" + hashlib.sha256(
                    (fp + str(now)).encode("utf-8")).hexdigest()[:16]
                self._x("""INSERT INTO experiences(
                    id,fingerprint,trigger,action,lesson,scope,exceptions,tags,files,
                    tools,language,evidence,success,fail,confidence,utility,status,
                    created,updated,search_text,embed_model,embed_dim,
                    human_approved,kind,consolidated,strategy,corrected_from,revision)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (new_id, fp, new_trigger, new_action, new_lesson, new_scope,
                     new_exc, json.dumps(new_tags), json.dumps(files),
                     json.dumps(tools), _language(files), 1, 0, 0, 0.5, 0.5,
                     "active", now, now, search, "", 0, 0, "experience", 1,
                     strategy, exp_id, int(old.get("revision") or 0) + 1))
            self._x("UPDATE experiences SET status='superseded', superseded_by=?, "
                    "consolidated=1, updated=? WHERE id=?",
                    (new_id, now, exp_id))
            self._commit()
        self._fts_upsert(new_id, search)
        self._embed_experience(new_id, search)
        self._detect_contradiction(new_id)
        self._assign_cluster(new_id)
        return (f"✅ 已修正 {exp_id} → {new_id}；旧经验标记 superseded。"
                f"原因: {reason[:200] or '人工纠正'}")

    def record_gap(self, user: str, reason: str, missing_tools=None,
                   suggested_goal: str = "", th: str = "") -> str:
        th = th or task_hash(user)
        missing = [str(x) for x in (missing_tools or [])][:10]
        with self._lock:
            old = self._q("SELECT id FROM capability_gaps WHERE task_hash=? "
                          "AND status='open'", (th,), one=True)
            if old:
                self._x("UPDATE capability_gaps SET reason=?,missing_tools=?,"
                        "suggested_goal=?,updated=? WHERE id=?",
                        (str(reason)[:1500], json.dumps(missing),
                         str(suggested_goal)[:1000], time.time(), old["id"]))
                self._commit()
                return old["id"]
            gid = "gap_" + hashlib.sha256(
                (th + str(time.time())).encode("utf-8")).hexdigest()[:14]
            self._x("""INSERT INTO capability_gaps(
                id,ts,task_hash,user,reason,missing_tools,suggested_goal,status,
                updated) VALUES(?,?,?,?,?,?,?,?,?)""",
                (gid, time.time(), th, str(user)[:1000], str(reason)[:1500],
                 json.dumps(missing), str(suggested_goal)[:1000], "open",
                 time.time()))
            self._commit()
        return gid

    def pending_gaps(self, limit: int = 10) -> list:
        with self._lock:
            return self._q("SELECT * FROM capability_gaps WHERE status='open' "
                           "ORDER BY ts DESC LIMIT ?", (int(limit),))

    def resolve_gap(self, gap_id: str, resolved_by: str = "") -> str:
        with self._lock:
            self._x("UPDATE capability_gaps SET status='resolved', resolved_by=?, "
                    "updated=? WHERE id=?", (resolved_by, time.time(), gap_id))
            self._commit()
        return f"✅ 能力缺口 {gap_id} 已标记解决"

    def format_gaps(self, rows: list = None) -> str:
        rows = self.pending_gaps() if rows is None else rows
        if not rows:
            return "暂无未解决能力缺口。"
        lines = [f"未解决能力缺口 {len(rows)} 个："]
        for r in rows:
            try:
                tools = json.loads(r.get("missing_tools") or "[]")
            except Exception:
                tools = []
            lines.append(f"- {r.get('id')} | 缺工具: {', '.join(tools) or '-'} | "
                         f"建议目标: {str(r.get('suggested_goal') or '')[:160]}")
        return "\n".join(lines)

    def export(self, path: str = "") -> str:
        path = path or os.path.join(self.root, "export_experiences.jsonl")
        with self._lock:
            exps = self._q("SELECT * FROM experiences")
            clusters = self._q("SELECT * FROM clusters")
            plays = self._q("SELECT * FROM playbooks")
            prins = self._q("SELECT * FROM principles")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for kind, rows in (("experience", exps), ("pattern", clusters),
                                   ("playbook", plays), ("principle", prins)):
                    for r in rows:
                        r = dict(r)
                        for key in ("tags", "files", "tools", "steps"):
                            if isinstance(r.get(key), str):
                                try:
                                    r[key] = json.loads(r[key])
                                except Exception:
                                    pass
                        r = _jsonable(r)
                        r["_kind"] = kind
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return f"✅ 已导出到 {path}"
        except Exception as e:
            return f"❌ 导出失败: {e}"

    # ---------- 巩固与抽象 ----------
    def _llm_json(self, prompt: str, timeout: int = None) -> dict:
        from safety import reviewer_client, MODEL
        timeout = timeout or int(self.cfg.get("compress_timeout", 20) or 20)
        try:
            try:
                resp = reviewer_client.chat.completions.create(
                    model=os.getenv("REVIEWER_MODEL") or MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    response_format={"type": "json_object"})
            except Exception:
                resp = reviewer_client.chat.completions.create(
                    model=os.getenv("REVIEWER_MODEL") or MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout)
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

    def _episode_for(self, exp: dict) -> dict:
        eid = exp.get("source_episode")
        with self._lock:
            if eid:
                r = self._q("SELECT * FROM episodes WHERE id=?", (eid,), one=True)
                if r:
                    return r
            r = self._q("SELECT e.* FROM episodes e JOIN episode_links l "
                        "ON e.id=l.episode_id WHERE l.exp_id=? "
                        "ORDER BY e.ts DESC LIMIT 1", (exp["id"],), one=True)
            return r or {}

    def consolidate(self, limit: int = None, use_llm: bool = True) -> dict:
        limit = int(limit or self.cfg.get("consolidate_limit", 5) or 5)
        with self._lock:
            rows = self._q("SELECT * FROM experiences WHERE consolidated=0 "
                           "ORDER BY updated LIMIT ?", (limit,))
        done = 0
        for exp in rows:
            ep = self._episode_for(exp)
            if not ep:
                with self._lock:
                    self._x("UPDATE experiences SET consolidated=1 WHERE id=?",
                            (exp["id"],))
                    self._commit()
                continue
            tools = []
            try:
                tools = json.loads(ep.get("tools") or "[]")
            except Exception:
                pass
            data = {}
            if use_llm:
                prompt = f"""把下面任务压缩成可复用经验，只输出 JSON：
{{"trigger":"触发条件/用户意图（≤120字）","action":"有效做法（≤200字）",
"lesson":"可复用教训（≤200字，可空）",
"strategy":"可复用思考/执行策略（≤250字，可空；不要写隐藏推理，只写可执行方法）",
"scope":"适用范围（≤120字）","exceptions":"反例/边界（≤150字，可空）",
"tags":["最多6个标签"]}}

用户需求：{ep.get('user','')[:1500]}
最终回答：{ep.get('answer','')[:1800]}
已记录策略：{ep.get('strategy','')[:1000]}
工具序列：{', '.join(str(t) for t in tools[:20])}
结果：{ep.get('outcome','')}
"""
                data = self._llm_json(prompt)
            trigger = str(data.get("trigger") or exp.get("trigger") or "")[:500]
            action = str(data.get("action") or exp.get("action") or "")[:800]
            lesson = str(data.get("lesson") or exp.get("lesson") or "")[:1500]
            strategy = str(data.get("strategy") or ep.get("strategy")
                           or exp.get("strategy") or "")[:2500]
            scope = str(data.get("scope") or exp.get("scope") or "")[:500]
            exceptions = str(data.get("exceptions") or "")[:800]
            tags = data.get("tags") or json.loads(exp.get("tags") or "[]")
            tags = [str(x) for x in tags][:12]
            search = _search_text(trigger, action, lesson, strategy, scope,
                                  exceptions, " ".join(tags),
                                  exp.get("tools") or "")
            with self._lock:
                self._x("""UPDATE experiences SET trigger=?,action=?,lesson=?,
                    strategy=?,scope=?,exceptions=?,tags=?,search_text=?,
                    consolidated=1,updated=? WHERE id=?""",
                    (trigger, action, lesson, strategy, scope, exceptions,
                     json.dumps(tags), search, time.time(), exp["id"]))
                self._commit()
            self._fts_upsert(exp["id"], search)
            self._embed_experience(exp["id"], search)
            self._detect_contradiction(exp["id"])
            self._assign_cluster(exp["id"])
            done += 1
        self._abstract_clusters()
        self._meta_set("last_consolidation", time.time())
        return {"consolidated": done, "remaining": self.stats()["unconsolidated"]}

    def _embed_experience(self, exp_id: str, text: str):
        vecs, model, dim = embed_texts([text])
        with self._lock:
            self._x("UPDATE experiences SET vector=?,embed_model=?,embed_dim=? "
                    "WHERE id=?", (_vec_bytes(vecs[0]), model, dim, exp_id))
            self._commit()

    def _detect_contradiction(self, exp_id: str):
        """同一做法结果相反的高相似经验 → 记录矛盾/边界，供人工与召回参考。"""
        with self._lock:
            exp = self._q("SELECT * FROM experiences WHERE id=?", (exp_id,), one=True)
            others = self._q("SELECT * FROM experiences WHERE id!=? AND status='active' "
                             "AND vector IS NOT NULL", (exp_id,))
        if not exp:
            return
        v = _vec_load(exp.get("vector"))
        if not v:
            return
        threshold = float(self.cfg.get("similarity_threshold", 0.90) or 0.90)
        e_ok = int(exp.get("success") or 0)
        e_bad = int(exp.get("fail") or 0)
        for r in others:
            if r.get("embed_model") != exp.get("embed_model"):
                continue
            ov = _vec_load(r.get("vector"))
            if not ov or _cos(v, ov) < threshold:
                continue
            r_ok, r_bad = int(r.get("success") or 0), int(r.get("fail") or 0)
            conflict = ((e_ok > 0 and r_bad > 0 and e_bad == 0)
                        or (e_bad > 0 and r_ok > 0 and e_ok == 0))
            if not conflict:
                continue
            a, b = sorted([exp_id, r["id"]])
            reason = (f"高度相似但结果不同：{exp_id} success/fail="
                      f"{e_ok}/{e_bad}，{r['id']} success/fail={r_ok}/{r_bad}")
            with self._lock:
                self._x("INSERT OR IGNORE INTO contradictions(a_id,b_id,reason,"
                        "status,created) VALUES(?,?,?,?,?)",
                        (a, b, reason, "open", time.time()))
                self._commit()

    def _cluster_centroid(self, cid: str):
        with self._lock:
            rows = self._q("SELECT vector FROM experiences WHERE cluster_id=? "
                           "AND vector IS NOT NULL", (cid,))
        vs = [_vec_load(r["vector"]) for r in rows]
        vs = [v for v in vs if v]
        if not vs:
            return []
        dim = len(vs[0])
        cen = [0.0] * dim
        for v in vs:
            for i, x in enumerate(v):
                cen[i] += x
        n = math.sqrt(sum(x * x for x in cen))
        return [x / n for x in cen] if n else cen

    def _assign_cluster(self, exp_id: str):
        with self._lock:
            exp = self._q("SELECT * FROM experiences WHERE id=?", (exp_id,), one=True)
            clusters = self._q("SELECT * FROM clusters WHERE status='active'")
        v = _vec_load(exp.get("vector"))
        if not v:
            return
        same = [c for c in clusters if c.get("embed_model") == exp.get("embed_model")]
        best, best_sim = None, -1.0
        for c in same:
            cv = _vec_load(c.get("centroid"))
            s = _cos(v, cv)
            if s > best_sim:
                best, best_sim = c, s
        threshold = float(self.cfg.get("cluster_threshold", 0.82) or 0.82)
        if best and best_sim >= threshold:
            cid = best["id"]
            success = int(best.get("success") or 0) + int(exp.get("success") or 0)
            fail = int(best.get("fail") or 0) + int(exp.get("fail") or 0)
            with self._lock:
                self._x("UPDATE experiences SET cluster_id=? WHERE id=?", (cid, exp_id))
                self._x("UPDATE clusters SET support=support+1, success=?, fail=?, "
                        "confidence=?, updated=? WHERE id=?",
                        (success, fail, _confidence(success, fail), time.time(), cid))
                self._commit()
            cen = self._cluster_centroid(cid)
            with self._lock:
                self._x("UPDATE clusters SET centroid=?, dim=? WHERE id=?",
                        (_vec_bytes(cen), len(cen), cid))
                self._commit()
        else:
            cid = "cl_" + hashlib.sha256(
                (str(exp_id) + str(time.time())).encode()).hexdigest()[:14]
            tags = json.loads(exp.get("tags") or "[]")
            search = _search_text(exp.get("trigger"), exp.get("action"),
                                  exp.get("lesson"), " ".join(tags))
            with self._lock:
                self._x("""INSERT INTO clusters(id,centroid,dim,embed_model,
                    trigger,action,scope,lesson,exceptions,tags,support,success,
                    fail,confidence,status,created,updated,search_text)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (cid, _vec_bytes(v), len(v), exp.get("embed_model"),
                     exp.get("trigger"), "", exp.get("scope"), "",
                     "", json.dumps(tags), 1, int(exp.get("success") or 0),
                     int(exp.get("fail") or 0),
                     _confidence(int(exp.get("success") or 0),
                                 int(exp.get("fail") or 0)),
                     "active", time.time(), time.time(), search))
                self._x("UPDATE experiences SET cluster_id=? WHERE id=?", (cid, exp_id))
                self._commit()

    def _cluster_members(self, cid: str, limit=10):
        with self._lock:
            return self._q("SELECT * FROM experiences WHERE cluster_id=? "
                           "ORDER BY success DESC, updated DESC LIMIT ?",
                           (cid, int(limit)))

    def _abstract_clusters(self):
        cfg = self.cfg
        min_pattern = int(cfg.get("min_evidence_for_pattern", 3) or 3)
        min_principle = int(cfg.get("min_evidence_for_principle", 5) or 5)
        with self._lock:
            clusters = self._q("SELECT * FROM clusters WHERE status='active' "
                               "AND support>=?", (min_pattern,))
        for c in clusters:
            members = self._cluster_members(c["id"], 10)
            if not members:
                continue
            if not c.get("lesson"):
                brief = []
                for m in members:
                    brief.append(f"- 触发:{_clip(m.get('trigger'),120)} | "
                                 f"做法:{_clip(m.get('action'),160)} | "
                                 f"经验:{_clip(m.get('lesson'),160)} | "
                                 f"结果:{m.get('success')}/{m.get('fail')}")
                prompt = f"""把以下相似经验抽象成一个模式，只输出 JSON：
{{"trigger":"触发条件","action":"推荐做法","lesson":"核心模式/教训",
"scope":"适用范围","exceptions":"反例/边界（可空）","tags":["最多6个"]}}

相似经验：
{chr(10).join(brief[:8])}
"""
                data = self._llm_json(prompt)
                if not data:
                    continue
                trigger = str(data.get("trigger") or members[0].get("trigger") or "")[:500]
                action = str(data.get("action") or members[0].get("action") or "")[:800]
                lesson = str(data.get("lesson") or "")[:1500]
                scope = str(data.get("scope") or "")[:500]
                exc = str(data.get("exceptions") or "")[:800]
                tags = [str(x) for x in (data.get("tags") or [])][:12]
                search = _search_text(trigger, action, lesson, scope, exc,
                                      " ".join(tags))
                with self._lock:
                    self._x("""UPDATE clusters SET trigger=?,action=?,lesson=?,
                        scope=?,exceptions=?,tags=?,search_text=?,updated=?
                        WHERE id=?""",
                        (trigger, action, lesson, scope, exc,
                         json.dumps(tags), search, time.time(), c["id"]))
                    self._commit()
                c.update({"trigger": trigger, "action": action,
                          "lesson": lesson, "scope": scope})
            if not self._has_playbook(c["id"]) and c.get("lesson"):
                self._make_playbook(c, members)
            sr = (int(c.get("success") or 0)
                  / max(1, int(c.get("success") or 0) + int(c.get("fail") or 0)))
            if ((c.get("support") or 0) >= min_principle and sr >= 0.7
                    and not self._has_principle(c["id"])):
                self._make_principle(c)

    def _has_playbook(self, cid):
        with self._lock:
            return bool(self._q("SELECT 1 FROM playbooks WHERE cluster_id=?",
                                (cid,), one=True))

    def _has_principle(self, cid):
        with self._lock:
            return bool(self._q("SELECT 1 FROM principles WHERE cluster_id=?",
                                (cid,), one=True))

    def _make_playbook(self, c, members):
        seqs = []
        for m in members[:8]:
            try:
                tools = json.loads(m.get("tools") or "[]")
            except Exception:
                tools = []
            if tools:
                seqs.append(" → ".join(tools))
        prompt = f"""根据下面模式与工具序列，生成可复用步骤，只输出 JSON：
{{"name":"playbook 名称","trigger":"何时使用","steps":[{{"goal":"步骤目标","tool":"工具名","args_hint":"参数提示"}}],"scope":"适用范围"}}

模式触发：{c.get('trigger','')}
模式做法：{c.get('action','')}
工具序列样例：{chr(10).join(seqs[:6])}
"""
        data = self._llm_json(prompt)
        if not data or not isinstance(data.get("steps"), list) or not data["steps"]:
            return
        pid = "pb_" + hashlib.sha256(
            (c["id"] + json.dumps(data, ensure_ascii=False)).encode()).hexdigest()[:14]
        text = _search_text(data.get("name"), data.get("trigger"), data.get("scope"))
        vecs, model, dim = embed_texts([text])
        with self._lock:
            self._x("""INSERT OR REPLACE INTO playbooks(
                id,name,trigger,steps,tools,scope,evidence,success,fail,
                confidence,status,human_approved,cluster_id,created,updated,
                search_text,vector,embed_model,embed_dim)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pid, str(data.get("name") or "")[:200],
                 str(data.get("trigger") or "")[:500],
                 json.dumps(data.get("steps")[:10], ensure_ascii=False),
                 json.dumps(seqs[:10]), str(data.get("scope") or "")[:500],
                 int(c.get("support") or 0), int(c.get("success") or 0),
                 int(c.get("fail") or 0), _confidence(int(c.get("success") or 0),
                                                      int(c.get("fail") or 0)),
                 "candidate", 0, c["id"], time.time(), time.time(), text,
                 _vec_bytes(vecs[0]), model, dim))
            self._commit()

    def _make_principle(self, c):
        members = self._cluster_members(c["id"], 8)
        if not members:
            return
        brief = "\n".join(
            f"- {_clip(m.get('trigger'),100)} → {_clip(m.get('lesson') or m.get('action'),160)}"
            for m in members)
        prompt = f"""把以下模式抽象成一条原则（简洁、可执行、带边界），只输出 JSON：
{{"text":"原则文本（≤160字）","scope":"适用范围（≤120字）"}}

{chr(10).join(brief)}
"""
        data = self._llm_json(prompt)
        if not data or not data.get("text"):
            return
        pid = "pr_" + hashlib.sha256(
            (c["id"] + str(data["text"])).encode()).hexdigest()[:14]
        text = _search_text(data.get("text"), data.get("scope"))
        vecs, model, dim = embed_texts([text])
        with self._lock:
            self._x("""INSERT OR REPLACE INTO principles(
                id,text,scope,evidence,success,fail,confidence,status,
                human_approved,cluster_id,created,updated,search_text,vector,
                embed_model,embed_dim)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pid, str(data.get("text"))[:1000],
                 str(data.get("scope") or "")[:500], int(c.get("support") or 0),
                 int(c.get("success") or 0), int(c.get("fail") or 0),
                 _confidence(int(c.get("success") or 0), int(c.get("fail") or 0)),
                 "candidate", 0, c["id"], time.time(), time.time(), text,
                 _vec_bytes(vecs[0]), model, dim))
            self._commit()

    def _schedule_consolidate(self):
        cfg = self.cfg
        if os.environ.get("AI_AGENT_NO_BG") == "1" or not cfg.get("consolidate_on_save", True):
            return
        try:
            n = self.stats().get("unconsolidated", 0)
            if n < int(cfg.get("consolidate_after", 10) or 10):
                return
            with _BG["lock"]:
                if _BG["thread"] is not None and _BG["thread"].is_alive():
                    return
                t = threading.Thread(target=self._bg_consolidate, daemon=True)
                _BG["thread"] = t
                t.start()
        except Exception:
            pass

    def _bg_consolidate(self):
        try:
            self.consolidate(limit=int(self.cfg.get("consolidate_after", 10) or 10),
                             use_llm=True)
        except Exception as e:
            print(f"[experience] 后台巩固失败: {e}")

    # ---------- 审批 / 重建 ----------
    def approve(self, target_id: str, approve: bool = True) -> str:
        with self._lock:
            for table in ("playbooks", "principles"):
                r = self._q(f"SELECT id FROM {table} WHERE id=?", (target_id,), one=True)
                if r:
                    status = "approved" if approve else "rejected"
                    self._x(f"UPDATE {table} SET status=?, human_approved=? WHERE id=?",
                            (status, 1 if approve else 0, target_id))
                    self._commit()
                    return f"✅ {table} {target_id} → {status}"
        return f"未找到 {target_id}"

    def rebuild(self, force_vectors: bool = False) -> str:
        with self._lock:
            rows = self._q("SELECT * FROM experiences")
        vec_done = 0
        for r in rows:
            search = _search_text(r.get("trigger"), r.get("action"),
                                  r.get("lesson"), r.get("strategy"),
                                  r.get("scope"), r.get("exceptions"),
                                  r.get("tags"), r.get("files"), r.get("tools"))
            with self._lock:
                self._x("UPDATE experiences SET search_text=? WHERE id=?",
                        (search, r["id"]))
                self._commit()
            self._fts_upsert(r["id"], search)
            if force_vectors:
                self._embed_experience(r["id"], search)
                vec_done += 1
        if force_vectors:
            with self._lock:
                self._x("UPDATE experiences SET cluster_id=NULL")
                self._x("DELETE FROM clusters")
                self._commit()
            for r in rows:
                self._assign_cluster(r["id"])
        return f"✅ 重建完成：{len(rows)} 条经验，向量重建 {vec_done}"


# ================= 单例 =================

_STORE = None
_STORE_LOCK = threading.Lock()


def get_store() -> ExperienceStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ExperienceStore()
        else:
            _STORE.refresh_cfg()
        return _STORE


# ================= 兼容层 =================

def auto_save(user: str, answer: str, tool_calls: list,
              steps: int = 0, outcome: str = "success",
              strategy: str = "") -> str:
    cfg = _cfg()
    if not cfg.get("auto_save", True):
        return "skip:auto_save_disabled"
    if not tool_calls and len((user or "").strip()) < 30 and not strategy:
        return "skip:too_short"
    try:
        res = get_store().record_episode(user, answer, tool_calls,
                                         steps=steps, outcome=outcome,
                                         strategy=strategy)
        if not res.get("id"):
            return "skip:" + str(res.get("skipped") or "not_saved")
        return ("merged:" if res.get("merged") else "saved:") + res["id"]
    except Exception as e:
        return f"skip:error:{e}"


def recall(query: str, k: int = None, min_score: float = None) -> list:
    try:
        return get_store().recall(query, k=k, min_score=min_score)
    except Exception:
        return []


def route(query: str, k: int = None) -> dict:
    try:
        return get_store().route(query, k=k)
    except Exception as e:
        return {"mode": "none", "cards": [], "text": "",
                "playbook": None, "reason": f"error:{e}"}


def format_cards(cards, token_budget: int = None) -> str:
    try:
        return get_store().format_cards(cards, token_budget=token_budget)
    except Exception:
        return ""


def record_recall_outcome(th: str, outcome: str):
    try:
        get_store().record_recall_outcome(th, outcome)
    except Exception:
        pass
