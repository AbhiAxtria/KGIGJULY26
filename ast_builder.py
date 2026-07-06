"""
Build the canonical graph from the real AST, across a multi-file corpus.

- Handles nested procedures (Procedure CONTAINS Procedure).
- Resolves calls across all files in the corpus; unresolved targets become
  :External (e.g. a routine defined in a file you didn't include).
- Classifies each routine's *logic pattern* (the migration-relevant shape of the
  logic) from the AST: SET_BASED_TRANSFORM, ROW_BY_ROW_CURSOR, ORCHESTRATION,
  OLTP_BOOKKEEPING, or PROCEDURAL.
"""
from __future__ import annotations

import re
import hashlib as _hashlib
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

from . import plsql_ast as ast
from .plsql_ast import (Routine, Package, Decl, Block, LoopStmt, IfStmt, CaseStmt,
                        SqlStmt, ExecImmediate, CallStmt, AssignStmt, SimpleStmt,
                        DDLObject)
from .sql_analyzer import analyze_statement
from .graph_model import Graph, nid

_VIEW_HINT = re.compile(r"(?i)(^v_|^vw_|_v$|_vw$|_view$)")
_BUILTIN_CALLS = {"dbms_output", "dbms_stats", "raise_application_error"}


def parse_corpus(paths: List[str]) -> List[Tuple[str, list]]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            out.append((p, ast.parse_program(f.read())))
    return out


def _is_view(name: str, catalog: Optional[dict]) -> bool:
    if catalog and name.lower() in {v.lower() for v in catalog.get("views", [])}:
        return True
    return bool(_VIEW_HINT.search(name))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _res_id(statement_type: str, name: str) -> str:
    h = _hashlib.md5(f"{statement_type}:{name}".lower().encode()).hexdigest()[:8].upper()
    return f"RES_{statement_type.upper()}_{h}"


def _dml_operation(kind: str) -> str:
    k = (kind or "").upper()
    if k == "TRUNCATE":
        return "TRUNCATE"
    if k == "DROP":
        return "DROP"
    if k in ("INSERT", "UPDATE", "DELETE", "MERGE", "SELECT", "WITH"):
        return "DML"
    return "DML"


def _file_type(nodes) -> str:
    has_pkg = any(isinstance(n, Package) for n in nodes)
    has_routine = any(isinstance(n, Routine) for n in nodes)
    has_ddl = any(isinstance(n, DDLObject) for n in nodes)
    if has_pkg:
        return "package"
    if has_routine and has_ddl:
        return "mixed"
    if has_routine:
        return "plsql"
    if has_ddl:
        return "ddl"
    return "sql"


def _seq_props(text: str) -> dict:
    p = {}
    m = re.search(r"(?i)START\s+WITH\s+(\d+)", text)
    if m:
        p["start_with"] = int(m.group(1))
    m = re.search(r"(?i)INCREMENT\s+BY\s+(\d+)", text)
    if m:
        p["increment_by"] = int(m.group(1))
    m = re.search(r"(?i)\b(NOCACHE)\b|\bCACHE\s+(\d+)", text)
    if m:
        p["cache"] = "NOCACHE" if m.group(1) else m.group(2)
    return p


def _dblink_props(text: str) -> dict:
    p = {"is_public": "PUBLIC" in text.upper()}
    m = re.search(r"(?i)USING\s+'([^']+)'", text)
    if m:
        p["remote_db"] = m.group(1)
    m = re.search(r"(?i)CONNECT\s+TO\s+(\w+)", text)
    if m:
        p["connect_user"] = m.group(1)
    return p


class _Builder:
    def __init__(self, catalog, include_columns):
        self.g = Graph()
        self.catalog = catalog
        self.include_columns = include_columns
        self.registry: Dict[str, str] = {}          # name(lower) -> routine node id
        self.pending_calls: List[Tuple[str, List[str]]] = []
        self.current_file = ""
        self.schema_name = "UNKNOWN"
        self.provenance: Dict[str, str] = {}        # repo_name, git_branch, ...
        self.op_of: Dict[str, Tuple[str, str]] = {} # node id -> (operation_type, object_type)

    def _enrich(self, node_id: str, statement_type: str):
        """Attach the common config properties carried by every Resource node."""
        n = self.g.nodes.get(node_id)
        if n is None:
            return
        name = n.props.get("name", "")
        n.props.setdefault("res_id", _res_id(statement_type, name))
        n.props["type"] = statement_type
        n.props["schema_name"] = (self.schema_name or "UNKNOWN")
        n.props.setdefault("created_at", _now_iso())
        n.props.setdefault("enabled", True)
        if self.current_file:
            n.props["repo_file_path"] = self.current_file
            n.props["git_file_exists"] = True
            n.props["found_in_repo"] = True
        for k in ("repo_name", "git_repo_name", "git_branch_name"):
            if self.provenance.get(k):
                n.props[k] = self.provenance[k]

    def _db_operation(self, resource_id: str, operation_type: str,
                      statement_type: str, name: str):
        """Record the operation metadata used to annotate the CONTAINS edge.

        No separate DBOperation node is created \u2014 the CONTAINS relationship
        carries operationType and objectType directly.
        """
        self.op_of[resource_id] = (operation_type, statement_type)

    def _record_ddl(self, node_id: str, text: str, is_full_def: bool):
        """Append a DDL statement to the node's history.

        is_full_def=True for CREATE / CREATE OR REPLACE (sets the current `ddl`);
        False for ALTER (kept in history, current `ddl` unchanged).
        """
        if not text:
            return
        node = self.g.nodes[node_id]
        hist = node.props.get("ddl_history")
        if not isinstance(hist, list):
            hist = []
            if node.props.get("ddl"):
                hist.append(node.props["ddl"])
        if not hist or hist[-1] != text:
            hist.append(text)
        node.props["ddl_history"] = hist
        node.props["modification_count"] = max(0, len(hist) - 1)
        node.props["last_modified_ddl"] = hist[-1]
        node.props["ddl_hash"] = _hashlib.md5("||".join(hist).encode()).hexdigest()[:12]
        if is_full_def:
            node.props["ddl"] = text
        elif "ddl" not in node.props:
            node.props["ddl"] = text
        if self.current_file:
            node.props["source_file"] = self.current_file

    def relation(self, name: str):
        label = "View" if _is_view(name, self.catalog) else "Table"
        rid = nid(label.lower(), name.lower())     # identity is case-insensitive
        self.g.add_node(rid, label, name=name.lower())
        return rid, label

    # ---- routine ----
    def routine(self, r: Routine, parent_id: Optional[str], qual: str, schema_id):
        if r.is_forward:
            return None
        rid = nid("routine", qual.lower())
        label = "Function" if r.kind == "FUNCTION" else "Procedure"
        owner = qual.rsplit(".", 1)[0] if "." in qual else ""
        self.g.add_node(rid, label, name=r.name, owner=owner, qualified=qual,
                        return_type=r.return_type, param_count=len(r.params),
                        loc=(r.source.count(chr(10)) + 1) if r.source else 0)
        self._record_ddl(rid, r.source, is_full_def=True)
        self.registry.setdefault(r.name.lower(), rid)
        self.registry.setdefault(qual.lower(), rid)
        stype = "FUNCTION" if r.kind == "FUNCTION" else "PROCEDURE"
        self._enrich(rid, stype)
        op = "REPLACE" if "OR REPLACE" in (r.source or "").upper() else "CREATE"
        self._db_operation(rid, op, stype, r.name)
        if parent_id:
            self.g.add_edge(parent_id, "CONTAINS", rid)
        elif schema_id:
            self.g.add_edge(schema_id, "DB_OPERATION", rid)

        # declarations
        pragmas = []
        for p in r.params:
            sid = nid(rid, "param", p.name)
            self.g.add_node(sid, "Parameter", name=p.name, datatype=p.datatype, mode=p.mode)
            self.g.add_edge(rid, "DECLARES", sid)
        for d in r.decls:
            if isinstance(d, Routine):
                continue
            lbl = {"VARIABLE": "Variable", "CONSTANT": "Constant",
                   "CURSOR": "Cursor", "TYPE": "Type", "PRAGMA": None}[d.kind]
            if d.kind == "PRAGMA":
                pragmas.append(d.name.upper())
                continue
            sid = nid(rid, "decl", d.kind, d.name)
            self.g.add_node(sid, lbl, name=d.name, datatype=d.datatype)
            self.g.add_edge(rid, "DECLARES", sid)
            if d.kind == "CURSOR":
                self.g.add_edge(rid, "USES_CURSOR", sid)

        # body block
        block_id = nid(rid, "block")
        self.g.add_node(block_id, "Block", name=f"{r.name}_body")
        self.g.add_edge(rid, "HAS_BLOCK", block_id)

        feats: Dict[str, int] = {}
        calls: List[str] = []
        # cursors declared -> explicit_cursor feature
        n_cursors = sum(1 for d in r.decls if isinstance(d, Decl) and d.kind == "CURSOR")
        if n_cursors:
            feats["explicit_cursor"] = n_cursors
        if any(pr.startswith("AUTONOMOUS") for pr in pragmas):
            feats["autonomous_txn"] = 1
        if any(isinstance(d, Routine) for d in r.decls):
            feats["nested_procedure"] = sum(1 for d in r.decls if isinstance(d, Routine))

        self._walk(r.body, block_id, feats, calls, in_loop=False)
        if r.handlers:
            feats["exception_handler"] = len(r.handlers)
            eh_id = nid(rid, "exc")
            self.g.add_node(eh_id, "ExceptionHandler",
                            name="; ".join(h.when for h in r.handlers)[:60])
            self.g.add_edge(rid, "HANDLES", eh_id)
            for h in r.handlers:
                self._walk(h.body, block_id, feats, calls, in_loop=False)

        for feat, cnt in feats.items():
            fid = nid("feature", feat)
            self.g.add_node(fid, "Feature", name=feat)
            self.g.add_edge(rid, "HAS_FEATURE", fid, count=cnt)

        # logic pattern (the "understanding")
        self.g.nodes[rid].props["logic_pattern"] = self._classify(r, feats, calls)

        self.pending_calls.append((rid, calls))

        # nested routines
        for d in r.decls:
            if isinstance(d, Routine):
                self.routine(d, rid, f"{qual}.{d.name}", schema_id)
        return rid

    # ---- statement walk ----
    def _walk(self, stmts, block_id, feats, calls, in_loop):
        for s in stmts:
            if isinstance(s, SqlStmt):
                self._sql(s, block_id, feats)
            elif isinstance(s, ExecImmediate):
                feats["execute_immediate"] = feats.get("execute_immediate", 0) + 1
                if s.dynamic_target:
                    tid, _ = self.relation(s.dynamic_target)
                    st_id = nid(block_id, "dyn", s.dynamic_target)
                    self.g.add_node(st_id, "SqlStatement", kind="TRUNCATE",
                                    dynamic=True, text=s.text[:200])
                    self.g.add_edge(block_id, "EXECUTES", st_id)
                    self.g.add_edge(st_id, "WRITES", tid)
            elif isinstance(s, CallStmt):
                base = s.callee.split(".")[0].lower()
                if base in _BUILTIN_CALLS or s.callee.lower().startswith("dbms_"):
                    feats["dbms_pkg_call"] = feats.get("dbms_pkg_call", 0) + 1
                elif s.callee.lower().startswith("utl_"):
                    feats["utl_pkg_call"] = feats.get("utl_pkg_call", 0) + 1
                else:
                    calls.append(s.callee)
            elif isinstance(s, AssignStmt):
                if ".nextval" in s.text.lower():
                    feats["sequence_nextval"] = feats.get("sequence_nextval", 0) + 1
            elif isinstance(s, SimpleStmt):
                kw = s.keyword.lower()
                if kw in ("commit", "rollback", "savepoint"):
                    feats[kw] = feats.get(kw, 0) + 1
                if kw == "raise" and "raise_application_error" in s.text.lower():
                    feats["raise_app_error"] = feats.get("raise_app_error", 0) + 1
            elif isinstance(s, LoopStmt):
                lk = {"FOR": "cursor_for_loop" if s.cursor_ref or "select" in s.iter_text.lower()
                      else "for_loop", "WHILE": "while_loop", "BASIC": "basic_loop"}[s.loop_kind]
                feats[lk] = feats.get(lk, 0) + 1
                loop_id = nid(block_id, "loop", lk, str(len(self.g.edges)))
                self.g.add_node(loop_id, "Loop", name=lk, iter=s.iter_text[:80])
                self.g.add_edge(block_id, "CONTAINS", loop_id)
                self._walk(s.body, block_id, feats, calls, in_loop=True)
            elif isinstance(s, IfStmt):
                feats["if_branch"] = feats.get("if_branch", 0) + 1
                br_id = nid(block_id, "branch", str(len(self.g.edges)))
                self.g.add_node(br_id, "Branch", name="IF")
                self.g.add_edge(block_id, "CONTAINS", br_id)
                for _cond, body in s.branches:
                    self._walk(body, block_id, feats, calls, in_loop)
            elif isinstance(s, CaseStmt):
                feats["if_branch"] = feats.get("if_branch", 0) + 1
            elif isinstance(s, Block):
                self._walk(s.body, block_id, feats, calls, in_loop)
                for h in s.handlers:
                    feats["exception_handler"] = feats.get("exception_handler", 0) + 1
                    self._walk(h.body, block_id, feats, calls, in_loop)

    def _sql(self, s: SqlStmt, block_id, feats):
        a = analyze_statement(s.text)
        st_id = nid(block_id, "stmt", str(len(self.g.edges)))
        self.g.add_node(
            st_id, "SqlStatement", kind=a.kind, joins=a.joins, aggregates=a.aggregates,
            window_funcs=a.window_funcs, has_group_by=a.has_group_by,
            oracle_specific=",".join(a.oracle_specific), parse_ok=a.parse_ok,
            spark_sql=a.spark_sql or "", text=s.text[:500],
        )
        self.g.add_edge(block_id, "EXECUTES", st_id)
        # database link references: name@dblink
        for link in set(re.findall(r"@(\w+)", s.text)):
            lid = nid("dblink", link.lower())
            self.g.add_node(lid, "DatabaseLink", name=link.lower())
            self.g.add_edge(st_id, "VIA_DBLINK", lid)
        for t in a.reads:
            rid, lbl = self.relation(t)
            self.g.add_edge(st_id, "READS", rid)
            if lbl == "View":
                self.g.add_edge(st_id, "USES_VIEW", rid)
        for t in a.writes:
            tid, _ = self.relation(t)
            self.g.add_edge(st_id, "WRITES", tid)
        if self.include_columns:
            for col in a.columns:
                cid = nid(st_id, "col", col)
                self.g.add_node(cid, "Column", name=col)
                self.g.add_edge(st_id, "REFERENCES", cid)

    def _classify(self, r: Routine, feats: Dict[str, int], calls: List[str]) -> str:
        row_loop = any(k in feats for k in ("cursor_for_loop", "while_loop", "basic_loop"))
        # does a loop body contain DML or calls? (row-by-row)
        row_dml = row_loop and self._loop_has_work(r.body)
        set_based = self._has_set_based(r.body)
        own_dml = self._has_any_dml(r.body)
        if row_dml:
            return "ROW_BY_ROW_CURSOR"
        if set_based:
            return "SET_BASED_TRANSFORM"
        if len(calls) >= 2 and not own_dml:
            return "ORCHESTRATION"
        if own_dml and not row_loop and len(calls) == 0:
            return "OLTP_BOOKKEEPING"
        return "PROCEDURAL"

    def _loop_has_work(self, stmts) -> bool:
        for s in stmts:
            if isinstance(s, LoopStmt):
                if self._has_any_dml(s.body) or any(isinstance(x, CallStmt) for x in s.body):
                    return True
                if self._loop_has_work(s.body):
                    return True
            elif isinstance(s, IfStmt):
                for _c, b in s.branches:
                    if self._loop_has_work(b):
                        return True
        return False

    def _has_any_dml(self, stmts) -> bool:
        for s in stmts:
            if isinstance(s, (SqlStmt, ExecImmediate)):
                return True
            if isinstance(s, LoopStmt) and self._has_any_dml(s.body):
                return True
            if isinstance(s, IfStmt):
                for _c, b in s.branches:
                    if self._has_any_dml(b):
                        return True
            if isinstance(s, Block) and self._has_any_dml(s.body):
                return True
        return False

    def _has_set_based(self, stmts) -> bool:
        for s in stmts:
            if isinstance(s, SqlStmt):
                a = analyze_statement(s.text)
                if s.kind == "MERGE" or a.joins or a.has_group_by or \
                   (s.kind == "INSERT" and "select" in s.text.lower()):
                    return True
            if isinstance(s, Block) and self._has_set_based(s.body):
                return True
        return False

    def ddl(self, d: DDLObject, schema_id):
        if d.kind == "TABLE":
            tid, _ = self.relation(d.name)
            if schema_id:
                self.g.add_edge(schema_id, "DB_OPERATION", tid)
            self._record_ddl(tid, d.text, is_full_def=True)
            self.g.nodes[tid].props["column_count"] = len(d.column_defs)
            self._add_columns(tid, d.column_defs)
        elif d.kind == "ALTER_TABLE":
            tid, _ = self.relation(d.name)
            self._record_ddl(tid, d.text, is_full_def=False)
            self._add_columns(tid, d.column_defs)
            if d.column_defs:
                cur = self.g.nodes[tid].props.get("column_count", 0)
                self.g.nodes[tid].props["column_count"] = cur + len(d.column_defs)
        elif d.kind == "ALTER":
            self._alter(d, schema_id)
        elif d.kind == "SEQUENCE":
            sid = nid("sequence", d.name.lower())
            self.g.add_node(sid, "Sequence", name=d.name.lower(), **_seq_props(d.text))
            self._record_ddl(sid, d.text, is_full_def=True)
            if schema_id:
                self.g.add_edge(schema_id, "DB_OPERATION", sid)
        elif d.kind == "INDEX":
            iid = nid("index", d.name.lower())
            self.g.add_node(iid, "Index", name=d.name.lower(),
                            unique=("UNIQUE" in d.text.upper()))
            self._record_ddl(iid, d.text, is_full_def=True)
            if schema_id:
                self.g.add_edge(schema_id, "DB_OPERATION", iid)
            if d.target:
                tid, _ = self.relation(d.target)
                self.g.add_edge(iid, "INDEXES", tid)
                for c in d.columns:
                    cid = nid(tid, "col", c)
                    self.g.add_node(cid, "Column", name=c)
                    self.g.add_edge(cid, "BELONGS_TO", tid)
                    self.g.add_edge(iid, "ON_COLUMN", cid)
        elif d.kind == "SYNONYM":
            sid = nid("synonym", d.name.lower())
            self.g.add_node(sid, "Synonym", name=d.name.lower(),
                            target=d.target.lower(),
                            is_public=("PUBLIC" in d.text.upper()))
            self._record_ddl(sid, d.text, is_full_def=True)
            if schema_id:
                self.g.add_edge(schema_id, "DB_OPERATION", sid)
            if d.target:
                base = d.target.split("@")[0]
                if "@" in d.target:
                    tid = nid("external", base.lower())
                    self.g.add_node(tid, "External", name=base.lower())
                else:
                    tid, _ = self.relation(base)
                self.g.add_edge(sid, "REFERS_TO", tid)
                for link in re.findall(r"@(\w+)", d.target):
                    lid = nid("dblink", link.lower())
                    self.g.add_node(lid, "DatabaseLink", name=link.lower())
                    self.g.add_edge(sid, "VIA_DBLINK", lid)
        elif d.kind == "DATABASE_LINK":
            lid = nid("dblink", d.name.lower())
            self.g.add_node(lid, "DatabaseLink", name=d.name.lower(),
                            **_dblink_props(d.text))
            self._record_ddl(lid, d.text, is_full_def=True)
            if schema_id:
                self.g.add_edge(schema_id, "DB_OPERATION", lid)
        elif d.kind == "VIEW":
            vid = nid("view", d.name.lower())
            self.g.add_node(vid, "View", name=d.name.lower())
            self._record_ddl(vid, d.text, is_full_def=True)
            if schema_id:
                self.g.add_edge(schema_id, "DB_OPERATION", vid)
            if d.query:
                a = analyze_statement("SELECT * FROM (" + d.query.rstrip(";") + ")")
                for t in a.reads:
                    rid, _ = self.relation(t)
                    self.g.add_edge(vid, "DEPENDS_ON", rid)

        # primary object id (for SqlFile -> DEFINES_OBJECT linkage)
        replace = "OR REPLACE" in d.text.upper()
        mapping = {
            "TABLE": ("TABLE", "CREATE"),
            "ALTER_TABLE": ("TABLE", "ALTER"),
            "SEQUENCE": ("SEQUENCE", "REPLACE" if replace else "CREATE"),
            "INDEX": ("INDEX", "CREATE"),
            "SYNONYM": ("SYNONYM", "REPLACE" if replace else "CREATE"),
            "DATABASE_LINK": ("DATABASE_LINK", "CREATE"),
            "VIEW": ("VIEW", "REPLACE" if replace else "CREATE"),
        }
        oid = None
        if d.kind in ("TABLE", "ALTER_TABLE"):
            oid = self.relation(d.name)[0]
        elif d.kind == "SEQUENCE":
            oid = nid("sequence", d.name.lower())
        elif d.kind == "INDEX":
            oid = nid("index", d.name.lower())
        elif d.kind == "SYNONYM":
            oid = nid("synonym", d.name.lower())
        elif d.kind == "DATABASE_LINK":
            oid = nid("dblink", d.name.lower())
        elif d.kind == "VIEW":
            oid = nid("view", d.name.lower())
        elif d.kind == "ALTER":
            oid = self._alter_target_id(d)

        if oid is not None:
            if d.kind == "ALTER":
                stype = (d.obj_type or "OBJECT").upper()
                self._enrich(oid, stype)
                self._db_operation(oid, "ALTER", stype, d.name)
            else:
                stype, op = mapping[d.kind]
                self._enrich(oid, stype)
                self._db_operation(oid, op, stype, d.name)
        return oid

    def _alter_target_id(self, d: DDLObject):
        ot = (d.obj_type or "").upper()
        spec = {"SEQUENCE": ("sequence", None), "VIEW": ("view", None),
                "INDEX": ("index", None), "TRIGGER": ("trigger", None),
                "PACKAGE": ("pkg", None), "DATABASE_LINK": ("dblink", None)}
        if ot in ("PROCEDURE", "FUNCTION"):
            return self.registry.get(d.name.lower()) or nid("routine", d.name.lower())
        if ot in spec:
            key = d.name if ot == "PACKAGE" else d.name.lower()
            return nid(spec[ot][0], key)
        return None

    def top_statement(self, s, file_id):
        """A bare top-level SQL statement (DML/TRUNCATE/DROP).

        No SqlStatement node is created: the file CONTAINS the target object
        (the table it operates on), and the query text plus operationType /
        objectType are held on that CONTAINS relationship.
        """
        a = analyze_statement(s.text)
        op = _dml_operation(s.kind)
        obj_type = "TABLE"
        explicit = ""
        m = re.match(r"(?is)\s*(DROP|TRUNCATE)\s+(\w+)?\s+([\w\.]+)", s.text)
        if m and s.kind in ("DROP", "TRUNCATE"):
            obj_type = (m.group(2) or "TABLE").upper()
            explicit = (m.group(3) or "").lower()
        # target object(s): written table(s) for INSERT/UPDATE/DELETE/MERGE/
        # TRUNCATE/DROP, else read table(s) for a bare SELECT
        if explicit:
            targets = [explicit]
        else:
            targets = list(a.writes) or list(a.reads)
        query = s.text.strip()[:2000]
        for t in targets:
            rid, _ = self.relation(t)
            self.g.add_edge(file_id, "CONTAINS", rid,
                            operationType=op, objectType=obj_type, query=query)
        return None

    def _add_columns(self, tid, column_defs):
        for cd in column_defs:
            cid = nid(tid, "col", cd.name)
            self.g.add_node(cid, "Column", name=cd.name, datatype=cd.datatype,
                            nullable=cd.nullable, default=cd.default)
            self.g.add_edge(cid, "BELONGS_TO", tid)

    def _alter(self, d: DDLObject, schema_id):
        ot = (d.obj_type or "").upper()
        spec = {
            "SEQUENCE": ("sequence", "Sequence"),
            "VIEW": ("view", "View"),
            "INDEX": ("index", "Index"),
            "TRIGGER": ("trigger", "Trigger"),
            "PACKAGE": ("pkg", "Package"),
            "DATABASE_LINK": ("dblink", "DatabaseLink"),
        }
        if ot in ("PROCEDURE", "FUNCTION"):
            target = self.registry.get(d.name.lower())
            if target is None:
                label = "Function" if ot == "FUNCTION" else "Procedure"
                target = nid("routine", d.name.lower())
                self.g.add_node(target, label, name=d.name.lower())
                if schema_id:
                    self.g.add_edge(schema_id, "DB_OPERATION", target)
            self._record_ddl(target, d.text, is_full_def=False)
            return
        if ot in spec:
            prefix, label = spec[ot]
            key = d.name if ot == "PACKAGE" else d.name.lower()
            node_id = nid(prefix, key)
            if node_id not in self.g.nodes:
                self.g.add_node(node_id, label, name=key)
                if schema_id:
                    self.g.add_edge(schema_id, "DB_OPERATION", node_id)
            self._record_ddl(node_id, d.text, is_full_def=False)

    def finish(self):
        for caller_id, names in self.pending_calls:
            for name in names:
                target = self.registry.get(name.lower())
                if target is None:
                    # Use the routine-style id so that if this routine is defined
                    # in another file/run, its node MERGEs onto this placeholder
                    # (via the shared :Node id) instead of creating a duplicate.
                    target = nid("routine", name.lower())
                    self.g.add_node(target, "External", name=name.lower())
                self.g.add_edge(caller_id, "CALLS", target)
        return self.g


def build_graph_from_corpus(corpus, schema_name=None, catalog=None,
                            include_columns=True, provenance=None,
                            root_name=None) -> Graph:
    import os as _os
    b = _Builder(catalog, include_columns)
    b.schema_name = schema_name or "UNKNOWN"
    b.provenance = provenance or {}
    schema_id = None
    if schema_name:
        schema_id = nid("schema", schema_name)
        b.g.add_node(schema_id, "Schema", name=schema_name)

    # Root node: the input folder (multiple files) or the single input file.
    multi = len(corpus) > 1 or bool(root_name)
    root_id = None
    if multi:
        rname = root_name or _os.path.basename(
            _os.path.dirname(_os.path.commonprefix([p for p, _ in corpus])) or "input")
        root_id = nid("folder", rname or "input")
        b.g.add_node(root_id, "Folder", name=rname or "input")
        b._enrich(root_id, "FOLDER")
        b.g.nodes[root_id].props["type"] = "folder"

    for path, nodes in corpus:
        b.current_file = path
        file_id = nid("file", path)
        ftype = _file_type(nodes)
        b.g.add_node(file_id, "SqlFile", name=_os.path.basename(path), path=path)
        b._enrich(file_id, "SQLFILE")
        b.g.nodes[file_id].props["type"] = ftype          # package | ddl | plsql | mixed
        b.g.nodes[file_id].props["file_type"] = "sql"
        if root_id:
            b.g.add_edge(root_id, "CONTAINS", file_id)

        def _link(obj_id):
            if obj_id:
                op, otype = b.op_of.get(obj_id, ("", ""))
                b.g.add_edge(file_id, "CONTAINS", obj_id,
                             operationType=op, objectType=otype)

        for node in nodes:
            if isinstance(node, Package):
                pkg_id = nid("pkg", node.name)
                b.g.add_node(pkg_id, "Package", name=node.name, plsql_kind=node.kind)
                b._record_ddl(pkg_id, node.source, is_full_def=True)
                b._enrich(pkg_id, "PACKAGE")
                b._db_operation(pkg_id,
                                "REPLACE" if "OR REPLACE" in (node.source or "").upper() else "CREATE",
                                "PACKAGE", node.name)
                _link(pkg_id)
                if schema_id:
                    b.g.add_edge(schema_id, "DB_OPERATION", pkg_id)
                for m in node.members:
                    if isinstance(m, Routine):
                        mid = b.routine(m, pkg_id, f"{node.name}.{m.name}", schema_id)
                        _link(mid)
                    elif isinstance(m, Decl):
                        sid = nid(pkg_id, "decl", m.kind, m.name)
                        lbl = {"VARIABLE": "Variable", "CONSTANT": "Constant",
                               "CURSOR": "Cursor", "TYPE": "Type"}.get(m.kind)
                        if lbl:
                            b.g.add_node(sid, lbl, name=m.name, datatype=m.datatype)
                            b.g.add_edge(pkg_id, "DECLARES", sid)
            elif isinstance(node, Routine):
                rid = b.routine(node, None, node.name, schema_id)
                _link(rid)
            elif isinstance(node, ast.Trigger):
                tid = nid("trigger", node.name)
                b.g.add_node(tid, "Trigger", name=node.name)
                b._record_ddl(tid, node.source, is_full_def=True)
                b._enrich(tid, "TRIGGER")
                b._db_operation(tid,
                                "REPLACE" if "OR REPLACE" in (node.source or "").upper() else "CREATE",
                                "TRIGGER", node.name)
                _link(tid)
                if schema_id:
                    b.g.add_edge(schema_id, "DB_OPERATION", tid)
            elif isinstance(node, DDLObject):
                _link(b.ddl(node, schema_id))
            elif isinstance(node, SqlStmt):
                b.top_statement(node, file_id)

    return b.finish()
