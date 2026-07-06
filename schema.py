"""
Canonical graph model (ontology) for PL/SQL decomposition.

This is the single source of truth for *what kinds of nodes and relationships
may exist*. Every builder must produce only labels/relationships defined here,
and `validate_graph` enforces it. This makes the model uniform regardless of
the front end (regex today, ANTLR later) and regardless of whether a true AST
was used.

Decomposition hierarchy expressed by the model:

  SqlFile  (physical source file; type = package_body | ddl | standalone ...)
   └─DEFINES_OBJECT→ Resource (Package / Procedure / Function / Table / ...)

  Schema
   └─DEFINES→ Package / Procedure / Function / Trigger / Sequence / View / Table
       Package ─CONTAINS→ Procedure | Function
       Procedure/Function
           ─DECLARES→ Parameter | Variable | Constant | Cursor | Type
           ─CONTAINS→ Block (nested) | Procedure | Function   (nesting)
           ─HAS_BLOCK→ Block
           ─HANDLES→ ExceptionHandler
           ─CALLS→ Procedure | Function | External
           ─HAS_FEATURE→ Feature
           ─ASSESSED_AS→ MigrationAssessment        (knowledge layer)
       Block
           ─CONTAINS→ Loop | Branch | Block
           ─EXECUTES→ SqlStatement
       Loop ─USES_CURSOR→ Cursor
       SqlStatement (kind = SELECT|INSERT|UPDATE|DELETE|MERGE|TRUNCATE|DDL)
           ─READS→ Table | View
           ─WRITES→ Table
           ─USES_VIEW→ View
           ─REFERENCES→ Column
           ─USES_SEQUENCE→ Sequence
           ─USES_CONSTRUCT→ OracleConstruct           (knowledge layer)
       Column ─BELONGS_TO→ Table | View
       OracleConstruct ─MAPS_TO→ SparkConstruct        (knowledge base)

Resource super-label
---------------------
Procedure, Function, Table and Package are also tagged with the shared label
`Resource`, so they appear in the graph as e.g. (:Resource:Procedure). This lets
a single query span "all resources" while keeping the specific type. The primary
label (Procedure, Table, ...) is what `validate_graph` and the id-uniqueness
constraints key on; `Resource` is an additional label emitted at write time via
`node_labels()` / `cypher_labels()`.

Operations on objects
---------------------
Database operations are not separate nodes. The CONTAINS relationship from a
SqlFile to an object carries `operationType` (CREATE/ALTER/DROP/TRUNCATE/DML)
and `objectType` (TABLE/INDEX/PROCEDURE/...). For a bare DML statement the
CONTAINS edge also carries the `query` text, and no SqlStatement node is made.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

# label -> (description, key property other than id)
NODE_TYPES: Dict[str, Tuple[str, str]] = {
    "Folder":              ("Input folder / root of the parsed corpus", "name"),
    "SqlFile":             ("Physical source .sql file", "name"),
    "Schema":              ("Owning schema / namespace", "name"),
    "Package":             ("PL/SQL package spec or body", "name"),
    "Procedure":           ("Procedure (standalone or packaged/nested)", "name"),
    "Function":            ("Function", "name"),
    "Trigger":             ("Trigger", "name"),
    "Parameter":           ("Formal parameter of a routine", "name"),
    "Variable":            ("Local/global declared variable", "name"),
    "Constant":            ("Declared constant", "name"),
    "Cursor":              ("Explicit cursor declaration", "name"),
    "Type":                ("Record/collection/ref type", "name"),
    "Block":               ("BEGIN..END block (named or anonymous)", "name"),
    "Loop":                ("FOR/WHILE/BASIC/CURSOR_FOR loop", "name"),
    "Branch":              ("IF/CASE branch", "name"),
    "ExceptionHandler":    ("WHEN ... THEN exception handler", "name"),
    "SqlStatement":        ("Embedded SQL/DML/DDL statement", "kind"),
    "Table":               ("Physical table", "name"),
    "View":                ("View", "name"),
    "Column":              ("Column referenced by a statement", "name"),
    "Sequence":            ("Sequence object", "name"),
    "Synonym":             ("Synonym (alias for another object)", "name"),
    "Index":               ("Index on a table", "name"),
    "DatabaseLink":        ("Database link to a remote database", "name"),
    "External":            ("Call target not defined in the parsed corpus", "name"),
    "Feature":             ("Procedural signal (cursor loop, autonomous txn ...)", "name"),
    # Knowledge layer
    "MigrationAssessment": ("Per-routine convert/keep verdict + scores", "decision"),
    "OracleConstruct":     ("Oracle construct in the mapping knowledge base", "name"),
    "SparkConstruct":      ("PySpark/Spark equivalent", "name"),
}

# Super-labels applied in addition to the specific (primary) label.
# Every node also gets BASE_LABEL; file/folder nodes also get "File";
# object/resource nodes also get "Resource". So e.g.:
#   Procedure -> (:Procedure:Node:Resource)
#   SqlFile   -> (:SqlFile:File:Node:Resource)
#   Folder    -> (:Folder:File:Node:Resource)
BASE_LABEL = "Node"
SUPER_LABEL = "Resource"
RESOURCE_LABELS: Set[str] = {
    "Procedure", "Function", "Table", "View", "Package", "Sequence",
    "Index", "Synonym", "DatabaseLink", "Trigger", "SqlFile", "Folder",
}
FILE_LABELS: Set[str] = {"SqlFile", "Folder"}
_SYNTHETIC_LABELS = {BASE_LABEL, SUPER_LABEL, "File"}

# All labels that may appear on a node, including the synthetic super-labels.
ALL_LABELS: Set[str] = set(NODE_TYPES) | _SYNTHETIC_LABELS

# rel -> (allowed source labels, allowed dest labels, description)
REL_TYPES: Dict[str, Tuple[Set[str], Set[str], str]] = {
    "CONTAINS":      ({"Folder", "SqlFile", "Package", "Procedure", "Function", "Block"},
                      {"SqlFile", "Package", "Procedure", "Function", "Trigger",
                       "Sequence", "View", "Table", "Synonym", "Index",
                       "DatabaseLink", "Block", "Loop", "Branch"},
                      "containment: folder->file->object, and code nesting"),
    "DECLARES":      ({"Procedure", "Function", "Block", "Package"},
                      {"Parameter", "Variable", "Constant", "Cursor", "Type"},
                      "routine declares a symbol"),
    "HAS_BLOCK":     ({"Procedure", "Function", "Trigger"}, {"Block"}, "routine body block"),
    "HANDLES":       ({"Procedure", "Function", "Block"}, {"ExceptionHandler"}, "exception handling"),
    "CALLS":         ({"Procedure", "Function", "Block"},
                      {"Procedure", "Function", "External"}, "invokes routine"),
    "HAS_FEATURE":   ({"Procedure", "Function"}, {"Feature"}, "procedural signal present"),
    "EXECUTES":      ({"Procedure", "Function", "Block"}, {"SqlStatement"}, "runs a SQL statement"),
    "DB_OPERATION":  ({"Schema"},
                      {"Package", "Procedure", "Function", "Trigger", "Sequence",
                       "View", "Table", "Synonym", "Index", "DatabaseLink"},
                      "schema owns/operates on an object"),
    "READS":         ({"SqlStatement"}, {"Table", "View"}, "reads from relation"),
    "WRITES":        ({"SqlStatement", "Procedure", "Function"}, {"Table"}, "writes to table"),
    "USES_VIEW":     ({"SqlStatement"}, {"View"}, "depends on a view"),
    "REFERENCES":    ({"SqlStatement"}, {"Column"}, "references a column"),
    "USES_CURSOR":   ({"Loop", "Procedure", "Function"}, {"Cursor"}, "drives a cursor"),
    "USES_SEQUENCE": ({"SqlStatement", "Procedure", "Function"}, {"Sequence"}, "uses a sequence"),
    "BELONGS_TO":    ({"Column"}, {"Table", "View"}, "column of relation"),
    "INDEXES":       ({"Index"}, {"Table"}, "index covers table"),
    "ON_COLUMN":     ({"Index"}, {"Column"}, "index on column"),
    "REFERS_TO":     ({"Synonym"}, {"Table", "View", "Sequence", "External"},
                      "synonym alias target"),
    "DEPENDS_ON":    ({"View"}, {"Table", "View"}, "view depends on relation"),
    "VIA_DBLINK":    ({"SqlStatement", "Synonym"}, {"DatabaseLink"},
                      "reference through a database link"),
    # Knowledge layer
    "ASSESSED_AS":   ({"Procedure", "Function"}, {"MigrationAssessment"}, "migration verdict"),
    "USES_CONSTRUCT":({"SqlStatement", "Feature", "Procedure", "Function"},
                      {"OracleConstruct"}, "uses an Oracle construct"),
    "MAPS_TO":       ({"OracleConstruct"}, {"SparkConstruct"}, "Oracle -> Spark mapping"),
}



def node_labels(primary_label: str) -> List[str]:
    """All labels a node should carry: primary first, then synthetic super-labels.

    e.g. node_labels('Procedure') -> ['Procedure', 'Node', 'Resource']
         node_labels('SqlFile')   -> ['SqlFile', 'File', 'Node', 'Resource']
         node_labels('Block')     -> ['Block', 'Node']
    """
    labels = [primary_label]
    if primary_label in FILE_LABELS:
        labels.append("File")
    labels.append(BASE_LABEL)
    if primary_label in RESOURCE_LABELS:
        labels.append(SUPER_LABEL)
    return labels


def cypher_labels(primary_label: str) -> str:
    """Colon-joined label string for a Cypher MERGE, e.g. ':Resource:Procedure'."""
    return "".join(f":{lbl}" for lbl in node_labels(primary_label))


def is_resource(primary_label: str) -> bool:
    return primary_label in RESOURCE_LABELS


def cypher_constraints() -> List[str]:
    """Uniqueness constraints (also create the implicit index) for keyed nodes."""
    out = []
    for label in list(NODE_TYPES) + sorted(_SYNTHETIC_LABELS):
        out.append(
            f"CREATE CONSTRAINT IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE;"
        )
    return out


def validate_graph(graph) -> List[str]:
    """Return a list of schema violations; empty list means the graph conforms."""
    errors: List[str] = []
    for n in graph.nodes.values():
        if n.label not in NODE_TYPES:
            errors.append(f"unknown node label '{n.label}' (id={n.id})")
    for e in graph.edges:
        spec = REL_TYPES.get(e.rel)
        if spec is None:
            errors.append(f"unknown relationship '{e.rel}'")
            continue
        src_ok, dst_ok, _ = spec
        s = graph.nodes.get(e.src)
        d = graph.nodes.get(e.dst)
        if s and s.label not in src_ok:
            errors.append(f"{e.rel}: bad source label {s.label} (allowed {sorted(src_ok)})")
        if d and d.label not in dst_ok:
            errors.append(f"{e.rel}: bad dest label {d.label} (allowed {sorted(dst_ok)})")
    return errors
