from dataclasses import asdict, dataclass, field

from .extract_structure import _collect_refs, _loop_from_node


@dataclass
class IndexExpr:
    name: str
    indices: list


@dataclass
class PlainRef:
    name: str


@dataclass
class Loop:
    type: str
    header: str
    statements: list = field(default_factory=list)


@dataclass
class Statement:
    kind: str
    text: str


@dataclass
class Function:
    name: str
    statements: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    refs: list = field(default_factory=list)
    indices: list = field(default_factory=list)


@dataclass
class Structure:
    functions: list = field(default_factory=list)
    statements: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    refs: list = field(default_factory=list)
    indices: list = field(default_factory=list)


def _split_refs(entries):
    plains = [PlainRef(e["name"]) for e in entries if e["kind"] == "plain"]
    indices = [IndexExpr(e["name"], e["indices"]) for e in entries if e["kind"] == "index"]
    return plains, indices


def _statements_from_container(container):
    if container is None:
        return []
    statements = []
    for child in container.children:
        if not child.is_named or child.type == "comment":
            continue
        if child.type in ("for_statement", "while_statement"):
            info = _loop_from_node(child)
            body_block = next((c for c in child.children if c.type == "block"), None)
            inner = _statements_from_container(body_block)
            statements.append(Loop(info["type"], info["header"], inner))
        else:
            statements.append(Statement(child.type, child.text.decode("utf-8")))
    return statements


def _collect_for(node):
    entries = []
    _collect_refs(node, entries)
    return _split_refs(entries)


def _flatten_loops(statements):
    loops = []
    for s in statements:
        if isinstance(s, Loop):
            loops.append(s)
            loops.extend(_flatten_loops(s.statements))
    return loops


def build_structure(parse_tree):
    root = getattr(parse_tree, "root_node", parse_tree)
    structure = Structure()

    for child in root.children:
        if not child.is_named or child.type == "comment":
            continue
        if child.type == "function_definition":
            name = None
            block = None
            for c in child.children:
                if name is None and c.type == "identifier":
                    name = c.text.decode("utf-8")
                elif c.type == "block":
                    block = c
            statements = _statements_from_container(block)
            plains, indices = _collect_for(block)
            structure.functions.append(
                Function(name, statements, _flatten_loops(statements), plains, indices)
            )
        else:
            structure.statements.append(Statement(child.type, child.text.decode("utf-8")))

    top_entries = []
    for child in root.children:
        if child.is_named and child.type not in ("function_definition", "comment"):
            _collect_refs(child, top_entries)
    structure.refs, structure.indices = _split_refs(top_entries)
    structure.loops = _flatten_loops(structure.statements)
    return structure


def structure_to_dict(structure):
    return asdict(structure)
