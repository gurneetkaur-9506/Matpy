import ast

_BUILTINS = {
    "abs", "acos", "asin", "atan", "axis", "clc", "clear", "close", "cos",
    "disp", "end", "exp", "fft", "figure", "fprintf", "grid", "hold", "ifft",
    "length", "legend", "linspace", "log", "max", "mean", "min", "numel",
    "ones", "plot", "reshape", "sin", "size", "sqrt", "sum", "tan", "title",
    "xlabel", "ylabel", "zeros",
}


def _loop_from_node(node):
    loop_type = "for" if node.type == "for_statement" else "while"
    header = None
    body = None
    if loop_type == "for":
        for child in node.children:
            if child.type == "iterator":
                header = child.text.decode("utf-8")
            elif child.type == "block":
                body = child.text.decode("utf-8")
    else:
        # The 'while' keyword token must be skipped; the header is the
        # condition text between the keyword and the first body block.
        for child in node.children:
            if child.type == "while":
                continue
            if child.type == "block":
                body = child.text.decode("utf-8")
                break
            if header is None:
                header = child.text.decode("utf-8")
            else:
                header = "%s %s" % (header, child.text.decode("utf-8"))
    return {"type": loop_type, "header": header, "body": body}


def _collect_loops(node, loops):
    if node.type in ("for_statement", "while_statement"):
        loops.append(_loop_from_node(node))
        return
    for child in node.children:
        if child.type == "function_definition":
            continue
        _collect_loops(child, loops)


def _call_name(node):
    name = node.child_by_field_name("name")
    return name.text.decode("utf-8") if name is not None else None


def _index_like(args):
    args = [a for a in args if a.type not in ("(", ")", ",")]
    if not args:
        return False
    if any(a.type in ("spread_operator", "range") for a in args):
        return True
    return all(a.type in ("number", "identifier") for a in args)


def _collect_refs(node, refs):
    if node.type == "function_call":
        name = _call_name(node)
        args = []
        for c in node.children:
            if c.type == "arguments":
                args = [a for a in c.children if a.type not in ("(", ")", ",")]
        if name is not None and name not in _BUILTINS and _index_like(args):
            refs.append(
                {
                    "kind": "index",
                    "name": name,
                    "indices": [a.text.decode("utf-8") for a in args],
                }
            )
            return
        for a in args:
            _collect_refs(a, refs)
        return
    if node.type == "assignment":
        seen_eq = False
        for c in node.children:
            if c.type == "=":
                seen_eq = True
                continue
            if seen_eq:
                _collect_refs(c, refs)
        return
    if node.type == "for_statement":
        for c in node.children:
            if c.type == "iterator":
                skip = True
                for cc in c.children:
                    if cc.type == "=":
                        skip = False
                        continue
                    if not skip:
                        _collect_refs(cc, refs)
            else:
                _collect_refs(c, refs)
        return
    if node.type in ("command", "command_name", "command_argument"):
        return
    if node.type == "identifier":
        refs.append({"kind": "plain", "name": node.text.decode("utf-8")})
        return
    for c in node.children:
        _collect_refs(c, refs)


def _py_loop_from_node(node):
    if isinstance(node, ast.For):
        loop_type = "for"
        header = "for %s in %s" % (ast.unparse(node.target), ast.unparse(node.iter))
    else:
        loop_type = "while"
        header = "while %s" % ast.unparse(node.test)
    body = "\n".join(ast.unparse(s) for s in node.body)
    return {"type": loop_type, "header": header, "body": body}


def _py_collect_loops(node, loops):
    if isinstance(node, (ast.For, ast.While)):
        loops.append(_py_loop_from_node(node))
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            continue
        _py_collect_loops(child, loops)


def _py_slice_index(slice_node):
    if isinstance(slice_node, ast.Slice):
        parts = []
        for field in ("lower", "upper", "step"):
            value = getattr(slice_node, field, None)
            if value is not None:
                parts.append(ast.unparse(value))
        if not parts:
            return ":"
        return ":".join(parts)
    return ast.unparse(slice_node)


def _py_index_ref(subscript):
    if not isinstance(subscript.value, ast.Name):
        return None
    slice_node = subscript.slice
    if isinstance(slice_node, ast.Tuple):
        indices = [_py_slice_index(elt) for elt in slice_node.elts]
    else:
        indices = [_py_slice_index(slice_node)]
    return {"kind": "index", "name": subscript.value.id, "indices": indices}


def _py_collect_refs(node, refs):
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load):
            refs.append({"kind": "plain", "name": node.id})
        return
    if isinstance(node, ast.Subscript):
        ref = _py_index_ref(node)
        if ref is not None:
            refs.append(ref)
        return
    if isinstance(node, ast.FunctionDef):
        return
    for child in ast.iter_child_nodes(node):
        _py_collect_refs(child, refs)


def _extract_python(tree):
    functions = []
    refs = []
    for child in tree.body:
        if isinstance(child, ast.FunctionDef):
            loops = []
            _py_collect_loops(child, loops)
            function_refs = []
            for stmt in child.body:
                _py_collect_refs(stmt, function_refs)
            functions.append(
                {
                    "name": child.name,
                    "parameters": [a.arg for a in child.args.args],
                    "body": "\n".join(ast.unparse(s) for s in child.body),
                    "loops": loops,
                    "refs": function_refs,
                }
            )
        else:
            _py_collect_refs(child, refs)
    return {"functions": functions, "refs": refs}


def extract_structure(parse_tree):
    if isinstance(parse_tree, ast.AST):
        return _extract_python(parse_tree)
    root = getattr(parse_tree, "root_node", parse_tree)
    functions = []
    refs = []

    for child in root.children:
        if child.type == "function_definition":
            name = None
            body = None
            loops = []
            block = None
            arguments = None
            for c in child.children:
                if name is None and c.type == "identifier":
                    name = c.text.decode("utf-8")
                elif c.type == "function_arguments":
                    arguments = c
                elif c.type == "block":
                    body = c.text.decode("utf-8")
                    block = c
            _collect_loops(child, loops)
            function_refs = []
            if block is not None:
                _collect_refs(block, function_refs)
            functions.append(
                {
                    "name": name,
                    "parameters": [
                        a.text.decode("utf-8")
                        for a in arguments.children
                        if a.type == "identifier"
                    ]
                    if arguments is not None
                    else [],
                    "body": body,
                    "loops": loops,
                    "refs": function_refs,
                }
            )
        else:
            _collect_refs(child, refs)

    return {"functions": functions, "refs": refs}
