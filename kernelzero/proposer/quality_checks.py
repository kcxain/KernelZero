"""Static quality checks shared by dataset filtering and Proposer rewards."""

import ast
import hashlib

DESTROYING_LIKE_OPS = {
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "new_zeros",
    "new_empty",
    "new_ones",
}
VALUE_INDEPENDENT_FACTORIES = {
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "zeros",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "randint",
    "linspace",
    "logspace",
    "eye",
}
META_ATTRIBUTES = {"shape", "device", "dtype", "ndim"}


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def variants(operator):
    leaf = operator.rsplit(".", 1)[-1]
    return {operator, leaf, f"torch.{leaf}", f"nn.{leaf}", f"F.{leaf}"}


def module_hash(module):
    operators = "|".join(sorted(module.get("operators", [])))
    content = f"{module.get('code', '').strip()}|{operators}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def structural_hash(module):
    """Collapse variants that differ only in literal constants or formatting."""

    class NormalizeConstants(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, bool) or node.value is None:
                return node
            if isinstance(node.value, (int, float, complex)):
                return ast.copy_location(ast.Constant(value="__NUM__"), node)
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value="__STR__"), node)
            return node

    tree = NormalizeConstants().visit(ast.parse(module.get("code", "")))
    normalized = ast.dump(tree, include_attributes=False)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def assigned_names(statement):
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def assigned_self_attributes(statement):
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    return [
        dotted_name(target)
        for target in targets
        if isinstance(target, ast.Attribute) and dotted_name(target).startswith("self.")
    ]


def direct_calls(node):
    return [dotted_name(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)]


def output_depends_on_forward_value(forward):
    dependencies = {
        argument.arg: True for argument in forward.args.args if argument.arg != "self"
    }

    def depends(node):
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return dependencies.get(node.id, False)
        if isinstance(node, ast.Attribute):
            if node.attr in META_ATTRIBUTES:
                return False
            return depends(node.value)
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Call):
            leaf = dotted_name(node.func).rsplit(".", 1)[-1]
            if leaf in VALUE_INDEPENDENT_FACTORIES:
                return False
            if leaf == "arange":
                return any(depends(argument) for argument in node.args)
            receiver_dependency = (
                depends(node.func.value)
                if isinstance(node.func, ast.Attribute)
                and dotted_name(node.func.value) not in {"self", "torch", "nn", "F"}
                else False
            )
            return receiver_dependency or any(depends(argument) for argument in node.args) or any(
                depends(keyword.value) for keyword in node.keywords if keyword.arg != "out"
            )
        return any(depends(child) for child in ast.iter_child_nodes(node))

    return_dependency = False
    for statement in forward.body:
        if isinstance(statement, ast.Assign):
            value_dependency = depends(statement.value)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    dependencies[target.id] = value_dependency
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            dependencies[statement.target.id] = depends(statement.value)
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            dependencies[statement.target.id] = dependencies.get(
                statement.target.id, False
            ) or depends(statement.value)
        elif isinstance(statement, ast.Return):
            return_dependency = return_dependency or depends(statement.value)
        else:
            for nested in ast.walk(statement):
                if isinstance(nested, ast.Return):
                    return_dependency = return_dependency or depends(nested.value)
    return return_dependency


def quality_issues(module):
    code = module.get("code", "")
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax:{exc}"]

    model_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"]
    if not model_classes:
        return ["missing_model"]
    model = model_classes[0]
    forward = next(
        (node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward"),
        None,
    )
    init = next(
        (node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
        None,
    )
    if forward is None:
        return ["missing_forward"]

    if not output_depends_on_forward_value(forward):
        issues.append("output_not_dependent_on_forward_input_values")

    # Reject required calls whose result is never consumed, and value-destroying
    # overwrites such as x = torch.zeros_like(x).
    for index, statement in enumerate(forward.body):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        calls = direct_calls(value)
        rhs_names = {
            node.id
            for node in ast.walk(value)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        later_names = {
            node.id
            for later in forward.body[index + 1 :]
            for node in ast.walk(later)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for target in assigned_names(statement):
            for operator in module.get("operators", []):
                if any(call in variants(operator) for call in calls) and target not in later_names:
                    issues.append(f"unused_required_result:{operator}")
            if target in rhs_names and any(
                call.rsplit(".", 1)[-1] in DESTROYING_LIKE_OPS for call in calls
            ):
                issues.append("value_destroying_overwrite")

    # Required operators must contribute to forward. Functional calls should occur
    # directly in forward. Modules/Parameters may be created in __init__, but the
    # corresponding self attribute must then be referenced by forward.
    forward_calls = set(direct_calls(forward))
    forward_attributes = {
        dotted_name(node) for node in ast.walk(forward) if isinstance(node, ast.Attribute)
    }
    init_statements = (
        [node for node in ast.walk(init) if isinstance(node, (ast.Assign, ast.AnnAssign))]
        if init is not None
        else []
    )
    init_calls = set(direct_calls(init)) if init is not None else set()
    used_sequential = any(attribute in forward_attributes for attribute in {
        self_attr
        for statement in init_statements
        for self_attr in assigned_self_attributes(statement)
        if any(call.endswith("Sequential") for call in direct_calls(statement.value))
    })

    for operator in module.get("operators", []):
        op_variants = variants(operator)
        contributes = any(call in op_variants for call in forward_calls)

        if not contributes and init is not None:
            for statement in init_statements:
                value = statement.value
                mentions_operator = any(call in op_variants for call in direct_calls(value)) or any(
                    dotted_name(node) in op_variants
                    for node in ast.walk(value)
                    if isinstance(node, (ast.Name, ast.Attribute))
                )
                if mentions_operator and any(
                    attribute in forward_attributes for attribute in assigned_self_attributes(statement)
                ):
                    contributes = True
                    break

        # Modules placed in a local list and then wrapped by a used Sequential
        # contribute even though the intermediate list is not a self attribute.
        is_module = operator.startswith("torch.nn.") and not operator.startswith(
            "torch.nn.functional."
        )
        if not contributes and is_module and used_sequential:
            contributes = any(call in op_variants for call in init_calls)

        if not contributes:
            issues.append(f"required_operator_not_in_forward_graph:{operator}")

    if len(code) < 250:
        issues.append("code_too_short")
    if len(code) > 4000:
        issues.append("code_too_long")
    if "```" in module.get("raw_response", ""):
        issues.append("markdown_wrapped_response")
    return sorted(set(issues))
