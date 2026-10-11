import abc
import annotationlib
import functools as ft
import types
import typing as tp

_RECURSION_LIMIT = 16
_TypeParameter: tp.TypeAlias = tp.TypeVar | tp.TypeVarTuple | tp.ParamSpec


def _get_value_or_forwardref(f: tp.Callable, /):
    return annotationlib.call_evaluate_function(f, annotationlib.Format.FORWARDREF)


def _is_indeterminate_tuple(tup: type[tuple], /):
    return (
        (tp.get_origin(tup) or tup) is tuple
        and len(args := tp.get_args(tup)) == 2
        and args[-1] is Ellipsis
    )


def _is_unpacked(typx, /) -> bool:
    return getattr(typx, "__unpacked__", False)


def _unpack_tuple_items(typx, /) -> tuple[tp.Any, ...]:
    if tp.get_origin(typx) is tp.Unpack and not tp.is_typeddict(
        tup := tp.get_args(typx)[0]
    ):
        typx = tup
    if _is_unpacked(typx) and not _is_indeterminate_tuple(typx):
        return tp.get_args(typx)
    return (typx,)


def _get_tuple_arg_indices(
    typx: type[tuple], /
) -> tuple[
    tuple[tuple[tp.Any, ...] | tp.Any, slice[int | None, int | None, None]], ...
]:
    args = tp.get_args(typx)
    idx = next(
        (
            i
            for i, x in enumerate(args)
            if _is_unpacked(x) and _is_indeterminate_tuple(x)
        ),
        None,
    )
    if idx is None:
        out = ((args, slice(None)),)
    else:
        # the tuple generic alias is bisected by an unpacked tuple of
        # indefinite length.
        [t_indef, _] = tp.get_args(args[idx])
        if 0 < idx < len(args) - 1:
            head, tail = args[:idx], args[idx + 1 :]
            ni = len(head)
            nj = len(tail)
            out = (
                (head, slice(ni)),
                (t_indef, slice(ni, -nj)),
                (tail, slice(-nj, None)),
            )
        elif idx == 0:
            tail = args[idx + 1 :]
            n = len(tail)
            out = (t_indef, slice(-n)), (tail, slice(-n, None))
        else:
            head = args[:idx]
            n = len(head)
            out = (head, slice(n)), (t_indef, slice(n, None))
    return out


def _add_default_params(
    src_param_lst: tuple[_TypeParameter, ...], dst_params: dict[_TypeParameter, tp.Any]
):
    dst_params.update(
        (param, _get_value_or_forwardref(param.evaluate_default))
        for param in src_param_lst
        if param.has_default()
        if param.evaluate_default
    )


def _get_dst_params(typx, /):
    src_param_lst = getattr(typx, "__type_params__", ())
    if not src_param_lst:
        return {}
    src_args = tp.get_args(typx)
    dst_params = dict(zip(src_param_lst, src_args))
    if unresolved := src_param_lst[len(src_args) :]:
        _add_default_params(unresolved, dst_params)
    return dst_params


def _make_typevars_bound(
    typx, /, type_params: dict[_TypeParameter, tp.Any] | None = None, recursion_count=0
):
    if type_params is None:
        type_params = {}
    if type(typx) is tp.ForwardRef:
        try:
            typx = typx.evaluate(type_params=tuple(type_params))
        except NameError:
            return typx
    if recursion_count > _RECURSION_LIMIT:
        return object
    unpacked = _is_unpacked(typx)
    if args := tp.get_args(typx):
        type_params = _get_dst_params(typx) | type_params
        origin = tp.get_origin(typx) or typx
        if isinstance(origin, tp.TypeAliasType):
            typx = _make_typevars_bound(
                _get_value_or_forwardref(origin.evaluate_value),
                type_params,
                recursion_count + 1,
            )
            origin = tp.get_origin(typx) or typx
            args = tp.get_args(typx)
        if hasattr(typx, "copy_with"):
            copy_with = typx.copy_with
        else:
            copy_with = origin.__class_getitem__
        new_args = []
        for arg in args:
            new_arg = _make_typevars_bound(
                (type_params.get(arg, arg) if isinstance(arg, _TypeParameter) else arg),
                type_params,
                recursion_count + 1,
            )
            new_args.extend(_unpack_tuple_items(new_arg))
        typx = copy_with(tuple(new_args))
        # handle *tuple[...] star-unpack form
        if unpacked and type(typx) is types.GenericAlias and not _is_unpacked(typx):
            typx = next(iter(typx))
    return typx


class _SupportsGeneric[T = type[tp.Any]](tp.Protocol):
    __origin__: T
    __args__: tuple[object, ...]


class _checker_descriptor(abc.ABC):
    def __init__(self, typx: _SupportsGeneric, /):
        self.__args__ = typx.__args__

    def __get__(self, inst, owner=None, /):
        if inst is None:
            return self
        return self.__call__

    @abc.abstractmethod
    def __call__(self, obj: tp.Any, /) -> bool: ...


class _literal_instancecheck(_checker_descriptor):
    def __call__(self, obj, /):
        return obj in self.__args__


class _literal_subclasscheck(_checker_descriptor):
    def __call__(self, obj, /):
        return tp.get_origin(obj) is tp.Literal and set(tp.get_args(obj)).issubset(
            self.__args__
        )


@ft.cache
def _builtin_overloads() -> types.MappingProxyType[str, tuple[types.FunctionType, ...]]:
    import ast

    class TypeAliasTransformer(ast.NodeTransformer):
        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
            # check if this is a self-referential TypeAlias
            if (
                isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Name)
                and node.annotation.id == "TypeAlias"
                and node.value is not None
                and any(
                    isinstance(n, ast.Name) and n.id == node.target.id
                    for n in ast.walk(node.value)
                )
            ):
                # convert to PEP 695 type statement to lazily evaluate
                # the self-reference.
                return ast.copy_location(
                    ast.TypeAlias(
                        ast.Name(node.target.id, ast.Store()), [], node.value
                    ),
                    node,
                )
            return node

    class OverloadCollector(TypeAliasTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import _collections_abc
            import builtins

            abc_, builtins_ = map(vars, [_collections_abc, builtins])
            self.builtins = {
                k: v
                for k, v in map(
                    lambda k, d=(abc_ | builtins_): (k, d[k]),
                    abc_.keys() ^ builtins_.keys(),
                )
                if not isinstance(v, types.ModuleType)
                if callable(v) or not k.startswith("_")
            }
            self.overloads = None

        def collect(self, node, _globals=None, /):
            res = ast.fix_missing_locations(self.visit(node))
            prev = tp._overload_registry
            try:
                self.overloads = tp._overload_registry.__class__(
                    tp._overload_registry.default_factory
                )
                tp._overload_registry = self.overloads
                exec(compile(res, "<ast>", "exec", dont_inherit=True), _globals or {})
            finally:
                tp._overload_registry = prev
            return self.overloads

        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
            if node.module == "typing":
                # type_check_only does not exist at runtime
                node.names = [a for a in node.names if a.name != "type_check_only"]
            return node

        def _body(self, body: list[ast.stmt], depth: int = 0) -> list[ast.stmt]:
            out: list[ast.stmt] = []
            for s in body:
                if isinstance(s, ast.If):
                    s.body = self._body(s.body, depth)
                    s.orelse = self._body(s.orelse, depth) if s.orelse else []
                    out.append(s)
                elif isinstance(s, ast.ClassDef):
                    if s.name in self.builtins:
                        s.bases, s.keywords, s.decorator_list = [], [], []
                        s.body = self._body(s.body, depth + 1)
                        out += [s, ast.Delete([ast.Name(s.name, ast.Del())])]
                    else:
                        s.decorator_list = [
                            d
                            for d in s.decorator_list
                            if not (
                                isinstance(d, ast.Name) and d.id == "type_check_only"
                            )
                        ]
                        out.append(s)
                elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if depth or s.name in self.builtins:
                        is_prop = False
                        for d in s.decorator_list:
                            if isinstance(d, ast.Name) and d.id in {
                                "property",
                                "overload",
                            }:
                                is_prop = d.id == "property"
                                break
                            if (
                                isinstance(d, ast.Attribute)
                                and getattr(d.value, "id", None) == s.name
                            ):
                                is_prop = True
                                break
                        else:
                            s.decorator_list.append(ast.Name("overload", ast.Load()))
                        if not is_prop:
                            self.generic_visit(s)
                        out.append(s)
                        if not depth:
                            out.append(ast.Delete([ast.Name(s.name, ast.Del())]))
                    else:
                        out.append(s)
                elif not depth:
                    out.append(self.visit(s))
            return out or [ast.Pass()]

        def visit_Module(self, node: ast.Module) -> ast.Module:
            node.body = self._body(node.body)
            return node

        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            self.generic_visit(node)
            # types like slice, zip, and dict_items are subscriptable in
            # annotations, but do not define __class_getitem__ at runtime.
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in self.builtins
                and not hasattr(self.builtins[node.value.id], "__class_getitem__")
            ):
                # emit only the subscripted name
                return node.value
            return node

    import sys
    from importlib.machinery import SourceFileLoader
    from importlib.resources import as_file, files
    from importlib.util import module_from_spec, spec_from_loader

    class PyiLoader(SourceFileLoader):
        def source_to_code(self, data, path, *, _optimize=-1):
            tree = TypeAliasTransformer().visit(ast.parse(data, path))
            return compile(
                ast.fix_missing_locations(tree), path, "exec", dont_inherit=True
            )

        def path_stats(self, path):
            st = super().path_stats(path)
            from os import stat

            return {**st, "mtime": max(st["mtime"], stat(__file__).st_mtime)}

    STUBS = files(__name__) / "_stubs"
    with as_file(STUBS / "_typeshed" / "__init__.pyi") as path:
        spec = spec_from_loader(
            "_typeshed", PyiLoader("_typeshed", str(path)), is_package=True
        )
        assert spec is not None
        assert spec.loader is not None
        sys.modules["_typeshed"] = mod = module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            out = OverloadCollector().collect(
                ast.parse((STUBS / "builtins.pyi").read_bytes()),
                {"__name__": "builtins"},
            )
        finally:
            del sys.modules["_typeshed"]
    return types.MappingProxyType(
        {qualname: tuple(ovs.values()) for qualname, ovs in out["builtins"].items()}
    )
