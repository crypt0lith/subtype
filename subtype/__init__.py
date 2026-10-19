__all__ = ["isinstance", "subtype"]
import abc
import annotationlib
import ast
import builtins
import collections.abc
import enum
import functools as ft
import inspect
import sys
import types
import typing as tp
import weakref
from builtins import isinstance as isinstance_

_RECURSION_LIMIT = 16
_TypeParameter: tp.TypeAlias = tp.TypeVar | tp.TypeVarTuple | tp.ParamSpec
_MemberPairs: tp.TypeAlias = tuple[
    tuple[tuple[inspect.Signature, ...], tuple[inspect.Signature, ...]], ...
]


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
        if tp.get_origin(tup) is tuple:
            # Unpack[Tuple[...]] => *tuple[...]
            [typx] = tuple[*tp.get_args(tup)]
        else:
            typx = tup
    if _is_unpacked(typx) and not _is_indeterminate_tuple(typx):
        return tp.get_args(typx)
    return (typx,)


def _tuple_entries(typx: type[tuple], /) -> list[tuple[tp.Any, bool]] | None:
    """Return the element types of typx, each paired with whether it is
    indeterminate, or None if typx unpacks a TypeVarTuple.
    """
    if not hasattr(typx, "__args__"):
        # tuple => tuple[object, ...]
        return [(object, True)]
    if _is_indeterminate_tuple(typx):
        return [(tp.get_args(typx)[0], True)]
    entries = []
    for arg in tp.get_args(typx):
        for x in _unpack_tuple_items(arg):
            if isinstance_(x, tp.TypeVarTuple):
                return
            if _is_unpacked(x) and _is_indeterminate_tuple(x):
                entries.append((tp.get_args(x)[0], True))
            else:
                entries.append((x, False))
    return entries


def _adjust_tuple_entries(
    dst: list[tuple[tp.Any, bool]], src: list[tuple[tp.Any, bool]], /
) -> bool:
    """Pack the src entries that match the indeterminate entry of dst into one,
    and return whether dst and src then have the same length.
    """
    dst_i, src_i = (
        next((i for i, (_, indeterminate) in enumerate(x) if indeterminate), -1)
        for x in (dst, src)
    )
    if src_i >= 0:
        if src[src_i][0] is tp.Any:
            # an indeterminate Any expands to the length of dst
            while len(src) < len(dst):
                src.insert(src_i, src[src_i])
            if len(src) > len(dst):
                del src[src_i]
                src_i = -1
        elif dst_i < 0:
            return False
    if dst_i >= 0 and dst[dst_i][0] is tp.Any:
        while len(dst) < len(src):
            dst.insert(dst_i, dst[dst_i])
    n = len(src) - len(dst) + 1
    if dst_i >= 0 and n >= 0 and (src_i < 0 or dst_i <= src_i < dst_i + n):
        packed = [t for t, _ in src[dst_i : dst_i + n]]
        src[dst_i : dst_i + n] = [(tp.Union[*packed] if packed else tp.Any, False)]
    return len(dst) == len(src)


def _get_value_or_forwardref(f: tp.Callable, /):
    return annotationlib.call_evaluate_function(f, annotationlib.Format.FORWARDREF)


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
        if isinstance_(origin, tp.TypeAliasType):
            typx = _make_typevars_bound(
                _get_value_or_forwardref(origin.evaluate_value),
                type_params,
                recursion_count + 1,
            )
            origin = tp.get_origin(typx) or typx
            args = tp.get_args(typx)
        if hasattr(typx, "copy_with"):
            copy_with = typx.copy_with
        elif not hasattr(origin, "__class_getitem__"):
            return origin
        else:
            copy_with = origin.__class_getitem__
        new_args = []
        for arg in args:
            new_arg = _make_typevars_bound(
                (
                    type_params.get(arg, arg)
                    if isinstance_(arg, _TypeParameter)
                    else arg
                ),
                type_params,
                recursion_count + 1,
            )
            new_args.extend(_unpack_tuple_items(new_arg))
        typx = copy_with(tuple(new_args))
        # handle *tuple[...] star-unpack form
        if unpacked and type(typx) is types.GenericAlias and not _is_unpacked(typx):
            typx = next(iter(typx))
    return typx


class _ForwardRefSubscriptTransformer(ast.NodeTransformer):
    def __init__(
        self,
        *,
        globals: tp.Mapping[str, tp.Any] | None = None,
        locals: tp.Mapping[str, tp.Any] | None = None,
    ):
        self._ns = types.MappingProxyType(
            {**builtins.__dict__, **(globals or {}), **(locals or {})}
        )

    def visit_Subscript(self, node):
        self.generic_visit(node)
        # types like slice, zip, and dict_items are subscriptable in
        # annotations, but do not define __class_getitem__ at runtime.
        if (
            isinstance_(node.value, ast.Name)
            and node.value.id in self._ns
            and not hasattr(self._ns[node.value.id], "__class_getitem__")
        ):
            return ast.Call(
                ast.Name("__generic_alias__", ast.Load()), [node.value, node.slice], []
            )
        return node

    @classmethod
    def evaluate_forwardref(
        cls,
        forwardref: tp.ForwardRef,
        *,
        globals: dict[str, tp.Any] | None = None,
        locals: tp.Mapping[str, tp.Any] | None = None,
    ):
        if globals is None:
            globals = forwardref.__globals__
        if locals is None:
            locals = {}
        else:
            locals = dict(locals)
        if isinstance_(forwardref.__cell__, dict):
            for cell_name, cell in forwardref.__cell__.items():
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    pass
                else:
                    locals.setdefault(cell_name, cell_value)
        if forwardref.__extra_names__:
            locals.update(forwardref.__extra_names__)
        locals["__generic_alias__"] = types.GenericAlias
        if (node := forwardref.__ast_node__) is None:
            node = ast.parse(forwardref.__forward_arg__, mode="eval").body
        tree = ast.fix_missing_locations(
            cls(globals=globals, locals=locals).visit(ast.Expression(node))
        )
        code = compile(tree, "<ast>", "eval")
        return eval(code, globals, locals)


def _evaluate(typx: tp.TypeAliasType | tp.ForwardRef, /):
    if isinstance_(typx, tp.ForwardRef):
        return _ForwardRefSubscriptTransformer.evaluate_forwardref(typx)
    if tp.get_args(typx):
        return _make_typevars_bound(typx)
    return typx.__value__


class _SupportsGeneric[T = type[tp.Any]](tp.Protocol):
    __origin__: T
    __args__: tuple[object, ...]


class _AssignFlag(enum.IntFlag):
    DEFAULT = 0
    # src must also accept dst
    INVARIANT = enum.auto()
    # dst and src are swapped
    CONTRAVARIANT = enum.auto()


class _AssignContext(tp.TypedDict, total=False):
    assuming: frozenset[tuple[subtype, tp.Any]]
    assumed_variance: tp.Mapping[type, _AssignFlag]
    recursion_count: int
    seen: set[tuple[int, tp.Any]]
    self_class: tp.Any


class _checker(abc.ABC):
    def __init__(self, typx: _SupportsGeneric, /):
        self.__args__ = typx.__args__

    @abc.abstractmethod
    def __call__(self, obj: tp.Any, /, **ctx: tp.Unpack[_AssignContext]) -> bool: ...


@ft.cache
def _builtin_callables():
    import _collections_abc

    out = {
        k: v
        for k, v in map(
            lambda k, d=(_collections_abc.__dict__ | builtins.__dict__): (k, d[k]),
            _collections_abc.__dict__.keys() ^ builtins.__dict__.keys(),
        )
        if callable(v)
        if hasattr(builtins, k) or not k.startswith("_")
    }
    return types.MappingProxyType(out)


@ft.cache
def _load_stubs():
    class TypeAliasTransformer(_ForwardRefSubscriptTransformer):
        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
            # check if this is a self-referential TypeAlias
            if (
                isinstance_(node.target, ast.Name)
                and isinstance_(node.annotation, ast.Name)
                and node.annotation.id == "TypeAlias"
                and node.value is not None
                and any(
                    isinstance_(n, ast.Name) and n.id == node.target.id
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
        def __init__(self, module: types.ModuleType, /):
            super().__init__(globals=_builtin_callables())
            self._module = module
            # __builtins__ fallback for the stub definitions _body deletes
            self._runtime = builtins.__dict__ | {"overload": tp.overload} | vars(module)
            self.overloads = None
            self.classes = {}

        def collect(self, node, /):
            res = ast.fix_missing_locations(self.visit(node))
            name = self._module.__name__
            prev = tp._overload_registry
            try:
                self.overloads = tp._overload_registry.__class__(
                    tp._overload_registry.default_factory
                )
                tp._overload_registry = self.overloads
                exec(
                    compile(res, "<ast>", "exec", dont_inherit=True),
                    {
                        "__name__": name,
                        "__builtins__": self._runtime,
                        "__stub_class__": self._register,
                        "__generic_alias__": types.GenericAlias,
                    },
                )
            finally:
                tp._overload_registry = prev
            return self.overloads[name]

        def _register(self, typx: type, /):
            self.classes[typx.__name__] = typx
            return typx

        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
            if node.module == "typing":
                # type_check_only does not exist at runtime
                node.names = [a for a in node.names if a.name != "type_check_only"]
            return node

        def _body(self, body: list[ast.stmt], depth: int = 0) -> list[ast.stmt]:
            out: list[ast.stmt] = []
            for s in body:
                if isinstance_(s, ast.If):
                    s.body = self._body(s.body, depth)
                    s.orelse = self._body(s.orelse, depth) if s.orelse else []
                    out.append(s)
                elif isinstance_(s, ast.ClassDef):
                    if s.name in self._runtime:
                        bases = s.bases
                        s.bases, s.keywords, s.decorator_list = [], [], []
                        s.body = self._body(s.body, depth + 1)
                        if bases and not depth:
                            # evaluated once the module no longer binds the
                            # stub definitions
                            s.body.append(
                                ast.AnnAssign(
                                    ast.Name("__orig_bases__", ast.Store()),
                                    self.visit(ast.Tuple(bases, ast.Load())),
                                    None,
                                    simple=1,
                                )
                            )
                            s.decorator_list = [ast.Name("__stub_class__", ast.Load())]
                        out += [s, ast.Delete([ast.Name(s.name, ast.Del())])]
                    else:
                        s.decorator_list = [
                            d
                            for d in s.decorator_list
                            if not (
                                isinstance_(d, ast.Name) and d.id == "type_check_only"
                            )
                        ]
                        out.append(s)
                elif isinstance_(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if depth or s.name in self._runtime:
                        is_prop = False
                        for d in s.decorator_list:
                            if isinstance_(d, ast.Name) and d.id in {
                                "property",
                                "overload",
                            }:
                                is_prop = d.id == "property"
                                break
                            if (
                                isinstance_(d, ast.Attribute)
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

    import _collections_abc
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

    def runtime_class(module, name):
        owner = vars(module).get(name)
        owner = tp.get_origin(owner) or owner
        return owner if isinstance_(owner, type) else None

    STUBS = files(__name__) / "typeshed" / "stdlib"
    out = {}
    bases = {}
    with as_file(STUBS / "_typeshed" / "__init__.pyi") as path:
        spec = spec_from_loader(
            "_typeshed", PyiLoader("_typeshed", str(path)), is_package=True
        )
        assert spec is not None
        assert spec.loader is not None
        sys.modules["_typeshed"] = mod = module_from_spec(spec)
        vars(mod)["__generic_alias__"] = types.GenericAlias
        try:
            spec.loader.exec_module(mod)
            for stub, module in (
                ("builtins.pyi", builtins),
                ("_collections_abc.pyi", _collections_abc),
                ("typing.pyi", tp),
            ):
                collector = OverloadCollector(module)
                collected = collector.collect(ast.parse((STUBS / stub).read_bytes()))
                for qualname, ovs in collected.items():
                    head, sep, name = qualname.partition(".")
                    if not sep:
                        owner, name = module, head
                    elif (owner := runtime_class(module, head)) is None:
                        continue
                    out.setdefault(owner, {})[name] = tuple(ovs.values())
                for name, typx in collector.classes.items():
                    if (owner := runtime_class(module, name)) is not None:
                        bases[owner] = annotationlib.get_annotations(
                            typx, format=annotationlib.Format.FORWARDREF
                        )["__orig_bases__"]
        finally:
            del sys.modules["_typeshed"]
    overloads = {owner: types.MappingProxyType(d) for owner, d in out.items()}
    return types.MappingProxyType(overloads), types.MappingProxyType(bases)


def _stub_overloads() -> (
    types.MappingProxyType[
        type | types.ModuleType,
        types.MappingProxyType[str, tuple[types.FunctionType, ...]],
    ]
):
    """Map runtime classes and modules to their stub overloads, by name."""
    return _load_stubs()[0]


def _stub_bases() -> types.MappingProxyType[type, tuple[tp.Any, ...]]:
    """Map runtime classes to their generic bases in the stubs."""
    return _load_stubs()[1]


@ft.cache
def _stub_functions() -> (
    types.MappingProxyType[types.FunctionType, tuple[types.FunctionType, ...]]
):
    """Map runtime Python functions to their stub overloads."""
    out = {}
    for owner, members in _stub_overloads().items():
        for name, ovs in members.items():
            attr = vars(owner).get(name)
            # unwrap __func__ from a classmethod or staticmethod
            if isinstance_(func := getattr(attr, "__func__", attr), types.FunctionType):
                out[func] = ovs
    return types.MappingProxyType(out)


def _resolve_call(obj, /) -> tuple[tp.Sequence[tp.Callable], bool]:
    """Return the overloads that calling obj calls, and whether the call binds
    their self or cls parameter.
    """

    def lookup(mro, /, *names: str):
        for typx in mro:
            if (members := _stub_overloads().get(typx)) is not None:
                decls = [members[name] for name in names if name in members]
            else:
                attrs = (
                    (name, vars(typx)[name]) for name in names if name in vars(typx)
                )
                decls = [
                    (
                        _get_member_overloads(typx, name)
                        if isinstance_(getattr(a, "__func__", a), types.FunctionType)
                        else ()
                    )
                    for name, a in attrs
                ]
            if decls:
                return max(decls, key=len)
        return ()

    objtype = type(obj)
    if isinstance_(obj, types.FunctionType) or hasattr(objtype, "__func__"):
        ovs = tp.get_overloads(obj) or _stub_functions().get(
            getattr(obj, "__func__", obj), ()
        )
        return ovs, hasattr(objtype, "__self__")
    if hasattr(objtype, "__self__"):
        # __self__ is None for C static methods, so get the module name
        # or self type from __reduce__ instead.
        match obj.__reduce__():
            case str(name):
                # sys.modules[obj.__module__][name] => function
                members = _stub_overloads().get(sys.modules[obj.__module__], {})
                return members.get(name, ()), False
            case _, (m_self, name):
                # getattr(m_self, name) => unbound method
                if isinstance_(m_self, type):
                    # include the metaclass in attribute lookup
                    mro = (*m_self.__mro__, *type(m_self).__mro__)
                else:
                    mro = type(m_self).__mro__
                # special case for tp_new: __new__ is a C static method
                # where __self__ is m_self
                is_bound = getattr(obj, "__self__", None) is m_self
                return lookup(mro, name), is_bound and name != "__new__"
    if hasattr(objtype, "__objclass__"):
        # unbound C method descriptor
        return lookup(obj.__objclass__.__mro__, obj.__name__), False
    owner = next(typx for typx in objtype.__mro__ if "__call__" in vars(typx))
    call = vars(owner)["__call__"]
    if call is vars(type)["__call__"]:
        # the constructor type.__call__ delegates to
        return lookup(obj.__mro__, "__new__", "__init__"), True
    # binding loses the owner that overload-only declarations are keyed by
    if isinstance_(getattr(call, "__func__", call), types.FunctionType) and (
        ovs := _get_member_overloads(owner, "__call__")
    ):
        return ovs, not isinstance_(ovs[0], staticmethod)
    if hasattr(type(call), "__get__"):
        call = call.__get__(obj, objtype)
    return _resolve_call(call)


@ft.lru_cache
def _get_signatures(*funcs: tp.Callable, is_bound=False):
    sigs = map(
        ft.partial(
            inspect.signature, annotation_format=annotationlib.Format.FORWARDREF
        ),
        # unwrap __func__ from a classmethod or staticmethod
        (getattr(func, "__func__", func) for func in funcs),
    )
    if is_bound:
        sigs = (sig.replace(parameters=[*sig.parameters.values()][1:]) for sig in sigs)
    return tuple(sigs)


def _get_overloads(obj: tp.Callable, /) -> tuple[inspect.Signature, ...]:
    if not callable(obj):
        raise TypeError
    ovs, is_bound = _resolve_call(obj)
    if ovs:
        sigs = _get_signatures(*ovs, is_bound=is_bound)
    elif isinstance_(obj, (type, types.FunctionType)):
        sigs = _get_signatures(obj)
    else:
        sigs = (
            inspect.signature(obj, annotation_format=annotationlib.Format.FORWARDREF),
        )
    if isinstance_(obj, type):
        # an unannotated or __init__ signature returns Self
        return tuple(
            (
                sig.replace(return_annotation=tp.Self)
                if sig.return_annotation in (None, inspect.Signature.empty)
                else sig
            )
            for sig in sigs
        )
    return sigs


@ft.lru_cache
def _generic_bases(typx: type, /) -> tuple[tuple[_TypeParameter, ...], tuple]:
    """Return the type parameters and generic bases of typx."""
    if (bases := vars(typx).get("__orig_bases__")) is None:
        bases = _stub_bases().get(typx, typx.__bases__)
    if "_fields" in vars(typx) and builtins.tuple in typx.__bases__:
        # a namedtuple's tuple base has one element per field
        annotations = annotationlib.get_annotations(
            typx, format=annotationlib.Format.FORWARDREF
        )
        tuple_base = tuple.__class_getitem__(
            tuple(annotations.get(f, tp.Any) for f in typx._fields)
        )
        bases = tuple(
            tuple_base if b in (builtins.tuple, tp.NamedTuple) else b for b in bases
        )
    for base in bases:
        if tp.get_origin(base) in (tp.Generic, tp.Protocol):
            return tp.get_args(base), bases
    # only subclasses of _GenericAlias have a __parameters__ attribute
    params = (
        p for b in bases if tp.get_origin(b) for p in getattr(b, "__parameters__", ())
    )
    return tuple(dict.fromkeys(params)), bases


def _is_protocol(typx: type, /) -> bool:
    _, bases = _generic_bases(typx)
    return any((tp.get_origin(b) or b) is tp.Protocol for b in bases)


def _default_arg(param: tp.TypeVar, /, *, erased: bool):
    if erased:
        return tp.Any
    if param.has_default():
        return param.__default__
    if param.__constraints__:
        return tp.Union[param.__constraints__]
    return object if param.__bound__ is None else param.__bound__


def _type_args(
    typx, params: tuple[_TypeParameter, ...], /, *, erased: bool
) -> dict[tp.TypeVar, tp.Any] | None:
    """Map the type parameters of typx's origin to the arguments typx passes
    to them, or return None if a parameter is not a TypeVar.
    """
    args = tp.get_args(typx)
    if tp.get_origin(typx) is builtins.tuple and hasattr(typx, "__args__"):
        if (entries := _tuple_entries(typx)) is None:
            return
        # squash tuple type params into a union
        #   tuple[str, int] => (str | int,)
        #   tuple[()] => (Never,)
        args = (tp.Union[*(t for t, _ in entries)] if entries else tp.Never,)
    subst: dict[tp.TypeVar, tp.Any] = {}
    for i, param in enumerate(params):
        if not isinstance_(param, tp.TypeVar):
            return
        arg = args[i] if i < len(args) else _default_arg(param, erased=erased)
        subst[param] = tp.Any if arg is ... else arg
    return subst


def _substitute(typx, subst: tp.Mapping, /):
    if isinstance_(typx, tp.TypeVar):
        return subst.get(typx, typx)
    if tp.get_origin(typx) and (params := getattr(typx, "__parameters__", ())):
        return typx[*(subst.get(p, p) for p in params)]
    return typx


def _substitute_signature(sig: inspect.Signature, subst: tp.Mapping, /):
    return sig.replace(
        parameters=[
            p.replace(annotation=_substitute(p.annotation, subst))
            for p in sig.parameters.values()
        ],
        return_annotation=_substitute(sig.return_annotation, subst),
    )


def _free_params(typx, /) -> tuple[_TypeParameter, ...]:
    if isinstance_(typx, _TypeParameter):
        return (typx,)
    if tp.get_origin(typx):
        return getattr(typx, "__parameters__", ())
    return ()


def _specialize(src, dst: type, /, *, erased: bool):
    """Return the base of src whose origin is dst, with the type arguments of
    src substituted, or None if no generic base of src leads to dst.
    """
    is_protocol = _is_protocol(dst)
    stack: list[tuple[tp.Any, tp.Mapping]] = [(src, {})]
    while stack:
        typx, subst = stack.pop()
        origin = tp.get_origin(typx) or typx
        if origin is dst:
            return _substitute(typx, subst)
        if not (isinstance_(origin, type) and (is_protocol or issubclass(origin, dst))):
            continue
        params, bases = _generic_bases(origin)
        if (args := _type_args(typx, params, erased=erased)) is None:
            continue
        # map type params of origin to args of src
        subst = {p: _substitute(a, subst) for p, a in args.items()}
        stack += [(base, subst) for base in bases[::-1]]


def _map_args(src, dst: type, /, *, erased: bool) -> dict[tp.TypeVar, tp.Any] | None:
    """Map the type parameters of dst to the arguments src passes to them, or
    return None if no generic base of src leads to dst.
    """
    if (base := _specialize(src, dst, erased=erased)) is not None:
        params, _ = _generic_bases(dst)
        return _type_args(base, params, erased=erased)


def _erase(typx: type, /):
    """Return typx with the type arguments of a value of it at runtime."""
    params, _ = _generic_bases(typx)
    return types.GenericAlias(typx, (tp.Any,) * len(params)) if params else typx


def _get_member_overloads(owner: type, name: str, /) -> tp.Sequence[tp.Callable]:
    func = owner.__dict__[name]
    func = getattr(func, "__func__", func)
    if func is tp._overload_dummy:
        # only exists as overloads; no runtime function
        return list(
            tp._overload_registry.get(owner.__module__, {})
            .get(f"{owner.__qualname__}.{name}", {})
            .values()
        )
    return tp.get_overloads(func)


def _member_signatures(
    typx: type, name: str, /
) -> tuple[type, tuple[inspect.Signature, ...]] | None:
    """Return the class in the MRO of typx that declares the method name, with
    the signatures of its overloads bound to an instance, or None if no class
    declares it.
    """
    for owner in typx.__mro__:
        if (members := _stub_overloads().get(owner)) is not None:
            if name in members:
                return owner, _get_signatures(*members[name], is_bound=True)
            continue
        attr = vars(owner).get(name)
        if isinstance_(getattr(attr, "__func__", attr), types.FunctionType):
            ovs = _get_member_overloads(owner, name) or (attr,)
            is_bound = not isinstance_(ovs[0], staticmethod)
            return owner, _get_signatures(*ovs, is_bound=is_bound)


def _protocol_members(typx: type, /) -> set[str]:
    """Return the names of the methods that the protocol typx declares."""
    names = set()
    for base in typx.__mro__:
        if not _is_protocol(base):
            continue
        if (members := _stub_overloads().get(base)) is not None:
            names |= members.keys()
        else:
            names |= {
                name
                for name in base.__protocol_attrs__
                if isinstance_(
                    getattr(attr := vars(base).get(name), "__func__", attr),
                    types.FunctionType,
                )
            }
    return names


def _assign_type(dst: subtype, src, /, **ctx: tp.Unpack[_AssignContext]) -> bool:
    if src == dst.__subtype_form__:
        return True
    match tp.get_origin(src) or src:
        case None:
            return _assign_type(dst, types.NoneType, **ctx)
        case tp.Any | tp.Never | tp.NoReturn:
            return True
        case tp.Union:
            return all(_assign_type(dst, arg, **ctx) for arg in src.__args__)
        case tp.Literal if len(src.__args__) > 1:
            return all(
                _assign_type(dst, tp.Literal[arg], **ctx) for arg in src.__args__
            )
        case tp.Annotated:
            return _assign_type(dst, src.__origin__, **ctx)
        case tp.TypeAliasType() | tp.ForwardRef():
            if "assuming" not in ctx:
                ctx["assuming"] = frozenset()
            # a self-referential alias is its own cyclic lower bound
            if (dst, src) in ctx["assuming"]:
                return True
            ctx["assuming"] |= {(dst, src)}
            return _assign_type(dst, _evaluate(src), **ctx)
        case tp.NewType():
            return _assign_type(dst, src.__supertype__, **ctx)
        case tp.LiteralString:
            return _assign_type(dst, builtins.str, **ctx)
        case tp.Self:
            return _assign_type(dst, ctx.get("self_class", object), **ctx)
        case tp.TypeVar():
            if src.__constraints__:
                return all(_assign_type(dst, c, **ctx) for c in src.__constraints__)
            return _assign_type(dst, src.__bound__ or object, **ctx)
        case builtins.type if tp.get_args(src):
            [arg] = tp.get_args(src)
            if tp.get_origin(arg) is tp.Union:
                return all(_assign_type(dst, type[a], **ctx) for a in arg.__args__)
            if isinstance_(arg, tp.TypeVar):
                if arg.__constraints__:
                    return all(
                        _assign_type(dst, type[c], **ctx) for c in arg.__constraints__
                    )
                return _assign_type(dst, type[arg.__bound__ or object], **ctx)
    return dst.__subtype_subclasscheck__(src, **ctx)


def _assign_with_variance(
    dst, src, flags: int = _AssignFlag.DEFAULT, /, **ctx: tp.Unpack[_AssignContext]
) -> bool:
    if dst == src:
        return True
    if flags & _AssignFlag.CONTRAVARIANT:
        dst, src = src, dst
    return _assign_type(subtype(dst), src, **ctx) and (
        not flags & _AssignFlag.INVARIANT or _assign_type(subtype(src), dst, **ctx)
    )


def _declared_types(typx: type, /) -> list[tuple[tp.Any, _AssignFlag]]:
    """Return the types declared in the members and generic bases of typx,
    each with the variance its assignment requires.
    """
    out = []
    annotations = annotationlib.get_annotations(
        typx, format=annotationlib.Format.FORWARDREF
    )
    for name, form in annotations.items():
        final = False
        while tp.get_origin(form) in (tp.ClassVar, tp.Final):
            final |= tp.get_origin(form) is tp.Final
            [form] = tp.get_args(form)
        dunder = name.startswith("__") and name.endswith("__")
        if final or (name.startswith("_") and not dunder):
            out.append((form, _AssignFlag.DEFAULT))
        else:
            out.append((form, _AssignFlag.INVARIANT))
    for name, attr in vars(typx).items():
        if name in ("__new__", "__init__"):
            continue
        if isinstance_(attr, property):
            funcs = (attr.fget, attr.fset, attr.fdel)
        else:
            funcs = (attr,)
        for func in funcs:
            func = getattr(func, "__func__", func)
            if not isinstance_(func, types.FunctionType):
                continue
            if isinstance_(attr, property):
                ovs = tp.get_overloads(func) or (func,)
            else:
                ovs = _get_member_overloads(typx, name) or (attr,)
            is_bound = not isinstance_(ovs[0], staticmethod)
            for sig in _get_signatures(*ovs, is_bound=is_bound):
                out += [
                    (p.annotation, _AssignFlag.CONTRAVARIANT)
                    for p in sig.parameters.values()
                ]
                out.append((sig.return_annotation, _AssignFlag.DEFAULT))
    _, bases = _generic_bases(typx)
    out += [
        (base, _AssignFlag.DEFAULT)
        for base in bases
        if tp.get_origin(base) not in (tp.Generic, tp.Protocol, builtins.tuple)
    ]
    return out


def _assign_class_to_self(
    dst: _SupportsGeneric, src: _SupportsGeneric, /, **ctx: tp.Unpack[_AssignContext]
) -> bool:
    params, _ = _generic_bases(dst.__origin__)
    dst_subst, src_subst = (dict(zip(params, x.__args__)) for x in (dst, src))
    return all(
        _assign_with_variance(
            _substitute(form, dst_subst), _substitute(form, src_subst), flags, **ctx
        )
        for form, flags in _declared_types(dst.__origin__)
    )


class _VarianceDummy:
    pass


def _infer_variance(
    typx, /, **ctx: tp.Unpack[_AssignContext]
) -> dict[tp.TypeVar, _AssignFlag]:
    assumed_variance = ctx.setdefault("assumed_variance", {})
    params, _ = _generic_bases(typx)
    out = {}
    for i, param in enumerate(params):
        if not (isinstance_(param, tp.TypeVar) and param.__infer_variance__):
            continue
        bounds: list[type[tp.Any]] = [object, types.new_class(param.__name__)]
        bound_args = [[_VarianceDummy] * len(params) for _ in range(len(bounds))]
        for x, args in zip(bounds, bound_args):
            args[i] = x
        upper, lower = (typx[*b] for b in bound_args)
        if _assign_class_to_self(
            upper,
            lower,
            assumed_variance={**assumed_variance, typx: _AssignFlag.DEFAULT},
            self_class=typx,
        ):
            out[param] = _AssignFlag.DEFAULT
        elif _assign_class_to_self(
            lower,
            upper,
            assumed_variance={**assumed_variance, typx: _AssignFlag.CONTRAVARIANT},
            self_class=typx,
        ):
            out[param] = _AssignFlag.CONTRAVARIANT
        else:
            out[param] = _AssignFlag.INVARIANT
    return out


def _variance(
    typx: type, param: tp.TypeVar, /, **ctx: tp.Unpack[_AssignContext]
) -> _AssignFlag:
    assumed_variance = ctx.setdefault("assumed_variance", {})
    if (assumed := assumed_variance.get(typx)) is not None:
        return assumed
    if param.__covariant__:
        return _AssignFlag.DEFAULT
    if param.__contravariant__:
        return _AssignFlag.CONTRAVARIANT
    if not param.__infer_variance__:
        return _AssignFlag.INVARIANT
    if (variances := _inferred_variance.get(typx)) is None:
        variances = _inferred_variance[typx] = _infer_variance(typx, **ctx)
    return variances[param]


class _literal_instancecheck(_checker):
    def __call__(self, obj, /, **__):
        return any(type(obj) is type(a) and obj == a for a in self.__args__)


class _literal_subclasscheck(_literal_instancecheck):
    def __call__(self, obj, /, **ctx):
        return tp.get_origin(obj) is tp.Literal and all(
            map(ft.partial(super().__call__, **ctx), tp.get_args(obj))
        )


class _nominal_instancecheck(_checker):
    def __init__(self, typx: type, /):
        self.__args__ = (typx,)

    def __call__(self, obj, /, **__):
        return isinstance_(obj, self.__args__)


class _nominal_subclasscheck(_nominal_instancecheck):
    def __call__(self, obj, /, **ctx):
        if tp.get_origin(obj) is tp.Literal:
            return all(map(ft.partial(super().__call__, **ctx), tp.get_args(obj)))
        return issubclass(tp.get_origin(obj) or obj, self.__args__)


def _assign_tuple(dst, src, /, **ctx: tp.Unpack[_AssignContext]) -> bool:
    origin = tp.get_origin(src) or src
    if not (isinstance_(origin, type) and issubclass(origin, builtins.tuple)):
        return False
    base = _specialize(src, builtins.tuple, erased=False)
    if (
        base is None
        or (src_entries := _tuple_entries(base)) is None
        or (dst_entries := _tuple_entries(dst)) is None
    ):
        return True
    if not _adjust_tuple_entries(dst_entries, src_entries):
        return False
    return all(
        _assign_with_variance(d, s, **ctx)
        for (d, _), (s, _) in zip(dst_entries, src_entries)
    )


class _tuple_instancecheck(_checker):
    def __init__(self, typx: type[tuple], /):
        self.__form__ = typx

    def __call__(self, obj, /, **ctx):
        if not isinstance_(obj, builtins.tuple):
            return False
        if (entries := _tuple_entries(self.__form__)) is None:
            return True
        [*items] = zip(*entries)
        if not items:
            return not obj
        [[*ts], det] = items
        i = next((i for i, indeterminate in enumerate(det) if indeterminate), None)
        if i is not None:
            ts[i : i + 1] *= len(obj) - len(entries) + 1
        return len(ts) == len(obj) and all(
            subtype(t).__subtype_instancecheck__(v, **ctx) for t, v in zip(ts, obj)
        )


class _tuple_subclasscheck(_tuple_instancecheck):
    def __call__(self, obj, /, **ctx):
        return _assign_tuple(self.__form__, obj, **ctx)


def _callable_signature(typx: _SupportsGeneric, /) -> inspect.Signature:
    params, ret = tp.get_args(typx)
    if tp.get_origin(params) is tp.Concatenate:
        *params, tail = tp.get_args(params)
    elif isinstance_(params, list):
        tail = None
    else:
        params, tail = [], params
    sig = [
        inspect.Parameter(f"_{i}", inspect.Parameter.POSITIONAL_ONLY, annotation=t)
        for i, t in enumerate(params)
    ]
    if tail is ...:
        sig += [
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ]
    elif isinstance_(tail, tp.ParamSpec):
        sig += [
            inspect.Parameter(
                "args", inspect.Parameter.VAR_POSITIONAL, annotation=tail.args
            ),
            inspect.Parameter(
                "kwargs", inspect.Parameter.VAR_KEYWORD, annotation=tail.kwargs
            ),
        ]
    return inspect.Signature(sig, return_annotation=ret)


def _assign_signature(
    dst: inspect.Signature, src: inspect.Signature, /, **ctx: tp.Unpack[_AssignContext]
) -> bool:
    """Return whether src is assignable to dst, the signature of a Callable form."""

    def annotation(typx, /):
        return tp.Any if typx is inspect.Parameter.empty else typx

    shared = {
        p
        for typx in (
            dst.return_annotation,
            *(q.annotation for q in dst.parameters.values()),
        )
        for p in _free_params(typx)
    }
    solution: dict[_TypeParameter, tp.Any] = {}
    src_params = [*src.parameters.values()]
    for dst_param in dst.parameters.values():
        if dst_param.kind is not inspect.Parameter.POSITIONAL_ONLY:
            src_params = []
            break
        if not src_params or src_params[0].kind > inspect.Parameter.VAR_POSITIONAL:
            return False
        if src_params[0].kind is inspect.Parameter.VAR_POSITIONAL:
            src_param = src_params[0]
        else:
            src_param = src_params.pop(0)
        src_annotation = annotation(src_param.annotation)
        if not shared.intersection(_free_params(src_annotation)):
            _solve(src_annotation, dst_param.annotation, solution)
        if not _assign_with_variance(
            dst_param.annotation,
            _substitute(src_annotation, solution),
            _AssignFlag.CONTRAVARIANT,
            **ctx,
        ):
            return False
    if not all(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        or p.default is not inspect.Parameter.empty
        for p in src_params
    ):
        return False
    if not all(
        _assign_type(subtype(p), t, **ctx)
        for p, t in solution.items()
        if isinstance_(p, tp.TypeVar)
    ):
        return False
    return _assign_with_variance(
        dst.return_annotation,
        _substitute(annotation(src.return_annotation), solution),
        **ctx,
    )


def _specialize_member(
    typx, name: str, /, *, erased=False
) -> tuple[inspect.Signature, ...] | None:
    """Return the signatures of the method name of typx, with the type
    arguments of typx substituted, or None if no class in its MRO declares it.
    """
    if (member := _member_signatures(tp.get_origin(typx) or typx, name)) is None:
        return
    owner, sigs = member
    subst = _map_args(typx, owner, erased=erased) or {}
    return tuple(_substitute_signature(s, subst) for s in sigs)


def _type_call_signatures(typx, /) -> tuple[inspect.Signature, ...] | None:
    """Return the signatures of calling a value of typx, or None if a value of
    typx is not callable.
    """
    origin = tp.get_origin(typx) or typx
    if origin is collections.abc.Callable and tp.get_args(typx):
        return (_callable_signature(typx),)
    if origin is builtins.type and tp.get_args(typx):
        # a value of type[X] calls the constructor of X
        [arg] = tp.get_args(typx)
        return _get_overloads(tp.get_origin(arg) or arg)
    if not (isinstance_(origin, type) and issubclass(origin, collections.abc.Callable)):
        return
    if not tp.get_args(typx) and (params := _generic_bases(origin)[0]):
        typx = types.GenericAlias(
            origin, tuple(p.__default__ if p.has_default() else p for p in params)
        )
    if (sigs := _specialize_member(typx, "__call__")) is not None:
        return sigs
    call = inspect.getattr_static(origin, "__call__")
    ovs, _ = _resolve_call(call)
    return _get_signatures(*(ovs or (call,)), is_bound=True)


def _self_class(src, /, *, erased=False):
    """Return the type that Self binds to in the members of src."""
    if (tp.get_origin(src) or src) is builtins.type and tp.get_args(src):
        # a value of type[X] is the class X
        [arg] = tp.get_args(src)
        return arg
    return _erase(src) if erased else src


class _callable_instancecheck(_checker):
    def __init__(self, typx: _SupportsGeneric, /):
        self.signatures = _type_call_signatures(typx) or ()

    def __call__(self, obj, /, **ctx):
        if not callable(obj):
            return False
        # bind Self to the self class of the callable
        owner = getattr(obj, "__self__", obj)
        ctx["self_class"] = owner if isinstance_(owner, type) else type(owner)
        sigs = _get_overloads(obj)
        return all(
            any(_assign_signature(d, s, **ctx) for s in sigs) for d in self.signatures
        )


class _callable_subclasscheck(_callable_instancecheck):
    def __call__(self, obj, /, **ctx):
        if (sigs := _type_call_signatures(obj)) is None:
            return False
        ctx["self_class"] = _self_class(obj)
        return all(
            any(_assign_signature(d, s, **ctx) for s in sigs) for d in self.signatures
        )


@ft.lru_cache
def _member_pairs(dst, src, /, *, erased=False) -> _MemberPairs | None:
    """Pair the signatures of each method of the protocol dst with those of the
    same method of src, with the type arguments of both substituted, or return
    None if src does not declare one of them.
    """
    origin = tp.get_origin(src) or src
    pairs = []
    for name in _protocol_members(tp.get_origin(dst)):
        dst_sigs = _specialize_member(dst, name)
        if name == "__call__":
            # Callable[...] and type[X] are called through their signature
            # and constructor, not through a __call__ member
            src_sigs = _type_call_signatures(_erase(src) if erased else src)
        else:
            src_sigs = _specialize_member(src, name, erased=erased)
        if not isinstance_(origin, type) or dst_sigs is None or src_sigs is None:
            return
        pairs.append((dst_sigs, src_sigs))
    return tuple(pairs)


def _probe(check: tp.Callable[[tp.Any, type], bool], obj, dst: type, /) -> bool:
    """Return False if dst is a runtime-checkable protocol that obj does not
    satisfy, which the runtime decides by member names alone.
    """
    if not _is_protocol(dst):
        return True
    try:
        return check(obj, dst)
    except TypeError:
        return True


@ft.lru_cache
def _issubclass(src, dst: type, /) -> bool:
    try:
        return issubclass(tp.get_origin(src) or src, dst)
    except TypeError:
        # dst is a protocol that is not runtime-checkable
        return False


def _assign_protocol(
    dst: _SupportsGeneric,
    src,
    pairs: _MemberPairs,
    /,
    *,
    erased=False,
    **ctx: tp.Unpack[_AssignContext],
) -> bool:
    if "assuming" not in ctx:
        ctx["assuming"] = frozenset()
    # a self-referential member is its own cyclic lower bound.
    # e.g., the stub for Iterator.__iter__ returns itself:
    #
    #   __iter__(self) -> Iterator[_T_co]: ...
    #
    if (item := (subtype(dst), src)) in ctx["assuming"]:
        return True
    ctx["assuming"] |= {item}
    ctx["self_class"] = _self_class(src, erased=erased)
    return all(
        any(_assign_signature(d, s, **ctx) for s in src_sigs)
        for dst_sigs, src_sigs in pairs
        for d in dst_sigs
    )


def _assign_generic(
    dst: _SupportsGeneric, src, /, *, erased=False, **ctx: tp.Unpack[_AssignContext]
):
    if not _probe(issubclass, tp.get_origin(src) or src, dst.__origin__):
        return False
    src_args = _map_args(src, dst.__origin__, erased=erased)
    if (
        src_args is None
        and _is_protocol(dst.__origin__)
        and (pairs := _member_pairs(dst, src, erased=erased)) is not None
    ):
        return _assign_protocol(dst, src, pairs, erased=erased, **ctx)
    if not _issubclass(src, dst.__origin__):
        return False
    dst_args = _map_args(dst, dst.__origin__, erased=False)
    if src_args is None or dst_args is None:
        return True
    return all(
        _assign_with_variance(
            dst_arg, src_args[p], _variance(dst.__origin__, p, **ctx), **ctx
        )
        for p, dst_arg in dst_args.items()
    )


def _solve_signature(
    dst: inspect.Signature,
    src: tp.Iterable[inspect.Signature],
    solution: dict[_TypeParameter, tp.Any],
    /,
    *,
    self_class: tp.Any = object,
    assuming: frozenset = frozenset(),
) -> bool:
    """Bind the type parameters in the return type of dst to the return type of
    the first signature in src whose parameters accept those of dst.
    """
    # the parameters of dst select the signature but bind nothing
    unbound = {
        p: tp.Any for q in dst.parameters.values() for p in _free_params(q.annotation)
    }
    selector = _substitute_signature(dst, unbound).replace(return_annotation=tp.Any)
    for sig in src:
        if _assign_signature(selector, sig, self_class=self_class):
            ret = sig.return_annotation
            if ret is inspect.Signature.empty:
                ret = tp.Any
            elif ret is tp.Self:
                ret = self_class
            return _solve(dst.return_annotation, ret, solution, assuming=assuming)
    return False


def _solve(
    dst,
    src,
    solution: dict[_TypeParameter, tp.Any],
    /,
    *,
    erased=False,
    assuming: frozenset = frozenset(),
) -> bool:
    """Bind the type parameters in dst to the types src passes to them, and
    return whether src is assignable to dst.
    """
    if isinstance_(dst, tp.TypeVar):
        solution[dst] = tp.Union[solution[dst], src] if dst in solution else src
        return True
    if not (params := _free_params(dst)):
        return _assign_type(subtype(dst), src, self_class=src)
    if src is tp.Any:
        for p in params:
            solution.setdefault(p, tp.Any)
        return True
    origin = tp.get_origin(dst)
    if origin is tp.Union:
        fixed = [a for a in dst.__args__ if not _free_params(a)]
        # a bare TypeVar accepts any member of src, so it is tried last
        free = sorted(
            (a for a in dst.__args__ if _free_params(a)),
            key=lambda a: isinstance_(a, tp.TypeVar),
        )
        for s in src.__args__ if tp.get_origin(src) is tp.Union else (src,):
            if any(_assign_type(subtype(f), s) for f in fixed):
                continue
            # a member of dst selects s by its erasure, which binds nothing
            member = next(
                (
                    a
                    for a in free
                    if _assign_type(
                        subtype(_substitute(a, dict.fromkeys(_free_params(a), tp.Any))),
                        s,
                    )
                ),
                None,
            )
            if member is None or not _solve(member, s, solution, assuming=assuming):
                return False
        return True
    if tp.get_origin(src) is tp.Union:
        return all(_solve(dst, s, solution, assuming=assuming) for s in src.__args__)
    if not isinstance_(origin, type):
        return False
    # a self-referential member is its own cyclic lower bound.
    if (dst, src) in assuming:
        return True
    assuming |= {(dst, src)}
    if (base := _specialize(src, origin, erased=erased)) is not None:
        params, _ = _generic_bases(origin)
        src_args = _type_args(base, params, erased=erased)
        dst_args = _type_args(dst, params, erased=False)
        if src_args is None or dst_args is None:
            return True
        return all(
            _solve(a, src_args[p], solution, assuming=assuming)
            for p, a in dst_args.items()
        )
    if not _is_protocol(origin) or not _probe(
        issubclass, tp.get_origin(src) or src, origin
    ):
        return False
    if (pairs := _member_pairs(dst, src, erased=erased)) is None:
        return _issubclass(src, origin)
    self_class = _self_class(src, erased=erased)
    return all(
        _solve_signature(
            d, src_sigs, solution, self_class=self_class, assuming=assuming
        )
        for dst_sigs, src_sigs in pairs
        for d in dst_sigs
    )


def _solve_value(dst, obj, solution: dict[_TypeParameter, tp.Any], /) -> bool:
    """Bind the type parameters in dst to the types obj passes to them, and
    return whether obj is an instance of dst.
    """
    if not _free_params(dst):
        return isinstance_(obj, subtype(dst))
    if (tp.get_origin(dst) or dst) is collections.abc.Callable:
        if not callable(obj):
            return False
        try:
            sigs = _get_overloads(obj)
        except ValueError:
            return True
        owner = getattr(obj, "__self__", obj)
        return _solve_signature(
            _callable_signature(dst),
            sigs,
            solution,
            self_class=owner if isinstance_(owner, type) else type(owner),
        )
    if (src := _infer_type(obj)) is not None:
        ok, bound = _solve_cached(dst, src, frozenset(solution.items()))
    else:
        ok, bound = _solve_cached(
            dst, type(obj), frozenset(solution.items()), erased=True
        )
        params, _ = _generic_bases(type(obj))
        if (
            ok
            and params
            and isinstance_(obj, collections.abc.Collection)
            and (args := _map_args(dst, collections.abc.Iterable, erased=False))
        ):
            # dst is Iterable and obj is a Collection, so can be walked
            # to obtain runtime types without exhausting the iterable.
            [item] = args.values()
            bound = frozenset(solution.items())
            for t in set(map(type, obj)):
                ok, bound = _solve_cached(item, t, bound, erased=True)
                if not ok:
                    return False
    if ok:
        solution.update(bound)
    return ok


@ft.lru_cache
def _solve_cached(dst, src, bound: frozenset, /, *, erased=False):
    """Return the result of _solve and the solution it extends bound to."""
    solution = dict(bound)
    res = _solve(dst, src, solution, erased=erased)
    return res, frozenset(solution.items())


def _infer_type(obj, /):
    """Return the type of obj with the type arguments it was created with, or
    None if they are unknown.
    """
    if (src := getattr(obj, "__dict__", {}).get("__orig_class__")) is not None:
        return src
    try:
        reduced = obj.__reduce__()
    except TypeError:
        return
    # obj is equivalent to the result of calling func with args
    match reduced:
        case (func, tuple(args), *_) if callable(func):
            pass
        case _:
            return
    try:
        sigs = _get_overloads(func)
    except ValueError:
        return
    for sig in sigs:
        try:
            bound = sig.bind(*args)
        except TypeError:
            continue
        solution = {}
        params = ((sig.parameters[k], v) for k, v in bound.arguments.items())
        if not all(
            _solve_value(p.annotation, v, solution)
            for p, value in params
            if p.annotation is not p.empty
            for v in (value if p.kind is p.VAR_POSITIONAL else (value,))
        ):
            continue
        ret = sig.return_annotation
        if ret is sig.empty:
            return
        if ret is tp.Self:
            if not isinstance_(func, type):
                return
            type_params, _ = _generic_bases(func)
            ret = types.GenericAlias(func, type_params) if type_params else func
        return _substitute(ret, {p: solution.get(p, tp.Any) for p in _free_params(ret)})


class _generic_instancecheck(_checker):
    def __init__(self, typx: _SupportsGeneric, /):
        self.__form__ = typx
        self.call = (
            _callable_instancecheck(typx)
            if "__call__" in _protocol_members(typx.__origin__)
            else None
        )

    def __call__(self, obj, /, **ctx):
        if not _probe(isinstance_, obj, self.__form__.__origin__):
            return False
        # _assign_generic only sees type(obj), whose __call__ is type.__call__
        # for a class and FunctionType.__call__ for a function
        if not (self.call is None or self.call(obj, **ctx)):
            return False
        if (src := _infer_type(obj)) is not None:
            return _assign_generic(self.__form__, src, **ctx)
        if not _assign_generic(self.__form__, type(obj), erased=True, **ctx):
            return False
        getitem = _specialize_member(self.__form__, "__getitem__") or ()
        if getitem and callable(getattr(obj, "keys", None)):
            kts = [
                a
                for s in _specialize_member(self.__form__, "keys") or ()
                for a in (
                    _map_args(
                        s.return_annotation, collections.abc.Iterable, erased=False
                    )
                    or {}
                ).values()
            ]
            kt = tp.Union[*kts] if kts else object
            vt = tp.Union[
                *(
                    (
                        tp.Any
                        if s.return_annotation is inspect.Signature.empty
                        else s.return_annotation
                    )
                    for s in getitem
                )
            ]
            d = obj if isinstance_(obj, collections.abc.Mapping) else {**obj}
            return all(
                subtype(collections.abc.Collection[t]).__subtype_instancecheck__(
                    xs, **ctx
                )
                for t, xs in zip((kt, vt), (d.keys(), d.values()))
            )
        if rets := [
            (
                tp.Any
                if s.return_annotation is inspect.Signature.empty
                else s.return_annotation
            )
            for s in getitem
            if (p := next(iter(s.parameters.values()), None)) is not None
            and (
                p.annotation is inspect.Parameter.empty
                or _assign_type(subtype(p.annotation), int)
            )
        ]:
            item = subtype(tp.Union[*rets])
        elif isinstance_(obj, collections.abc.Collection) and (
            args := _map_args(self.__form__, collections.abc.Iterable, erased=False)
        ):
            [item] = map(subtype, args.values())
        else:
            return True
        ts = set(map(type, obj))
        nested = {
            t
            for t in ts
            if _issubclass(t, collections.abc.Collection) and _generic_bases(t)[0]
        }
        # the check of each nested item covers its type
        if not all(_assign_type(item, _erase(t), **ctx) for t in ts - nested):
            return False
        recursion_count = ctx.setdefault("recursion_count", 0)
        if (seen := ctx.get("seen")) is None:
            seen = ctx["seen"] = set()
        key = (id(obj), self.__form__)
        # a cycle has occurred, and is assignable to itself.
        if not nested or recursion_count > _RECURSION_LIMIT or key in seen:
            return True
        seen.add(key)
        try:
            return all(
                item.__subtype_instancecheck__(
                    x,
                    **(ctx | {"self_class": t, "recursion_count": recursion_count + 1}),
                )
                for x in obj
                if (t := type(x)) in nested
            )
        finally:
            seen.discard(key)


class _generic_subclasscheck(_generic_instancecheck):
    def __call__(self, obj, /, **ctx):
        if tp.get_origin(obj) is tp.Literal:
            return all(map(ft.partial(super().__call__, **ctx), tp.get_args(obj)))
        return _assign_generic(self.__form__, obj, **ctx)


class _type_instancecheck(_checker):
    def __init__(self, typx: _SupportsGeneric, /):
        [arg] = typx.__args__
        self.__form__ = subtype(arg)

    def __call__(self, obj, /, **ctx):
        return isinstance_(obj, type) and _assign_type(self.__form__, obj, **ctx)


class _type_subclasscheck(_type_instancecheck):
    def __call__(self, obj, /, **ctx):
        origin = tp.get_origin(obj) or obj
        if origin is builtins.type:
            [arg] = tp.get_args(obj) or [tp.Any]
        elif isinstance_(origin, type) and issubclass(origin, builtins.type):
            arg = object
        else:
            return False
        return _assign_type(self.__form__, arg, **ctx)


class _typevar_instancecheck(_checker):
    def __init__(self, typx: tp.TypeVar, /):
        self.__form__ = typx

    def bounds(self) -> tuple[tp.Any, ...]:
        return self.__form__.__constraints__ or (self.__form__.__bound__ or object,)

    def __call__(self, obj, /, **ctx):
        return any(
            subtype(t).__subtype_instancecheck__(obj, **ctx) for t in self.bounds()
        )


class _typevar_subclasscheck(_typevar_instancecheck):
    def __call__(self, obj, /, **ctx):
        return any(_assign_type(subtype(t), obj, **ctx) for t in self.bounds())


_subtypes = weakref.WeakValueDictionary()
_inferred_variance = weakref.WeakKeyDictionary()


class subtype(type):
    __subtype_form__: tp.Any
    __subtype_instancecheck__: _checker
    __subtype_subclasscheck__: _checker

    @tp.no_type_check
    @staticmethod
    def __new__(mcls, typx, /):
        if isinstance_(typx, mcls):
            if type(typx) is mcls:
                return typx
            typx = typx.__subtype_form__
        # ParamSpecArgs and ParamSpecKwargs are not hashable
        if not callable(getattr(type(typx), "__hash__", None)):
            key = type(typx), tp.get_origin(typx)
        else:
            key = typx
        if (cls := _subtypes.get(key)) is not None:
            return cls

        i_check = "__subtype_instancecheck__"
        s_check = "__subtype_subclasscheck__"
        namespace = {"__subtype_form__": typx}
        match tp.get_origin(typx) or typx:
            case None:
                return mcls(types.NoneType)
            case tp.Annotated:
                return mcls(typx.__origin__)
            case tp.Any:
                namespace.update(
                    dict.fromkeys((i_check, s_check), lambda _, **__: True)
                )
            case tp.Never | tp.NoReturn:
                namespace.update(
                    dict.fromkeys((i_check, s_check), lambda _, **__: False)
                )
            case tp.Literal:
                namespace[i_check] = _literal_instancecheck(typx)
                namespace[s_check] = _literal_subclasscheck(typx)
            case tp.Union:
                args = tuple(getattr(a, "__subtype_form__", a) for a in typx.__args__)
                if args != typx.__args__:
                    return mcls(tp.Union[args])
                members = tuple(map(mcls, typx.__args__))
                namespace[i_check] = lambda obj, **ctx: any(
                    m.__subtype_instancecheck__(obj, **ctx) for m in members
                )
                namespace[s_check] = lambda src, **ctx: any(
                    _assign_type(m, src, **ctx) for m in members
                )
            case tp.TypeAliasType() | tp.ForwardRef():
                # resolve on first check, so recursive references terminate
                resolved = ft.cache(lambda: mcls(_evaluate(typx)))
                namespace[i_check] = lambda obj, **ctx: (
                    resolved().__subtype_instancecheck__(obj, **ctx)
                )
                namespace[s_check] = lambda src, **ctx: _assign_type(
                    resolved(), src, **ctx
                )
            case tp.Self:  # type: ignore
                namespace[i_check] = lambda obj, **ctx: mcls(
                    ctx.get("self_class", tp.Any)
                ).__subtype_instancecheck__(obj, **ctx)
                namespace[s_check] = lambda src, **ctx: _assign_type(
                    mcls(ctx.get("self_class", tp.Any)), src, **ctx
                )
            case tp.NewType():
                supertype = mcls(typx.__supertype__)
                namespace[i_check] = lambda obj, **ctx: (
                    supertype.__subtype_instancecheck__(obj, **ctx)
                )
                # _assign_type admits typx and the NewTypes derived from it
                namespace[s_check] = lambda _, **__: False
            # tuple[()] has no arguments but is not the bare tuple class
            case builtins.tuple if hasattr(typx, "__args__"):
                namespace[i_check] = _tuple_instancecheck(typx)
                namespace[s_check] = _tuple_subclasscheck(typx)
            case type() as origin if _is_protocol(origin) and not tp.get_args(typx):
                alias = types.GenericAlias(origin, ())
                namespace[i_check] = _generic_instancecheck(alias)
                namespace[s_check] = _generic_subclasscheck(alias)
            case type() if not tp.get_args(typx):
                namespace[i_check] = _nominal_instancecheck(typx)
                namespace[s_check] = _nominal_subclasscheck(typx)
            case collections.abc.Callable:
                namespace[i_check] = _callable_instancecheck(typx)
                namespace[s_check] = _callable_subclasscheck(typx)
            case builtins.type:
                namespace[i_check] = _type_instancecheck(typx)
                namespace[s_check] = _type_subclasscheck(typx)
            case type():
                namespace[i_check] = _generic_instancecheck(typx)
                namespace[s_check] = _generic_subclasscheck(typx)
            case tp.TypeVar():
                namespace[i_check] = _typevar_instancecheck(typx)
                namespace[s_check] = _typevar_subclasscheck(typx)
            # the origin of P.args / P.kwargs is a ParamSpec reference.
            # key is tuple[type[_], P] here, since typx is not hashable.
            case tp.ParamSpec() if key is not typx:
                namespace.update(
                    dict.fromkeys((i_check, s_check), lambda _, **__: True)
                )
            # TODO: implement these
            case str() | tp.LiteralString | tp.TypeGuard | tp.TypeIs:
                raise NotImplementedError
            case _:
                raise TypeError
        return _subtypes.setdefault(
            key, super().__new__(mcls, mcls.__name__, (), namespace)
        )

    def __repr__(cls):
        return "{0.__name__}({0.__subtype_form__})".format(cls)

    def __instancecheck__(cls, obj, /):
        return cls.__subtype_instancecheck__(
            obj,
            self_class=getattr(obj, "__dict__", {}).get("__orig_class__", type(obj)),
        )

    def __subclasscheck__(cls, src, /):
        return _assign_type(cls, src, self_class=src)

    @tp.final
    def __call__(cls) -> tp.Never:
        raise TypeError(f"cannot instantiate {type(cls).__qualname__} objects")

    def __or__(cls, other, /):
        if isinstance_(other, mcls := type(cls)):
            return mcls(cls.__subtype_form__ | other.__subtype_form__)
        try:
            return mcls(cls.__subtype_form__ | other)
        except NotImplementedError, TypeError:
            return NotImplemented

    def __ror__(cls, other, /):
        try:
            return cls.__class__(other | cls.__subtype_form__)
        except NotImplementedError, TypeError:
            return NotImplemented


def isinstance(obj: object, typx: tp.Any, /) -> bool:
    return isinstance_(obj, subtype(typx))
