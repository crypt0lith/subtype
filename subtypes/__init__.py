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
        # a bare tuple takes the bound of its type parameter
        return [(object, True)]
    if _is_indeterminate_tuple(typx):
        return [(tp.get_args(typx)[0], True)]
    entries = []
    for arg in tp.get_args(typx):
        for x in _unpack_tuple_items(arg):
            if isinstance(x, tp.TypeVarTuple):
                return
            if _is_unpacked(x) and _is_indeterminate_tuple(x):
                entries.append((tp.get_args(x)[0], True))
            else:
                entries.append((x, False))
    return entries


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
        elif not hasattr(origin, "__class_getitem__"):
            return origin
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


class _Variance(enum.Enum):
    INVARIANT = enum.auto()
    COVARIANT = enum.auto()
    CONTRAVARIANT = enum.auto()


class _AssignContext(tp.TypedDict):
    assuming: frozenset[tuple[subtype, tp.Any]]
    assumed_variance: tp.Mapping[type, _Variance]
    self_class: tp.Any


class _checker(abc.ABC):
    def __init__(self, typx: _SupportsGeneric, /):
        self.__args__ = typx.__args__

    @abc.abstractmethod
    def __call__(self, obj: tp.Any, /, **ctx: tp.Unpack[_AssignContext]) -> bool: ...


class _literal_instancecheck(_checker):
    def __call__(self, obj, /, **ctx):
        # Literal[1] excludes True
        return any(type(obj) is type(a) and obj == a for a in self.__args__)


class _literal_subclasscheck(_literal_instancecheck):
    def __call__(self, obj, /, **ctx):
        return tp.get_origin(obj) is tp.Literal and all(
            map(ft.partial(super().__call__, **ctx), tp.get_args(obj))
        )


_PROMOTIONS = {float: (float, int), complex: (complex, float, int)}


class _nominal_instancecheck(_checker):
    def __init__(self, typx: type, /):
        self.__args__ = _PROMOTIONS.get(typx, (typx,))

    def __call__(self, obj, /, **ctx):
        return isinstance(obj, self.__args__)


class _nominal_subclasscheck(_nominal_instancecheck):
    def __call__(self, obj, /, **ctx):
        if tp.get_origin(obj) is tp.Literal:
            return all(map(ft.partial(super().__call__, **ctx), tp.get_args(obj)))
        return issubclass(tp.get_origin(obj) or obj, self.__args__)


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
            isinstance(node.value, ast.Name)
            and node.value.id in self._ns
            and not hasattr(self._ns[node.value.id], "__class_getitem__")
        ):
            # emit only the subscripted name
            return node.value
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
        if isinstance(forwardref.__cell__, dict):
            for cell_name, cell in forwardref.__cell__.items():
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    pass
                else:
                    locals.setdefault(cell_name, cell_value)
        if forwardref.__extra_names__:
            locals.update(forwardref.__extra_names__)
        if (node := forwardref.__ast_node__) is None:
            node = ast.parse(forwardref.__forward_arg__, mode="eval").body
        tree = ast.fix_missing_locations(
            cls(globals=globals, locals=locals).visit(ast.Expression(node))
        )
        code = compile(tree, "<ast>", "eval")
        return eval(code, globals, locals)


def _evaluate(typx: tp.TypeAliasType | tp.ForwardRef, /):
    if isinstance(typx, tp.ForwardRef):
        return _ForwardRefSubscriptTransformer.evaluate_forwardref(typx)
    if tp.get_args(typx):
        return _make_typevars_bound(typx)
    return typx.__value__


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
                if isinstance(s, ast.If):
                    s.body = self._body(s.body, depth)
                    s.orelse = self._body(s.orelse, depth) if s.orelse else []
                    out.append(s)
                elif isinstance(s, ast.ClassDef):
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
                                isinstance(d, ast.Name) and d.id == "type_check_only"
                            )
                        ]
                        out.append(s)
                elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if depth or s.name in self._runtime:
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
        return owner if isinstance(owner, type) else None

    STUBS = files(__name__) / "_stubs"
    out = {}
    bases = {}
    with as_file(STUBS / "_typeshed" / "__init__.pyi") as path:
        spec = spec_from_loader(
            "_typeshed", PyiLoader("_typeshed", str(path)), is_package=True
        )
        assert spec is not None
        assert spec.loader is not None
        sys.modules["_typeshed"] = mod = module_from_spec(spec)
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
            if isinstance(func := getattr(attr, "__func__", attr), types.FunctionType):
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
                attrs = (vars(typx)[name] for name in names if name in vars(typx))
                decls = [
                    (
                        tp.get_overloads(a)
                        if isinstance(getattr(a, "__func__", a), types.FunctionType)
                        else ()
                    )
                    for a in attrs
                ]
            if decls:
                return max(decls, key=len)
        return ()

    objtype = type(obj)
    if isinstance(obj, types.FunctionType) or hasattr(objtype, "__func__"):
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
                if isinstance(m_self, type):
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
    call = next(
        vars(typx)["__call__"] for typx in objtype.__mro__ if "__call__" in vars(typx)
    )
    if call is vars(type)["__call__"]:
        # the constructor type.__call__ delegates to
        return lookup(obj.__mro__, "__new__", "__init__"), True
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
    else:
        sigs = (
            inspect.signature(obj, annotation_format=annotationlib.Format.FORWARDREF),
        )
    if isinstance(obj, type):
        # an unannotated or __init__ signature returns an instance of the class
        return tuple(
            (
                sig.replace(return_annotation=tp.Self)
                if sig.return_annotation in (None, inspect.Signature.empty)
                else sig
            )
            for sig in sigs
        )
    return sigs


def _default_arg(param: tp.TypeVar, /, *, erased: bool):
    # a value's type arguments are erased at runtime
    if erased:
        return tp.Any
    if param.has_default():
        return param.__default__
    if param.__constraints__:
        return tp.Union[param.__constraints__]
    return object if param.__bound__ is None else param.__bound__


def _generic_bases(typx: type, /) -> tuple[tuple[_TypeParameter, ...], tuple]:
    """Return the type parameters and generic bases of typx."""
    if (bases := vars(typx).get("__orig_bases__")) is None:
        bases = _stub_bases().get(typx, typx.__bases__)
    if "_fields" in vars(typx) and builtins.tuple in typx.__bases__:
        # a namedtuple's tuple base has one element per field
        annotations = annotationlib.get_annotations(
            typx, format=annotationlib.Format.FORWARDREF
        )
        tuple_base = tuple[*(annotations.get(f, tp.Any) for f in typx._fields)]
        bases = tuple(
            tuple_base if b in (builtins.tuple, tp.NamedTuple) else b for b in bases
        )
    for base in bases:
        if tp.get_origin(base) in (tp.Generic, tp.Protocol):
            return tp.get_args(base), bases
    # a bare typing alias like typing.MappingView has no __parameters__
    params = (
        p for b in bases if tp.get_origin(b) for p in getattr(b, "__parameters__", ())
    )
    return tuple(dict.fromkeys(params)), bases


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
        # the type parameter of tuple is the union of its elements
        args = (tp.Union[*(t for t, _ in entries)] if entries else tp.Never,)
    subst: dict[tp.TypeVar, tp.Any] = {}
    for i, param in enumerate(params):
        if not isinstance(param, tp.TypeVar):
            return
        arg = args[i] if i < len(args) else _default_arg(param, erased=erased)
        subst[param] = tp.Any if arg is ... else arg
    return subst


def _specialize(src, dst: type, /, *, erased: bool):
    """Return the base of src whose origin is dst, with the type arguments of
    src substituted, or None if no generic base of src leads to dst.
    """
    stack: list[tuple[tp.Any, tp.Mapping]] = [(src, {})]
    while stack:
        typx, subst = stack.pop()
        origin = tp.get_origin(typx) or typx
        if origin is dst:
            return _substitute(typx, subst)
        # a base that is not a subclass of dst cannot lead to it
        if not (isinstance(origin, type) and issubclass(origin, dst)):
            continue
        params, bases = _generic_bases(origin)
        if (args := _type_args(typx, params, erased=erased)) is None:
            continue
        # map the type parameters of origin to the arguments of src
        subst = {p: _substitute(a, subst) for p, a in args.items()}
        stack += [(base, subst) for base in bases[::-1]]


def _map_args(src, dst: type, /, *, erased: bool) -> dict[tp.TypeVar, tp.Any] | None:
    """Map the type parameters of dst to the arguments src passes to them, or
    return None if no generic base of src leads to dst.
    """
    if (base := _specialize(src, dst, erased=erased)) is not None:
        params, _ = _generic_bases(dst)
        return _type_args(base, params, erased=erased)


def _adjust_tuple_entries(
    dst: list[tuple[tp.Any, bool]], src: list[tuple[tp.Any, bool]], /
) -> bool:
    """Pack the src entries that match the indeterminate entry of dst into one,
    and return whether dst and src then have the same length.
    """
    dst_i = next((i for i, (_, indeterminate) in enumerate(dst) if indeterminate), -1)
    src_i = next((i for i, (_, indeterminate) in enumerate(src) if indeterminate), -1)
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


def _substitute(typx, subst: tp.Mapping, /):
    if isinstance(typx, tp.TypeVar):
        return subst.get(typx, typx)
    if tp.get_origin(typx) and (params := getattr(typx, "__parameters__", ())):
        return typx[*(subst.get(p, p) for p in params)]
    return typx


class _VarianceDummy:
    pass


def _variance(
    typx: type, param: tp.TypeVar, /, *, assumed_variance: tp.Mapping[type, _Variance]
) -> _Variance:
    if (assumed := assumed_variance.get(typx)) is not None:
        return assumed
    if param.__covariant__:
        return _Variance.COVARIANT
    if param.__contravariant__:
        return _Variance.CONTRAVARIANT
    if not param.__infer_variance__:
        return _Variance.INVARIANT
    if (variances := _inferred_variance.get(typx)) is None:
        variances = _inferred_variance[typx] = _infer_variance(
            typx, assumed_variance=assumed_variance
        )
    return variances[param]


def _infer_variance(
    typx, /, *, assumed_variance: tp.Mapping[type, _Variance]
) -> dict[tp.TypeVar, _Variance]:
    params, _ = _generic_bases(typx)
    out = {}
    for param in params:
        if not (isinstance(param, tp.TypeVar) and param.__infer_variance__):
            continue
        bound = types.new_class(param.__name__)
        upper = typx[*(object if p is param else _VarianceDummy for p in params)]
        lower = typx[*(bound if p is param else _VarianceDummy for p in params)]
        # references to typx in its members compare by the assumed variance
        if _assign_class_to_self(
            upper,
            lower,
            assuming=frozenset(),
            assumed_variance={**assumed_variance, typx: _Variance.COVARIANT},
            self_class=typx,
        ):
            out[param] = _Variance.COVARIANT
        elif _assign_class_to_self(
            lower,
            upper,
            assuming=frozenset(),
            assumed_variance={**assumed_variance, typx: _Variance.CONTRAVARIANT},
            self_class=typx,
        ):
            out[param] = _Variance.CONTRAVARIANT
        else:
            out[param] = _Variance.INVARIANT
    return out


def _assign_class_to_self(
    dst: _SupportsGeneric, src: _SupportsGeneric, /, **ctx: tp.Unpack[_AssignContext]
) -> bool:
    params, _ = _generic_bases(dst.__origin__)
    dst_subst = dict(zip(params, dst.__args__))
    src_subst = dict(zip(params, src.__args__))
    return all(
        _assign_with_variance(
            _substitute(form, dst_subst), _substitute(form, src_subst), variance, **ctx
        )
        for form, variance in _declared_types(dst.__origin__)
    )


def _declared_types(typx: type, /) -> list[tuple[tp.Any, _Variance]]:
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
            out.append((form, _Variance.COVARIANT))
        else:
            # a frozen dataclass or NamedTuple field is replaceable by __replace__
            out.append((form, _Variance.INVARIANT))
    for name, attr in vars(typx).items():
        if name in ("__new__", "__init__"):
            continue
        is_bound = not isinstance(attr, staticmethod)
        if isinstance(attr, property):
            funcs = (attr.fget, attr.fset, attr.fdel)
        else:
            funcs = (attr,)
        for func in funcs:
            func = getattr(func, "__func__", func)
            if not isinstance(func, types.FunctionType):
                continue
            ovs = tp.get_overloads(func) or (func,)
            for sig in _get_signatures(*ovs, is_bound=is_bound):
                out += [
                    (p.annotation, _Variance.CONTRAVARIANT)
                    for p in sig.parameters.values()
                ]
                out.append((sig.return_annotation, _Variance.COVARIANT))
    _, bases = _generic_bases(typx)
    out += [
        (base, _Variance.COVARIANT)
        for base in bases
        if tp.get_origin(base) not in (tp.Generic, tp.Protocol, builtins.tuple)
    ]
    return out


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
            # a union dst can accept each value from a different member
            return all(
                _assign_type(dst, tp.Literal[arg], **ctx) for arg in src.__args__
            )
        case tp.Annotated:
            return _assign_type(dst, src.__origin__, **ctx)
        case tp.TypeAliasType() | tp.ForwardRef():
            # a recursive alias reaches the same pair again, which then holds
            if (dst, src) in ctx["assuming"]:
                return True
            ctx["assuming"] |= {(dst, src)}
            return _assign_type(dst, _evaluate(src), **ctx)
        case tp.NewType():
            return _assign_type(dst, src.__supertype__, **ctx)
        case tp.Self:
            return _assign_type(dst, ctx["self_class"], **ctx)
        case tp.TypeVar():
            if src.__constraints__:
                return all(_assign_type(dst, c, **ctx) for c in src.__constraints__)
            return _assign_type(dst, src.__bound__ or object, **ctx)
        case builtins.type if tp.get_args(src):
            [arg] = tp.get_args(src)
            # type[...] distributes over a union, and takes the bound or
            # constraints of a TypeVar
            if tp.get_origin(arg) is tp.Union:
                return all(_assign_type(dst, type[a], **ctx) for a in arg.__args__)
            if isinstance(arg, tp.TypeVar):
                if arg.__constraints__:
                    return all(
                        _assign_type(dst, type[c], **ctx) for c in arg.__constraints__
                    )
                return _assign_type(dst, type[arg.__bound__ or object], **ctx)
    return dst.__subtype_subclasscheck__(src, **ctx)


def _assign_with_variance(
    dst, src, variance: _Variance, /, **ctx: tp.Unpack[_AssignContext]
) -> bool:
    if dst == src:
        return True
    if variance is not _Variance.CONTRAVARIANT and not _assign_type(
        subtype(dst), src, **ctx
    ):
        return False
    return variance is _Variance.COVARIANT or _assign_type(subtype(src), dst, **ctx)


def _assign_generic(
    dst: _SupportsGeneric, src, /, *, erased=False, **ctx: tp.Unpack[_AssignContext]
):
    if not issubclass(tp.get_origin(src) or src, dst.__origin__):
        return False
    src_args = _map_args(src, dst.__origin__, erased=erased)
    dst_args = _map_args(dst, dst.__origin__, erased=False)
    if src_args is None or dst_args is None:
        return True
    return all(
        _assign_with_variance(
            dst_arg,
            src_args[p],
            _variance(dst.__origin__, p, assumed_variance=ctx["assumed_variance"]),
            **ctx,
        )
        for p, dst_arg in dst_args.items()
    )


class _generic_instancecheck(_checker):
    def __init__(self, typx: _SupportsGeneric, /):
        self.__form__ = typx

    def __call__(self, obj, /, **ctx):
        if (src := getattr(obj, "__dict__", {}).get("__orig_class__")) is not None:
            return _assign_generic(self.__form__, src, **ctx)
        return _assign_generic(self.__form__, type(obj), erased=True, **ctx)


class _generic_subclasscheck(_generic_instancecheck):
    def __call__(self, obj, /, **ctx):
        if tp.get_origin(obj) is tp.Literal:
            return all(map(ft.partial(super().__call__, **ctx), tp.get_args(obj)))
        return _assign_generic(self.__form__, obj, **ctx)


def _assign_tuple(dst, src, /, **ctx: tp.Unpack[_AssignContext]) -> bool:
    origin = tp.get_origin(src) or src
    if not (isinstance(origin, type) and issubclass(origin, builtins.tuple)):
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
        _assign_with_variance(d, s, _Variance.COVARIANT, **ctx)
        for (d, _), (s, _) in zip(dst_entries, src_entries)
    )


class _tuple_instancecheck(_checker):
    def __init__(self, typx: type[tuple], /):
        self.__form__ = typx

    def __call__(self, obj, /, **ctx):
        if not isinstance(obj, builtins.tuple):
            return False
        if (entries := _tuple_entries(self.__form__)) is None:
            return True
        ts = [t for t, _ in entries]
        i = next(
            (i for i, (_, indeterminate) in enumerate(entries) if indeterminate), None
        )
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
    elif isinstance(params, list):
        tail = None
    else:
        params, tail = [], params
    if isinstance(tail, tp.ParamSpec):
        raise NotImplementedError
    sig = [
        inspect.Parameter(f"_{i}", inspect.Parameter.POSITIONAL_ONLY, annotation=t)
        for i, t in enumerate(params)
    ]
    if tail is ...:
        sig += [
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ]
    return inspect.Signature(sig, return_annotation=ret)


def _assign_signature(
    dst: inspect.Signature, src: inspect.Signature, /, **ctx: tp.Unpack[_AssignContext]
) -> bool:
    """Return whether src is assignable to dst, the signature of a Callable form."""

    def annotation(typx, /):
        return tp.Any if typx is inspect.Parameter.empty else typx

    if not _assign_with_variance(
        dst.return_annotation,
        annotation(src.return_annotation),
        _Variance.COVARIANT,
        **ctx,
    ):
        return False
    src_params = [*src.parameters.values()]
    for dst_param in dst.parameters.values():
        # the ... of dst accepts the remaining parameters of src
        if dst_param.kind is not inspect.Parameter.POSITIONAL_ONLY:
            return True
        if not src_params or src_params[0].kind > inspect.Parameter.VAR_POSITIONAL:
            return False
        if src_params[0].kind is inspect.Parameter.VAR_POSITIONAL:
            src_param = src_params[0]
        else:
            src_param = src_params.pop(0)
        if not _assign_with_variance(
            dst_param.annotation,
            annotation(src_param.annotation),
            _Variance.CONTRAVARIANT,
            **ctx,
        ):
            return False
    return all(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        or p.default is not inspect.Parameter.empty
        for p in src_params
    )


class _callable_instancecheck(_checker):
    def __init__(self, typx: _SupportsGeneric, /):
        self.signature = _callable_signature(typx)

    def __call__(self, obj, /, **ctx):
        if not callable(obj):
            return False
        # Self in the signature of obj is the class obj binds to
        owner = getattr(obj, "__self__", obj)
        ctx["self_class"] = owner if isinstance(owner, type) else type(owner)
        return any(
            _assign_signature(self.signature, sig, **ctx) for sig in _get_overloads(obj)
        )


class _callable_subclasscheck(_callable_instancecheck):
    def __call__(self, obj, /, **ctx):
        origin = tp.get_origin(obj) or obj
        if origin is collections.abc.Callable and tp.get_args(obj):
            sigs = (_callable_signature(obj),)
        elif origin is builtins.type and tp.get_args(obj):
            # a value of type[X] calls the constructor of X
            [ctx["self_class"]] = tp.get_args(obj)
            sigs = _get_overloads(tp.get_origin(ctx["self_class"]) or ctx["self_class"])
        elif isinstance(origin, type) and issubclass(origin, collections.abc.Callable):
            call = inspect.getattr_static(origin, "__call__")
            ovs, _ = _resolve_call(call)
            sigs = _get_signatures(*(ovs or (call,)), is_bound=True)
        else:
            return False
        return any(_assign_signature(self.signature, sig, **ctx) for sig in sigs)


class _type_instancecheck(_checker):
    def __init__(self, typx: _SupportsGeneric, /):
        [arg] = typx.__args__
        self.__form__ = subtype(arg)

    def __call__(self, obj, /, **ctx):
        return isinstance(obj, type) and _assign_type(self.__form__, obj, **ctx)


class _type_subclasscheck(_type_instancecheck):
    def __call__(self, obj, /, **ctx):
        origin = tp.get_origin(obj) or obj
        if origin is builtins.type:
            # a bare type is type[Any]
            [arg] = tp.get_args(obj) or [tp.Any]
        elif isinstance(origin, type) and issubclass(origin, builtins.type):
            # an instance of a metaclass is a class
            arg = object
        else:
            return False
        return _assign_type(self.__form__, arg, **ctx)


_subtypes = weakref.WeakValueDictionary()
_inferred_variance = weakref.WeakKeyDictionary()


class subtype(type):
    __subtype_form__: tp.Any
    __subtype_instancecheck__: _checker
    __subtype_subclasscheck__: _checker

    def __new__(mcls, typx, /):
        if (cls := _subtypes.get(typx)) is not None:
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
                    dict.fromkeys((i_check, s_check), lambda _, **ctx: True)
                )
            case tp.Never | tp.NoReturn:
                namespace.update(
                    dict.fromkeys((i_check, s_check), lambda _, **ctx: False)
                )
            case tp.Literal:
                namespace[i_check] = _literal_instancecheck(typx)
                namespace[s_check] = _literal_subclasscheck(typx)
            case tp.Union:
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
                    ctx["self_class"]
                ).__subtype_instancecheck__(obj, **ctx)
                namespace[s_check] = lambda src, **ctx: _assign_type(
                    mcls(ctx["self_class"]), src, **ctx
                )
            case tp.NewType():
                supertype = mcls(typx.__supertype__)
                namespace[i_check] = lambda obj, **ctx: (
                    supertype.__subtype_instancecheck__(obj, **ctx)
                )
                # _assign_type admits typx and the NewTypes derived from it
                namespace[s_check] = lambda _, **ctx: False
            # tuple[()] has no arguments but is not the bare tuple class
            case builtins.tuple if hasattr(typx, "__args__"):
                namespace[i_check] = _tuple_instancecheck(typx)
                namespace[s_check] = _tuple_subclasscheck(typx)
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
            # TODO: implement these
            case tp.TypeVar() | str() | tp.LiteralString | tp.TypeGuard | tp.TypeIs:
                raise NotImplementedError
            case _:
                raise TypeError
        return _subtypes.setdefault(
            typx, super().__new__(mcls, mcls.__name__, (), namespace)
        )

    def __repr__(cls):
        return "{0.__name__}({0.__subtype_form__})".format(cls)

    def __instancecheck__(cls, obj, /):
        return cls.__subtype_instancecheck__(
            obj,
            assuming=frozenset(),
            assumed_variance={},
            self_class=getattr(obj, "__dict__", {}).get("__orig_class__", type(obj)),
        )

    def __subclasscheck__(cls, src, /):
        return _assign_type(
            cls, src, assuming=frozenset(), assumed_variance={}, self_class=src
        )

    @tp.final
    def __call__(cls) -> tp.Never:
        raise TypeError(f"cannot instantiate {type(cls).__qualname__} objects")
