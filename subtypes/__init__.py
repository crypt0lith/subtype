import abc
import annotationlib
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


def _unpack_tuple_items(typx, /) -> tuple[tp.Any, ...]:
    if tp.get_origin(typx) is tp.Unpack and not tp.is_typeddict(
        tup := tp.get_args(typx)[0]
    ):
        typx = tup
    if getattr(typx, "__unpacked__", False) and not _is_indeterminate_tuple(typx):
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
            if getattr(x, "__unpacked__", False) and _is_indeterminate_tuple(x)
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
    unpacked = getattr(typx, "__unpacked__", False)
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
        if (
            unpacked
            and type(typx) is types.GenericAlias
            and not getattr(typx, "__unpacked__", False)
        ):
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
