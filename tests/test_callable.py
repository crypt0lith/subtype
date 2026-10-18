import collections.abc
import typing as tp

import pytest

from subtype import subtype


def f(x: int) -> str: ...


def g(x: int, y: str = "", *, z: bool = False) -> str: ...


def h(*args: int) -> bool: ...


def k(x: int, *, y: str) -> str: ...


def any_args(*args, **kwargs): ...


def ident[S](x: S) -> S: ...


@tp.overload
def ov(x: int) -> int: ...
@tp.overload
def ov(x: str) -> str: ...
def ov(x): ...


class A:
    def __init__(self, x: int) -> None: ...

    def m(self, x: int) -> tp.Self: ...


class B:
    def __init__(self, x): ...


class Fn(tp.Protocol):
    def __call__(self, x: object) -> bool: ...


class Fn2:
    def __call__(self, x: int, y: int) -> int: ...


class Gen[T](tp.Protocol):
    def __call__(self, x: T) -> T: ...


class GenSub[T](Gen[T], tp.Protocol): ...


class OverloadCall(tp.Protocol):
    @tp.overload
    def __call__(self, x: int) -> int: ...
    @tp.overload
    def __call__(self, x: str) -> str: ...


class Identity(tp.Protocol):
    def __call__[S](self, x: S) -> S: ...


class IdentityIntBound(tp.Protocol):
    def __call__[S: int](self, x: S) -> S: ...


class Forward[**P](tp.Protocol):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> int: ...


P = tp.ParamSpec("P")


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (f, tp.Callable[[int], str], True),
        (f, tp.Callable[[int], object], True),
        (f, tp.Callable[[int], int], False),
    ],
)
def test_instance_return(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (f, tp.Callable[[bool], str], True),
        (f, tp.Callable[[object], str], False),
        (f, tp.Callable[[], str], False),
        (f, tp.Callable[[int, int], str], False),
    ],
)
def test_instance_params(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (f, tp.Callable[..., str], True),
        (f, tp.Callable[tp.Concatenate[int, ...], str], True),
        (f, tp.Callable[tp.Concatenate[str, ...], str], False),
        (any_args, tp.Callable[[int, str], int], True),
    ],
)
def test_instance_gradual(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (g, tp.Callable[[int], str], True),
        (g, tp.Callable[[int, str], str], True),
        (g, tp.Callable[[int, str, bool], str], False),
    ],
)
def test_instance_defaults(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (h, tp.Callable[[], bool], True),
        (h, tp.Callable[[int, int, int], bool], True),
        (h, tp.Callable[[int, str], bool], False),
    ],
)
def test_instance_varargs(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [(k, tp.Callable[[int], str], False), (k, tp.Callable[[int, str], str], False)],
)
def test_instance_keyword_only(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (ov, tp.Callable[[int], int], True),
        (ov, tp.Callable[[str], str], True),
        (ov, tp.Callable[[bytes], bytes], False),
    ],
)
def test_instance_overload(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


def test_instance_generic_function():
    assert isinstance(ident, subtype(tp.Callable[[int], int]))


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (f, tp.Callable[P, str], True),
        (f, tp.Callable[P, int], False),
        (f, tp.Callable[tp.Concatenate[int, P], str], True),
        (f, tp.Callable[tp.Concatenate[str, P], str], False),
    ],
)
def test_instance_paramspec(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [(len, tp.Callable[[list[int]], int], True), (len, tp.Callable[[int], int], False)],
)
def test_instance_builtin(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (A, tp.Callable[[int], A], True),
        (A, tp.Callable[[str], A], False),
        (B, tp.Callable[[int], B], True),
        (B, tp.Callable[[int], str], False),
        (dict, tp.Callable[[], dict], True),
        (int, tp.Callable[[str], int], True),
        (int, tp.Callable[[str], str], False),
    ],
)
def test_instance_class(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (A(1).m, tp.Callable[[int], A], True),
        (Fn2(), tp.Callable[[int, int], int], True),
        (1, tp.Callable[..., tp.Any], False),
    ],
)
def test_instance_object(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tp.Callable[[int], str], tp.Callable[[int], str], True),
        (tp.Callable[[object], bool], tp.Callable[[int], int], True),
        (tp.Callable[[int], str], tp.Callable[[object], str], False),
        (tp.Callable[[int], str], tp.Callable[[int], int], False),
        (tp.Callable[[int, str], str], tp.Callable[[int], str], False),
    ],
)
def test_subclass_callable(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tp.Callable[..., str], tp.Callable[[int, int], str], True),
        (tp.Callable[[int], str], tp.Callable[..., str], True),
        (
            tp.Callable[tp.Concatenate[int, ...], str],
            tp.Callable[[int, str], str],
            True,
        ),
        (collections.abc.Callable, tp.Callable[[int], int], True),
    ],
)
def test_subclass_gradual(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Fn, tp.Callable[[int], bool], True),
        (Fn, tp.Callable[[int], str], False),
        (Fn2, tp.Callable[[int, int], int], True),
        (Fn2, tp.Callable[[int], int], False),
    ],
)
def test_subclass_dunder_call(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Gen[str], tp.Callable[[str], str], True),
        (Gen[str], tp.Callable[[int], int], False),
        (Gen[bool], tp.Callable[[bool], int], True),
        (GenSub[int], tp.Callable[[int], int], True),
        (GenSub[int], tp.Callable[[str], str], False),
        (Gen, tp.Callable[[int], int], True),
        (Gen, tp.Callable[[int], str], False),
    ],
)
def test_subclass_generic_protocol(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (OverloadCall, tp.Callable[[int], int], True),
        (OverloadCall, tp.Callable[[str], str], True),
        (OverloadCall, tp.Callable[[bytes], bytes], False),
    ],
)
def test_subclass_overload_protocol(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Identity, tp.Callable[[int], int], True),
        (IdentityIntBound, tp.Callable[[str], object], False),
    ],
)
def test_subclass_method_typevar(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Forward, tp.Callable[[int], int], True),
        (Forward, tp.Callable[[int], str], False),
    ],
)
def test_subclass_paramspec(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (int, tp.Callable[..., tp.Any], False),
        (tp.Literal[1], tp.Callable[..., tp.Any], False),
    ],
)
def test_subclass_not_callable(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize("src", [tp.Callable[[int], str] | Fn, Fn | Gen[int]])
def test_subclass_union(src):
    assert issubclass(src, subtype(tp.Callable[[int], object]))
