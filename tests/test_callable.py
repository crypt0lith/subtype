import collections.abc
import typing as tp

import pytest

from subtypes import subtype


def f(x: int) -> str: ...


def g(x: int, y: str = "", *, z: bool = False) -> str: ...


def h(*args: int) -> bool: ...


def k(x: int, *, y: str) -> str: ...


def any_args(*args, **kwargs): ...


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


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (f, tp.Callable[[int], str], True),
        (f, tp.Callable[[int], object], True),
        (f, tp.Callable[[bool], str], True),
        (f, tp.Callable[[object], str], False),
        (f, tp.Callable[[int], int], False),
        (f, tp.Callable[[], str], False),
        (f, tp.Callable[[int, int], str], False),
        (f, tp.Callable[..., str], True),
        (f, tp.Callable[tp.Concatenate[int, ...], str], True),
        (f, tp.Callable[tp.Concatenate[str, ...], str], False),
        (g, tp.Callable[[int], str], True),
        (g, tp.Callable[[int, str], str], True),
        (g, tp.Callable[[int, str, bool], str], False),
        (h, tp.Callable[[], bool], True),
        (h, tp.Callable[[int, int, int], bool], True),
        (h, tp.Callable[[int, str], bool], False),
        (k, tp.Callable[[int], str], False),
        (k, tp.Callable[[int, str], str], False),
        (any_args, tp.Callable[[int, str], int], True),
        (ov, tp.Callable[[int], int], True),
        (ov, tp.Callable[[str], str], True),
        (ov, tp.Callable[[bytes], bytes], False),
        (len, tp.Callable[[list[int]], int], True),
        (len, tp.Callable[[int], int], False),
        (A, tp.Callable[[int], A], True),
        (A, tp.Callable[[str], A], False),
        (A(1).m, tp.Callable[[int], A], True),
        (B, tp.Callable[[int], B], True),
        (B, tp.Callable[[int], str], False),
        (dict, tp.Callable[[], dict], True),
        (int, tp.Callable[[str], int], True),
        (int, tp.Callable[[str], str], False),
        (Fn2(), tp.Callable[[int, int], int], True),
        (1, tp.Callable[..., tp.Any], False),
    ],
)
def test_instancecheck(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tp.Callable[[int], str], tp.Callable[[int], str], True),
        (tp.Callable[[object], bool], tp.Callable[[int], int], True),
        (tp.Callable[[int], str], tp.Callable[[object], str], False),
        (tp.Callable[[int], str], tp.Callable[[int], int], False),
        (tp.Callable[..., str], tp.Callable[[int, int], str], True),
        (tp.Callable[[int], str], tp.Callable[..., str], True),
        (tp.Callable[[int, str], str], tp.Callable[[int], str], False),
        (
            tp.Callable[tp.Concatenate[int, ...], str],
            tp.Callable[[int, str], str],
            True,
        ),
        (collections.abc.Callable, tp.Callable[[int], int], True),
        (Fn, tp.Callable[[int], bool], True),
        (Fn, tp.Callable[[int], str], False),
        (Fn2, tp.Callable[[int, int], int], True),
        (Fn2, tp.Callable[[int], int], False),
        (int, tp.Callable[..., tp.Any], False),
        (tp.Literal[1], tp.Callable[..., tp.Any], False),
        (tp.Callable[[int], str] | Fn, tp.Callable[[int], object], True),
    ],
)
def test_subclasscheck(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected
