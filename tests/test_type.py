import abc
import typing as tp

import pytest

from subtypes import subtype

T = tp.TypeVar("T", bound=int)
C = tp.TypeVar("C", int, str)


class A:
    def __init__(self, x: int) -> None: ...


class B:
    def __init__(self, x: str) -> None: ...


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (int, type[int], True),
        (bool, type[int], True),
        (str, type[int], False),
        (1, type[int], False),
        (str, type[int | str], True),
        (str, type[tp.Any], True),
        (list, type[list[object]], True),
        (list, type[list[int]], False),
        (int, tp.Type[int], True),
    ],
)
def test_instancecheck(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (type[int], type[int], True),
        (type[bool], type[int], True),
        (type[int], type[bool], False),
        (type[int | str], type[int], False),
        (type[int | str], type[int] | type[str], True),
        (type[int | str], type[int | str | bytes], True),
        (type[T], type[int], True),
        (type[C], type[int], False),
        (type[C], type[int] | type[str], True),
        (type, type[int], True),
        (abc.ABCMeta, type[object], True),
        (abc.ABCMeta, type[int], False),
        (int, type[int], False),
        (type[int], type, True),
        (type[int], object, True),
        (type[int], int, False),
        (type[int], tp.Callable[[str], int], True),
        (type[int], tp.Callable[[], str], False),
        (type[A], tp.Callable[[int], A], True),
        (type[A], tp.Callable[[str], A], False),
        (type[A | int], tp.Callable[[int], object], True),
        (type[A | B], tp.Callable[[int], object], False),
        (type[T], tp.Callable[[str], int], True),
        (type[tp.Any], tp.Callable[[int], str], True),
        (type, tp.Callable[[int], str], True),
    ],
)
def test_subclasscheck(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected
