import collections.abc
import typing as tp

import pytest

from subtypes import subtype


class A(collections.abc.Iterable[int]):
    def __iter__(self): ...


class B[T: int](collections.abc.Iterable[T]):
    def __iter__(self): ...


class G[T = int](collections.abc.Iterable[T]):
    def __iter__(self): ...


class IntList(list[int]): ...


class BareIterable(collections.abc.Iterable):
    def __iter__(self): ...


class Box[T]:
    def __init__(self, item=None): ...


type Ints = list[int]
type Json = int | list[Json]
type J1 = list[J1]
type J2 = list[J2]
type J3 = list[list[J3]]


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[int], collections.abc.Iterable[int], True),
        (list[bool], collections.abc.Iterable[int], True),
        (list[str], collections.abc.Iterable[int], False),
        (list[int], collections.abc.Iterable[float], True),
        (list[int], list[int], True),
        (list[bool], list[int], False),
        (list[int], list[bool], False),
        (list[int | str], list[str | int], True),
        (list[tp.Any], list[int], True),
        (list[int], list[tp.Any], True),
        (list, collections.abc.Iterable[int], False),
        (list, collections.abc.Iterable[object], True),
        (collections.abc.Sequence, collections.abc.Iterable[int], False),
        (collections.abc.Sequence[tp.Any], collections.abc.Iterable[int], True),
        (collections.abc.Sequence[...], collections.abc.Iterable[int], True),
        (tp.Sequence[int], tp.Iterable[int], True),
        (A, collections.abc.Iterable[int], True),
        (A, collections.abc.Iterable[str], False),
        (B, collections.abc.Iterable[int], True),
        (B, collections.abc.Iterable[bool], False),
        (B[bool], collections.abc.Iterable[int], True),
        (G, collections.abc.Iterable[int], True),
        (G, collections.abc.Iterable[bool], False),
        (IntList, collections.abc.Sequence[int], True),
        (IntList, collections.abc.Sequence[str], False),
        (IntList, list[int], True),
        (BareIterable, collections.abc.Iterable[int], False),
        (BareIterable, collections.abc.Iterable[object], True),
        (dict[str, int], collections.abc.Mapping[str, float], True),
        (dict[str, int], collections.abc.Mapping[bytes, int], False),
        (dict[bool, int], collections.abc.Mapping[int, int], False),
        (
            dict[str, list[bool]],
            collections.abc.Mapping[str, collections.abc.Sequence[int]],
            True,
        ),
        (
            dict[str, list[str]],
            collections.abc.Mapping[str, collections.abc.Sequence[int]],
            False,
        ),
        (str, collections.abc.Sequence[str], True),
        (str, collections.abc.Iterable[int], False),
        (bytes, collections.abc.Sequence[int], True),
        (frozenset[int], tp.AbstractSet[int], True),
        (tp.Literal["ab"], collections.abc.Iterable[str], True),
        (tp.Literal[b"x"], collections.abc.Iterable[str], False),
        (
            tp.Literal["ab", b"x"],
            collections.abc.Iterable[str] | collections.abc.Iterable[int],
            True,
        ),
        (collections.abc.Generator[int], collections.abc.Iterator[int], True),
        (
            collections.abc.Generator[int, int, None],
            collections.abc.Generator[int],
            False,
        ),
        (
            collections.abc.Generator[int, None, None],
            collections.abc.Generator[int, int, None],
            False,
        ),
        (
            collections.abc.Generator[int, object, None],
            collections.abc.Generator[int, int, None],
            True,
        ),
        (int, collections.abc.Iterable[int], False),
        (
            list[int],
            collections.abc.Iterable[str] | collections.abc.Sequence[int],
            True,
        ),
        (Ints, collections.abc.Iterable[int], True),
        (Ints, collections.abc.Iterable[str], False),
        (Box[int], Box[int], True),
        (Box[bool], Box[int], True),
        (Box[int], Box[bool], False),
        (list[Json], Json, True),
        (list[int], Json, False),
        (J1, J2, True),
        (J1, J3, True),
        (list[J1], J2, True),
        (list[int], J1, False),
    ],
)
def test_subclasscheck(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ([1], list[int], True),
        ([1], collections.abc.Iterable[int], True),
        ((), collections.abc.Sequence[int], True),
        ("ab", collections.abc.Iterable[int], False),
        ("ab", collections.abc.Sequence[str], True),
        (b"ab", collections.abc.Sequence[int], True),
        (1, collections.abc.Iterable[int], False),
        ({}, collections.abc.Mapping[str, int], True),
        ({}.keys(), collections.abc.KeysView[int], True),
        ({}.keys(), collections.abc.Sequence[int], False),
        ((x for x in ()), collections.abc.Iterator[int], True),
        (A(), collections.abc.Iterable[int], True),
        (A(), collections.abc.Iterable[str], False),
        (B(), collections.abc.Iterable[int], True),
        (Box[int](), Box[int], True),
        (Box[str](), Box[int], False),
        (Box(), Box[int], True),
        ([1, [2]], Json, True),
        ("a", Json, False),
    ],
)
def test_instancecheck(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected
