import collections
import collections.abc
import typing as tp

import pytest

from subtypes import subtype


class SupportsGet[T](tp.Protocol):
    def get(self, /) -> T: ...


class IntGetter:
    def get(self, /) -> int: ...


class Nothing: ...


class Counter[T]:
    def __iter__(self) -> tp.Self: ...

    def __next__(self) -> T: ...


class Seq:
    def __getitem__(self, i: int, /) -> str: ...


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (IntGetter, SupportsGet[int], True),
        (IntGetter, SupportsGet[float], True),
        (IntGetter, SupportsGet[str], False),
        (Nothing, SupportsGet[int], False),
        (Counter[int], collections.abc.Iterator[int], True),
        (Counter[int], collections.abc.Iterable[int], True),
        (Counter[str], collections.abc.Iterator[int], False),
        (zip, collections.abc.Iterable[object], True),
        (zip, collections.abc.Iterable[tuple[int, str]], False),
    ],
)
def test_subclasscheck(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (IntGetter(), SupportsGet[int], True),
        (IntGetter(), SupportsGet[str], False),
        (Nothing(), SupportsGet[int], False),
        # an erased Self takes Any for its type arguments
        (Counter(), collections.abc.Iterable[int], True),
        # a class with no members to compare relies on the runtime
        ((x for x in ()), collections.abc.Iterator[int], True),
    ],
)
def test_instancecheck(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (zip(*[range(10)] * 2), tp.Iterable[tuple[int, int]], True),
        (zip(*[range(10)] * 2), tp.Iterable[tuple[int, str]], False),
        (zip(*[range(10)] * 2), tp.Iterable[tuple[int, str, int]], False),
        (zip(), tp.Iterable[int], True),
        (zip(*[[1]] * 8), tp.Iterable[tuple[int, ...]], True),
        (zip(zip("a"), "b"), tp.Iterable[tuple[tuple[str], str]], True),
        (zip(zip("a"), "b"), tp.Iterable[tuple[tuple[int], str]], False),
        (iter(range(3)), tp.Iterator[int], True),
        (iter(range(3)), tp.Iterator[str], False),
        (iter("ab"), tp.Iterable[str], True),
        (iter("ab"), tp.Iterable[int], False),
        (iter(Seq()), tp.Iterator[str], True),
        (iter(Seq()), tp.Iterator[int], False),
        (map(str, [1]), tp.Iterable[str], True),
        (map(str, [1]), tp.Iterable[int], False),
        (map(lambda x: x, [1]), tp.Iterable[int], True),
        (filter(None, iter(range(2))), tp.Iterable[int], True),
        (filter(None, iter(range(2))), tp.Iterable[str], False),
        (enumerate("ab"), tp.Iterable[tuple[int, str]], True),
        (enumerate("ab"), tp.Iterable[tuple[str, str]], False),
        # the reduce callable of this has no signature
        (collections.deque([1]), tp.Iterable[int], True),
    ],
)
def test_inferred(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected
