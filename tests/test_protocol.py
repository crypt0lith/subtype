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


class Box[T]: ...


class Pair[T1, T2]: ...


class Reduces:
    def __new__(cls, x, /):
        self = super().__new__(cls)
        self.x = x
        return self

    def __reduce__(self):
        return type(self), (self.x,)


class Either[T](Reduces):
    def __init__(self, x: Box[T] | set[T], /) -> None: ...


class Swap[T](Reduces):
    def __init__(self, x: Pair[T, str] | Pair[str, T], /) -> None: ...


class Wrapped[T](Reduces):
    def __init__(self, x: T | Box[T], /) -> None: ...


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (IntGetter, SupportsGet[int], True),
        (IntGetter, SupportsGet[object], True),
        (IntGetter, SupportsGet[str], False),
        (Nothing, SupportsGet[int], False),
    ],
)
def test_subclass_members(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Counter[int], collections.abc.Iterator[int], True),
        (Counter[int], collections.abc.Iterable[int], True),
        (Counter[str], collections.abc.Iterator[int], False),
        (zip, collections.abc.Iterable[object], True),
        (zip, collections.abc.Iterable[tuple[int, str]], False),
    ],
)
def test_subclass_iterator(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (IntGetter(), SupportsGet[int], True),
        (IntGetter(), SupportsGet[str], False),
        (Nothing(), SupportsGet[int], False),
    ],
)
def test_instance_members(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


def test_instance_iterator():
    # an erased Self takes Any for its type arguments
    assert isinstance(Counter(), subtype(collections.abc.Iterable[int]))


def test_instance_generator():
    # a class with no members to compare relies on the runtime
    assert isinstance((x for x in ()), subtype(collections.abc.Iterator[int]))


@pytest.fixture
def zip_pairs():
    return zip(*[range(10)] * 2)


@pytest.mark.parametrize(
    "dst, expected",
    [
        (tp.Iterable[tuple[int, int]], True),
        (tp.Iterable[tuple[int, str]], False),
        (tp.Iterable[tuple[int, str, int]], False),
    ],
)
def test_inferred_zip(zip_pairs, dst, expected):
    assert isinstance(zip_pairs, subtype(dst)) is expected


def test_inferred_zip_empty():
    assert isinstance(zip(), subtype(tp.Iterable[int]))


def test_inferred_zip_wide():
    assert isinstance(zip(*[[1]] * 8), subtype(tp.Iterable[tuple[int, ...]]))


@pytest.fixture
def zip_nested():
    return zip(zip("a"), "b")


@pytest.mark.parametrize(
    "dst, expected",
    [
        (tp.Iterable[tuple[tuple[str], str]], True),
        (tp.Iterable[tuple[tuple[int], str]], False),
    ],
)
def test_inferred_zip_nested(zip_nested, dst, expected):
    assert isinstance(zip_nested, subtype(dst)) is expected


@pytest.fixture
def range_iterator():
    return iter(range(3))


@pytest.mark.parametrize(
    "dst, expected", [(tp.Iterator[int], True), (tp.Iterator[str], False)]
)
def test_inferred_range_iterator(range_iterator, dst, expected):
    assert isinstance(range_iterator, subtype(dst)) is expected


@pytest.fixture
def str_iterator():
    return iter("ab")


@pytest.mark.parametrize(
    "dst, expected", [(tp.Iterable[str], True), (tp.Iterable[int], False)]
)
def test_inferred_str_iterator(str_iterator, dst, expected):
    assert isinstance(str_iterator, subtype(dst)) is expected


@pytest.fixture
def seq_iterator():
    return iter(Seq())


@pytest.mark.parametrize(
    "dst, expected", [(tp.Iterator[str], True), (tp.Iterator[int], False)]
)
def test_inferred_seq_iterator(seq_iterator, dst, expected):
    assert isinstance(seq_iterator, subtype(dst)) is expected


@pytest.fixture
def map_str():
    return map(str, [1])


@pytest.mark.parametrize(
    "dst, expected", [(tp.Iterable[str], True), (tp.Iterable[int], False)]
)
def test_inferred_map(map_str, dst, expected):
    assert isinstance(map_str, subtype(dst)) is expected


def test_inferred_map_lambda():
    assert isinstance(map(lambda x: x, [1]), subtype(tp.Iterable[int]))


@pytest.fixture
def filter_none():
    return filter(None, iter(range(2)))


@pytest.mark.parametrize(
    "dst, expected", [(tp.Iterable[int], True), (tp.Iterable[str], False)]
)
def test_inferred_filter(filter_none, dst, expected):
    assert isinstance(filter_none, subtype(dst)) is expected


@pytest.fixture
def enumerate_str():
    return enumerate("ab")


@pytest.mark.parametrize(
    "dst, expected",
    [(tp.Iterable[tuple[int, str]], True), (tp.Iterable[tuple[str, str]], False)],
)
def test_inferred_enumerate(enumerate_str, dst, expected):
    assert isinstance(enumerate_str, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (frozenset("a"), collections.abc.Set[str], True),
        (frozenset("a"), collections.abc.Set[int], False),
        (frozenset(["a", 1]), collections.abc.Set[str], False),
        (frozenset(["a", 1]), collections.abc.Set[str | int], True),
        # the reduce callable of this has no signature
        (collections.deque([1]), tp.Iterable[int], True),
    ],
)
def test_inferred_collection(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (Either(Box[int]()), Either[int], True),
        (Either(Box[int]()), Either[str], False),
        (Swap(Pair[str, int]()), Swap[int], True),
        (Swap(Pair[str, int]()), Swap[str], False),
        (Wrapped(Box[int]()), Wrapped[int], True),
        (Wrapped(Box[int]()), Wrapped[Box[int]], False),
    ],
)
def test_inferred_union_param(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected
