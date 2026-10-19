import collections.abc
import typing as tp

import pytest

from subtype import subtype


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


class SupportsKeysAndGetItem[K, V](tp.Protocol):
    def keys(self) -> collections.abc.Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


class KeysAndGetItem:
    def keys(self):
        return set("ab")

    def __getitem__(self, key, /):
        return 42


class SupportsLenAndGetItem[T](tp.Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, k: int, /) -> T: ...


class LenAndGetItem:
    def __len__(self):
        return 2

    def __getitem__(self, index, /):
        if index >= len(self):
            raise IndexError
        return index


class SupportsGetItem[T](tp.Protocol):
    def __getitem__(self, k: int, /) -> T: ...


class GetItem:
    def __getitem__(self, index, /):
        if index >= 2:
            raise IndexError
        return index


class UnparameterizedList(list): ...


type Ints = list[int]
type NestedInts = int | list[NestedInts]
type NestedDict = int | dict[str, NestedDict]
type J1 = list[J1]
type J2 = list[J2]
type J3 = list[list[J3]]


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[int], collections.abc.Iterable[int], True),
        (list[bool], collections.abc.Iterable[int], True),
        (list[str], collections.abc.Iterable[int], False),
        (tp.Sequence[int], tp.Iterable[int], True),
        (frozenset[int], tp.AbstractSet[int], True),
    ],
)
def test_subclass_covariant(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[int], list[int], True),
        (list[bool], list[int], False),
        (list[int], list[bool], False),
        (list[int | str], list[str | int], True),
    ],
)
def test_subclass_invariant(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[tp.Any], list[int], True),
        (list[int], list[tp.Any], True),
        (collections.abc.Sequence[tp.Any], collections.abc.Iterable[int], True),
        (collections.abc.Sequence[...], collections.abc.Iterable[int], True),
    ],
)
def test_subclass_any(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list, collections.abc.Iterable[int], False),
        (list, collections.abc.Iterable[object], True),
        (collections.abc.Sequence, collections.abc.Iterable[int], False),
        (BareIterable, collections.abc.Iterable[int], False),
        (BareIterable, collections.abc.Iterable[object], True),
    ],
)
def test_subclass_bare(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (str, collections.abc.Sequence[str], True),
        (str, collections.abc.Iterable[int], False),
        (bytes, collections.abc.Sequence[int], True),
        (int, collections.abc.Iterable[int], False),
    ],
)
def test_subclass_non_generic(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (A, collections.abc.Iterable[int], True),
        (A, collections.abc.Iterable[str], False),
        (IntList, collections.abc.Sequence[int], True),
        (IntList, collections.abc.Sequence[str], False),
        (IntList, list[int], True),
    ],
)
def test_subclass_specialized_base(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (B, collections.abc.Iterable[int], True),
        (B, collections.abc.Iterable[bool], False),
        (G, collections.abc.Iterable[int], True),
        (G, collections.abc.Iterable[bool], False),
    ],
)
def test_subclass_unspecialized(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (B[bool], collections.abc.Iterable[int], True),
        (Box[int], Box[int], True),
        (Box[bool], Box[int], True),
        (Box[int], Box[bool], False),
    ],
)
def test_subclass_user_generic(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (dict[str, bool], collections.abc.Mapping[str, int], True),
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
    ],
)
def test_subclass_mapping(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tp.Literal["ab"], collections.abc.Iterable[str], True),
        (tp.Literal[b"x"], collections.abc.Iterable[str], False),
    ],
)
def test_subclass_literal(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (
            tp.Literal["ab", b"x"],
            collections.abc.Iterable[str] | collections.abc.Iterable[int],
            True,
        ),
        (
            list[int],
            collections.abc.Iterable[str] | collections.abc.Sequence[int],
            True,
        ),
    ],
)
def test_subclass_union(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
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
    ],
)
def test_subclass_generator(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Ints, collections.abc.Iterable[int], True),
        (Ints, collections.abc.Iterable[str], False),
    ],
)
def test_subclass_alias(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[NestedInts], NestedInts, True),
        (list[int], NestedInts, False),
        (J1, J2, True),
        (J1, J3, True),
        (list[J1], J2, True),
        (list[int], J1, False),
    ],
)
def test_subclass_recursive_alias(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ([1], list[int], True),
        ([1], collections.abc.Iterable[int], True),
        ((), collections.abc.Sequence[int], True),
        (["a"], list[int], False),
        ([True], collections.abc.Sequence[int], True),
        (["a"], collections.abc.Iterable[int], False),
        (["a"], collections.abc.Reversible[int], False),
        (UnparameterizedList(["a"]), list[int], False),
        (LenAndGetItem(), SupportsLenAndGetItem[int], True),
        (LenAndGetItem(), SupportsLenAndGetItem[str], False),
        ([1], SupportsLenAndGetItem[str], False),
        (GetItem(), SupportsGetItem[int], True),
        (GetItem(), SupportsGetItem[str], False),
    ],
)
def test_instance_builtin(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


def test_instance_generator():
    assert isinstance((x for x in ()), subtype(collections.abc.Iterator[int]))


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ("ab", collections.abc.Iterable[int], False),
        ("ab", collections.abc.Sequence[str], True),
        (b"ab", collections.abc.Sequence[int], True),
        ("ab", collections.abc.Sequence[collections.abc.Sequence[str]], True),
        (1, collections.abc.Iterable[int], False),
    ],
)
def test_instance_non_generic(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ({}, collections.abc.Mapping[str, int], True),
        ({1: 1}, collections.abc.Mapping[str, int], False),
        ({"a": 1}, collections.abc.Mapping[str, int], True),
        ({"a": "b"}, collections.abc.Mapping[str, int], False),
        ({"a": ["b"]}, dict[str, list[int]], False),
        ({"a": "b"}, collections.abc.Collection[str], True),
        (KeysAndGetItem(), SupportsKeysAndGetItem[str, int], True),
        (KeysAndGetItem(), SupportsKeysAndGetItem[str, str], False),
        (KeysAndGetItem(), SupportsKeysAndGetItem[int, int], False),
        ({"a": "b"}, SupportsKeysAndGetItem[str, int], False),
        (collections.defaultdict(int, a="b"), dict[str, int], False),
        ({}.keys(), collections.abc.KeysView[int], True),
        ({}.keys(), collections.abc.Sequence[int], False),
    ],
)
def test_instance_mapping(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (A(), collections.abc.Iterable[int], True),
        (A(), collections.abc.Iterable[str], False),
        (B(), collections.abc.Iterable[int], True),
        (Box[int](), Box[int], True),
        (Box[str](), Box[int], False),
        (Box(), Box[int], True),
    ],
)
def test_instance_user_generic(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ([1, [2]], NestedInts, True),
        ([1, ["a"]], NestedInts, False),
        ([[1], [2, ["a"]]], NestedInts, False),
        ([["a"]], list[list[int] | list[bytes]], False),
        ("a", NestedInts, False),
    ],
)
def test_instance_nested(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.fixture(params=[1, 2], ids=["direct", "indirect"])
def cycle(request):
    lists = [[] for _ in range(request.param)]
    for a, b in zip(lists, lists[1:] + lists[:1]):
        a.append(b)
    return lists[0]


@pytest.mark.parametrize(
    "dst, expected", [(NestedInts, True), (list[list[int]], False)]
)
def test_cycle(cycle, dst, expected):
    assert isinstance(cycle, subtype(dst)) is expected


def test_cycle_mapping():
    d = {}
    d["a"] = d
    assert isinstance(d, subtype(NestedDict))
