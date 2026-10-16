import collections
import collections.abc
import typing as tp

import pytest

from subtypes import subtype


class P(tp.NamedTuple):
    x: int
    y: str


class Q[T](tp.NamedTuple):
    x: T
    y: int


class Pair(tuple[int, str]): ...


class Ints[T](tuple[T, ...]): ...


R = collections.namedtuple("R", "a b")


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int, str], tuple[int, str], True),
        (tuple[int], tuple[()], False),
        (tuple[()], tp.Tuple[()], True),
    ],
)
def test_subclass_fixed(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int, int], tuple[int, ...], True),
        (tuple[()], tuple[int, ...], True),
        (tuple[int, ...], tuple[int, int], False),
        (tuple[bool, ...], tuple[int, ...], True),
    ],
)
def test_subclass_homogeneous(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[tp.Any, ...], tuple[int, int], True),
        (tuple[int, str], tuple[tp.Any, ...], True),
    ],
)
def test_subclass_any(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int], tuple, True),
        (tuple, tuple[int], False),
        (tuple, tuple[object, ...], True),
    ],
)
def test_subclass_bare(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[int], tuple[int], False),
        (tp.Literal[1], tuple[int], False),
        (tp.Literal[1, "a"], tuple[int, ...], False),
    ],
)
def test_subclass_not_tuple(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int], tuple[int, *tuple[str, ...]], True),
        (tuple[int, str, str], tuple[int, *tuple[str, ...]], True),
        (tuple[str], tuple[int, *tuple[str, ...]], False),
        (tuple[()], tuple[int, *tuple[str, ...]], False),
        (tuple[int, int], tuple[int, *tuple[str, ...], int], True),
        (tuple[int, str], tp.Tuple[int, tp.Unpack[tp.Tuple[str, ...]]], True),
    ],
)
def test_subclass_fixed_to_unpacked(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int, *tuple[str, ...], int], tuple[int, *tuple[str, ...], int], True),
        (tuple[int, ...], tuple[int, *tuple[str, ...], int], False),
        (tuple[int, *tuple[int, ...]], tuple[int, ...], True),
        (tuple[int, ...], tuple[int, *tuple[int, ...]], False),
        (
            tuple[int, *tuple[str, ...], str, bytes],
            tuple[int, *tuple[str, ...], bytes],
            True,
        ),
        (tuple[int, str, *tuple[str, ...]], tuple[int, *tuple[str, ...]], True),
        (tuple[str, *tuple[int, ...]], tuple[*tuple[int, ...], str], False),
    ],
)
def test_subclass_unbounded_unpacked(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int, *tuple[str, int]], tuple[int, str, int], True),
        (tuple[int, *tuple[str, int]], tuple[int, ...], False),
    ],
)
def test_subclass_fixed_unpacked(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (P, tuple[int, str], True),
        (P, tuple[int, int], False),
        (P, collections.abc.Sequence[int], False),
        (P, collections.abc.Sequence[int | str], True),
        (Q[bytes], tuple[bytes, int], True),
        (Q[bytes], tuple[str, int], False),
        (R, tuple[int, int], True),
        (R, tuple[int], False),
    ],
)
def test_subclass_namedtuple(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (Pair, tuple[int, str], True),
        (Pair, tuple[int, ...], False),
        (Pair, tuple[int | str, ...], True),
        (Ints[int], tuple[int, ...], True),
        (Ints[int], tuple[str, ...], False),
    ],
)
def test_subclass_tuple_base(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tuple[int, str], collections.abc.Sequence[int], False),
        (tuple[()], collections.abc.Sequence[int], True),
    ],
)
def test_subclass_sequence(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ((1, "a"), tuple[int, str], True),
        ((1, 2), tuple[int, str], False),
        ((1,), tuple[int, str], False),
        ((), tuple[()], True),
        ((1,), tuple[()], False),
        (P(1, "a"), tuple[int, str], True),
        ([1, "a"], tuple[int, str], False),
    ],
)
def test_instance_fixed(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ((), tuple[int, ...], True),
        ((1, 2, 3), tuple[int, ...], True),
        ((1, "a"), tuple[int, ...], False),
        ((True, 2), tuple[int, ...], True),
    ],
)
def test_instance_homogeneous(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ((1, 2), tuple[int, *tuple[str, ...], int], True),
        ((1, "a", "b", 2), tuple[int, *tuple[str, ...], int], True),
        ((1,), tuple[int, *tuple[str, ...], int], False),
    ],
)
def test_instance_unpacked(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (((1,), (2, 3)), tuple[tuple[int, ...], ...], True),
        (((1,), ("x",)), tuple[tuple[int, ...], ...], False),
        (((), ()), tuple[tp.Self, ...], True),
        (((), 1), tuple[tp.Self, ...], False),
    ],
)
def test_instance_nested(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected
