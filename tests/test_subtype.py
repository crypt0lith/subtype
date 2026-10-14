import annotationlib
import collections.abc
import gc
import inspect
import typing as tp

import pytest

from subtypes import _subtypes, subtype

UserId = tp.NewType("UserId", int)
AdminId = tp.NewType("AdminId", UserId)
type IntOrStr = int | str
type Pair[T] = T | tuple
type Json = int | list[Json]
T = tp.TypeVar("T", bound=int)
C = tp.TypeVar("C", int, str)
U = tp.TypeVar("U")


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (1, tp.Any, True),
        (1, tp.Never, False),
        (1, tp.Literal[1], True),
        (True, tp.Literal[1], False),
        (1, tp.Literal[True], False),
        ("a", tp.Literal[1, "a"], True),
        (None, None, True),
        (0, None, False),
        (1, int | str, True),
        (1.0, int | str, False),
        (None, tp.Optional[int], True),
        (1, tp.Annotated[int, "x"], True),
        (1, float, True),
        (1.0, complex, True),
        ("a", float, False),
        (1, UserId, True),
        (1, AdminId, True),
        ("a", IntOrStr, True),
        (b"", IntOrStr, False),
        ((), Pair[int], True),
        (1, Pair[int], True),
        (1.0, Pair[int], False),
        (1, tp.ForwardRef("int"), True),
        ([], collections.abc.Iterable, True),
        (1, collections.abc.Iterable, False),
    ],
)
def test_instancecheck(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (bool, int, True),
        (int, float, True),
        (float, int, False),
        (tp.Any, int, True),
        (tp.Never, int, True),
        (int, tp.Never, False),
        (tp.Never, tp.Never, True),
        (int, tp.Any, True),
        (None, tp.Optional[int], True),
        (int | None, tp.Optional[int], True),
        (int | bytes, int | str, False),
        (bool | str, int | str, True),
        (tp.Literal[1], int, True),
        (tp.Literal[1, "a"], int | str, True),
        (tp.Literal[1, b""], int | str, False),
        (tp.Literal[1], tp.Literal[1, 2], True),
        (tp.Literal[True], tp.Literal[1], False),
        (tp.Literal[1, 2], tp.Literal[1], False),
        (tp.Literal[1] | tp.Literal[2], tp.Literal[1, 2], True),
        (int, tp.Literal[1], False),
        (tp.Literal[None], None, True),
        (tp.Annotated[bool, "x"], int, True),
        (UserId, int, True),
        (AdminId, UserId, True),
        (AdminId, int, True),
        (int, UserId, False),
        (UserId, AdminId, False),
        (tp.Literal[1], UserId, False),
        (IntOrStr, int | str | bytes, True),
        (IntOrStr, int, False),
        (Json, Json, True),
        (T, int, True),
        (T, str, False),
        (C, int | str, True),
        (C, int, False),
        (U, object, True),
        (U, int, False),
        (list[int], collections.abc.Iterable, True),
        (list[int], collections.abc.Mapping, False),
        (tp.Callable[[int], int], collections.abc.Callable, True),
        (tp.ForwardRef("bool"), int, True),
    ],
)
def test_subclasscheck(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "dst", [type[T], tp.Callable[tp.ParamSpec("P"), int], T, "int"]
)
def test_not_implemented(dst):
    with pytest.raises(NotImplementedError):
        subtype(dst)


def test_cache():
    assert subtype(int) is subtype(int)
    assert subtype(tp.Annotated[int, "x"]) is subtype(int)
    assert subtype(None) is subtype(type(None))
    subtype(bytearray)
    gc.collect()
    assert bytearray not in _subtypes


def test_annotationlib_forwardref():
    def f(x: Undefined): ...  # noqa: F821

    ref = (
        inspect.signature(f, annotation_format=annotationlib.Format.FORWARDREF)
        .parameters["x"]
        .annotation
    )
    assert type(ref) is tp.ForwardRef
    assert ref.__ast_node__ is None
    globals()["Undefined"] = bytes
    try:
        assert isinstance(b"", subtype(ref))
        assert not isinstance(1, subtype(ref))
    finally:
        del globals()["Undefined"]
