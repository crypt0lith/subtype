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
type NestedInts = int | list[NestedInts]
T = tp.TypeVar("T", bound=int)
C = tp.TypeVar("C", int, str)
U = tp.TypeVar("U")
P = tp.ParamSpec("P")


@pytest.mark.parametrize("obj, dst", [(1, float), (1.0, complex)])
def test_instance_no_promotion(obj, dst):
    assert not isinstance(obj, subtype(dst))


@pytest.mark.parametrize(
    "obj, dst, expected", [(1, tp.Any, True), (1, tp.Never, False)]
)
def test_instance_any_never(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (None, None, True),
        (0, None, False),
        (None, tp.Optional[int], True),
        (1, int | str, True),
        (1.0, int | str, False),
    ],
)
def test_instance_union(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (1, tp.Literal[1], True),
        (True, tp.Literal[1], False),
        (1, tp.Literal[True], False),
        ("a", tp.Literal[1, "a"], True),
    ],
)
def test_instance_literal(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


def test_instance_annotated():
    assert isinstance(1, subtype(tp.Annotated[int, "x"]))


@pytest.mark.parametrize("obj, dst, expected", [(1, UserId, True), (1, AdminId, True)])
def test_instance_newtype(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        ("a", IntOrStr, True),
        (b"", IntOrStr, False),
        ((), Pair[int], True),
        (1, Pair[int], True),
        (1.0, Pair[int], False),
    ],
)
def test_instance_alias(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


def test_instance_forwardref():
    assert isinstance(1, subtype(tp.ForwardRef("int")))


@pytest.mark.parametrize(
    "obj, dst, expected",
    [([], collections.abc.Iterable, True), (1, collections.abc.Iterable, False)],
)
def test_instance_bare_abc(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected", [(bool, int, True), (int, float, False), (float, int, False)]
)
def test_subclass_nominal(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tp.Any, int, True),
        (tp.Never, int, True),
        (int, tp.Never, False),
        (tp.Never, tp.Never, True),
        (int, tp.Any, True),
    ],
)
def test_subclass_any_never(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (None, tp.Optional[int], True),
        (int | None, tp.Optional[int], True),
        (int | bytes, int | str, False),
        (bool | str, int | str, True),
    ],
)
def test_subclass_union(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (tp.Literal[1], int, True),
        (tp.Literal[1, "a"], int | str, True),
        (tp.Literal[1, b""], int | str, False),
        (tp.Literal[1], tp.Literal[1, 2], True),
        (tp.Literal[True], tp.Literal[1], False),
        (tp.Literal[1, 2], tp.Literal[1], False),
        (tp.Literal[1] | tp.Literal[2], tp.Literal[1, 2], True),
        (int, tp.Literal[1], False),
        (tp.Literal[None], None, True),
    ],
)
def test_subclass_literal(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


def test_subclass_annotated():
    assert issubclass(tp.Annotated[bool, "x"], subtype(int))


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (UserId, int, True),
        (AdminId, UserId, True),
        (AdminId, int, True),
        (int, UserId, False),
        (UserId, AdminId, False),
        (tp.Literal[1], UserId, False),
    ],
)
def test_subclass_newtype(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (IntOrStr, int | str | bytes, True),
        (IntOrStr, int, False),
        (NestedInts, NestedInts, True),
    ],
)
def test_subclass_alias(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (T, int, True),
        (T, str, False),
        (C, int | str, True),
        (C, int, False),
        (U, object, True),
        (U, int, False),
    ],
)
def test_subclass_typevar(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize(
    "obj, dst, expected",
    [
        (1, T, True),
        ("a", T, False),
        ("a", C, True),
        (b"", C, False),
        (object(), U, True),
    ],
)
def test_instance_typevar(obj, dst, expected):
    assert isinstance(obj, subtype(dst)) is expected


@pytest.mark.parametrize(
    "src, dst, expected",
    [(bool, T, True), (str, T, False), (str, C, True), (bytes, C, False)],
)
def test_subclass_typevar_dst(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


@pytest.mark.parametrize("obj, expected", [(bool, True), (str, False)])
def test_instance_type_typevar(obj, expected):
    assert isinstance(obj, subtype(type[T])) is expected


@pytest.mark.parametrize("typx", [T, P.args])
def test_type_parameter_form(typx):
    assert subtype(typx).__subtype_form__ is typx


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (list[int], collections.abc.Iterable, True),
        (list[int], collections.abc.Mapping, False),
        (tp.Callable[[int], int], collections.abc.Callable, True),
    ],
)
def test_subclass_bare_abc(src, dst, expected):
    assert issubclass(src, subtype(dst)) is expected


def test_subclass_forwardref():
    assert issubclass(tp.ForwardRef("bool"), subtype(int))


def test_not_implemented():
    with pytest.raises(NotImplementedError):
        subtype("int")


def test_cache():
    assert subtype(int) is subtype(int)
    assert subtype(tp.Annotated[int, "x"]) is subtype(int)
    assert subtype(None) is subtype(type(None))
    subtype(bytearray)
    gc.collect()
    assert bytearray not in _subtypes


@pytest.fixture
def undefined_ref():
    def f(x: Undefined): ...  # noqa: F821

    return (
        inspect.signature(f, annotation_format=annotationlib.Format.FORWARDREF)
        .parameters["x"]
        .annotation
    )


def test_annotationlib_forwardref(undefined_ref, monkeypatch):
    assert type(undefined_ref) is tp.ForwardRef
    assert undefined_ref.__ast_node__ is None
    monkeypatch.setitem(globals(), "Undefined", bytes)
    assert isinstance(b"", subtype(undefined_ref))
    assert not isinstance(1, subtype(undefined_ref))
