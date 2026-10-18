import collections.abc
import dataclasses
import typing as tp

import pytest

from subtype import subtype


class InitParam[T]:
    def __init__(self, item: T): ...


class Attribute[T]:
    item: T


@dataclasses.dataclass(frozen=True)
class FrozenDataclass[T]:
    item: T


class NamedTupleClass[T](tp.NamedTuple):
    item: T


class PrivateAttribute[T]:
    _item: T


class PrivateListAttribute[T]:
    _items: list[T]


class FinalAttribute[T]:
    item: tp.Final[T]


class Getter[T]:
    def get(self) -> T: ...


class CallbackGetter[T]:
    def get(self) -> collections.abc.Callable[[T], None]: ...


class Setter[T]:
    def set(self, x: T) -> None: ...


class ClassMethod[T]:
    @classmethod
    def make(cls, x: T) -> None: ...


class StaticMethod[T]:
    @staticmethod
    def f(x: T) -> None: ...


class Overload[T]:
    @tp.overload
    def f(self, x: T) -> None: ...
    @tp.overload
    def f(self, x: str) -> None: ...
    def f(self, x): ...


class OverloadOnly[T]:
    @tp.overload
    def f(self, x: T) -> None: ...
    @tp.overload
    def f(self, x: str) -> None: ...


class Property[T]:
    @property
    def item(self) -> T: ...


class SettableProperty[T]:
    @property
    def item(self) -> T: ...

    @item.setter
    def item(self, x: T) -> None: ...


class AttributeSubclass[T](Attribute[T]): ...


class GetterSubclass[T](Getter[T]): ...


class MutableSequenceSubclass[T](collections.abc.MutableSequence[T]): ...


class RecursiveReturn[T]:
    def next(self) -> RecursiveReturn[T]: ...


class RecursiveParam[T]:
    def merge(self, other: RecursiveParam[T]) -> None: ...


class MutualReturn[T]:
    def other(self) -> MutualParam[T]: ...

    def get(self) -> T: ...


class MutualParam[T]:
    def other(self, x: MutualReturn[T]) -> None: ...


class NestedParam[T]:
    def use(self, s: Setter[T]) -> None: ...


class SelfParam[T]:
    def get(self) -> T: ...

    def merge(self, other: tp.Self) -> None: ...


T_inf = tp.TypeVar("T_inf", infer_variance=True)


class LegacyTypeVar(tp.Generic[T_inf]):
    def get(self) -> T_inf: ...


class TwoParams[K, V]:
    def get(self, k: K) -> V: ...


@pytest.mark.parametrize(
    "cls, co, contra",
    [
        (Attribute, False, False),
        (FrozenDataclass, False, False),
        (NamedTupleClass, False, False),
    ],
)
def test_subclass_attribute(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


@pytest.mark.parametrize(
    "cls, co, contra",
    [(PrivateAttribute, True, False), (PrivateListAttribute, False, False)],
)
def test_subclass_private_attribute(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


def test_subclass_final_attribute():
    assert issubclass(FinalAttribute[bool], subtype(FinalAttribute[int]))
    assert not issubclass(FinalAttribute[int], subtype(FinalAttribute[bool]))


def test_subclass_init():
    assert issubclass(InitParam[bool], subtype(InitParam[int]))
    assert not issubclass(InitParam[int], subtype(InitParam[bool]))


@pytest.mark.parametrize(
    "cls, co, contra", [(Getter, True, False), (CallbackGetter, False, True)]
)
def test_subclass_method_return(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


@pytest.mark.parametrize(
    "cls, co, contra",
    [(Setter, False, True), (ClassMethod, False, True), (StaticMethod, False, True)],
)
def test_subclass_method_param(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


@pytest.mark.parametrize("cls", [Overload, OverloadOnly])
def test_subclass_overload(cls):
    assert not issubclass(cls[bool], subtype(cls[int]))
    assert issubclass(cls[int], subtype(cls[bool]))


@pytest.mark.parametrize(
    "cls, co, contra", [(Property, True, False), (SettableProperty, False, False)]
)
def test_subclass_property(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


@pytest.mark.parametrize(
    "cls, co, contra",
    [
        (AttributeSubclass, False, False),
        (GetterSubclass, True, False),
        (MutableSequenceSubclass, False, False),
    ],
)
def test_subclass_base_class(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


@pytest.mark.parametrize(
    "cls, co, contra",
    [
        (RecursiveReturn, True, False),
        (RecursiveParam, False, False),
        (MutualReturn, False, False),
        (MutualParam, False, True),
    ],
)
def test_subclass_recursive(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


def test_subclass_nested():
    assert issubclass(NestedParam[bool], subtype(NestedParam[int]))
    assert not issubclass(NestedParam[int], subtype(NestedParam[bool]))


def test_subclass_self_type():
    assert issubclass(SelfParam[bool], subtype(SelfParam[int]))
    assert not issubclass(SelfParam[int], subtype(SelfParam[bool]))


def test_subclass_legacy_typevar():
    assert issubclass(LegacyTypeVar[bool], subtype(LegacyTypeVar[int]))
    assert not issubclass(LegacyTypeVar[int], subtype(LegacyTypeVar[bool]))


def test_subclass_multiple_params():
    assert issubclass(TwoParams[int, bool], subtype(TwoParams[bool, int]))
    assert not issubclass(TwoParams[bool, bool], subtype(TwoParams[int, int]))


def test_instance_method():
    assert isinstance(Getter[bool](), subtype(Getter[int]))
    assert not isinstance(Setter[bool](), subtype(Setter[int]))
