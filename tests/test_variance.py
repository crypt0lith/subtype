import collections.abc
import dataclasses
import typing as tp

import pytest

from subtypes import subtype


class Unused[T]:
    def __init__(self, item: T): ...


class Attr[T]:
    item: T


class Private[T]:
    _item: T


class FinalAttr[T]:
    item: tp.Final[T]


class Getter[T]:
    def get(self) -> T: ...


class Setter[T]:
    def set(self, x: T) -> None: ...


class Prop[T]:
    @property
    def item(self) -> T: ...


class PropSet[T]:
    @property
    def item(self) -> T: ...

    @item.setter
    def item(self, x: T) -> None: ...


@dataclasses.dataclass(frozen=True)
class Frozen[T]:
    item: T


class NT[T](tp.NamedTuple):
    item: T


class SubAttr[T](Attr[T]): ...


class SubSeq[T](collections.abc.Sequence[T]): ...


class SubGetter[T](Getter[T]): ...


class Cyc1[T]:
    def other(self) -> Cyc2[T]: ...

    def get(self) -> T: ...


class Cyc2[T]:
    def other(self, x: Cyc1[T]) -> None: ...


class Chain[T]:
    def next(self) -> Chain[T]: ...


class Merge[T]:
    def merge(self, other: Merge[T]) -> None: ...


class SelfParam[T]:
    def get(self) -> T: ...

    def merge(self, other: tp.Self) -> None: ...


class SelfRet[T]:
    def get(self) -> T: ...

    def copy(self) -> tp.Self: ...


class Sink[T]:
    def put(self, x: T) -> None: ...


class SinkUser[T]:
    def use(self, s: Sink[T]) -> None: ...


class Two[K, V]:
    def get(self, k: K) -> V: ...


class StaticM[T]:
    @staticmethod
    def f(x: int) -> None: ...

    def get(self) -> T: ...


class Over[T]:
    @tp.overload
    def f(self, x: int) -> T: ...
    @tp.overload
    def f(self, x: str) -> None: ...
    def f(self, x): ...


class Cls[T]:
    @classmethod
    def make(cls, x: T) -> None: ...


class ListAttr[T]:
    _items: list[T]


class MethodVar[T]:
    def f[S](self, x: S) -> collections.abc.Mapping[S, T]: ...


T_inf = tp.TypeVar("T_inf", infer_variance=True)


class OldStyle(tp.Generic[T_inf]):
    def get(self) -> T_inf: ...


# expectations match pyright 1.1.414
@pytest.mark.parametrize(
    "cls, co, contra",
    [
        (Unused, True, False),
        (Attr, False, False),
        (Private, True, False),
        (FinalAttr, True, False),
        (Getter, True, False),
        (Setter, False, True),
        (Prop, True, False),
        (PropSet, False, False),
        (Frozen, False, False),
        (NT, False, False),
        (SubAttr, False, False),
        (SubSeq, True, False),
        (SubGetter, True, False),
        (Cyc1, False, False),
        (Cyc2, False, True),
        (Chain, True, False),
        (Merge, False, False),
        (SelfParam, True, False),
        (SelfRet, True, False),
        (SinkUser, True, False),
        (StaticM, True, False),
        (Over, True, False),
        (Cls, False, True),
        (ListAttr, False, False),
        (MethodVar, True, False),
        (OldStyle, True, False),
    ],
)
def test_inferred(cls, co, contra):
    assert issubclass(cls[bool], subtype(cls[int])) is co
    assert issubclass(cls[int], subtype(cls[bool])) is contra


def test_multiple_params():
    assert issubclass(Two[int, bool], subtype(Two[bool, int]))
    assert not issubclass(Two[bool, bool], subtype(Two[int, int]))


def test_instance():
    assert isinstance(Getter[bool](), subtype(Getter[int]))
    assert not isinstance(Setter[bool](), subtype(Setter[int]))
