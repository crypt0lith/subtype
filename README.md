Provides a metaclass that extends runtime type checking to the full Python typing spec.

## Usage

### `isinstance`

`subtype.isinstance(obj, T)` returns `True` if `obj` is an instance of the type expression `T`, and a static checker narrows `obj` to `T` when it does. Use it to check values whose type is unknown at runtime.

```python
import subtype

assert subtype.isinstance([[1, 2], [3]], list[list[int]])
assert not subtype.isinstance([[1, 2], ["3"]], list[list[int]])
```

Where the builtin `isinstance` raises `TypeError` for a parameterized generic, `subtype.isinstance` checks the items of a collection against its type arguments, including nested collections.

```python
from collections.abc import Iterator

assert subtype.isinstance(zip(range(3), "abc"), Iterator[tuple[int, str]])
assert not subtype.isinstance(zip(range(3), "abc"), Iterator[tuple[int, int]])
```

`subtype.isinstance` infers the item type of an iterator such as `zip` from the arguments it was created with, without advancing it. An iterator it can't trace back to its arguments, such as a generator, is checked by its class only, and its items are not checked.

```python
from collections.abc import Callable
from typing import Protocol


class StrCall(Protocol):
    def __call__(self, o: object, /) -> str: ...


assert subtype.isinstance(str, StrCall)
assert subtype.isinstance(str, Callable[[object], str])
```

A class matches a callback protocol by its constructor, the same way it matches `Callable[...]`.

```python
from typing import runtime_checkable


@runtime_checkable
class GetsInt(Protocol):
    def get(self) -> int: ...


class StrGetter:
    def get(self) -> str:
        return ""


assert isinstance(StrGetter(), GetsInt)
assert not subtype.isinstance(StrGetter(), GetsInt)
```

Protocols are matched by the signatures of their members, where `runtime_checkable` only checks that the members exist.

### `subtype`

`subtype(T)` turns the type expression `T` into a class that `issubclass` accepts. `issubclass(S, subtype(T))` returns `True` if the type `S` is assignable to `T`.

```python
from collections.abc import Sequence

from subtype import subtype

assert issubclass(list[bool], subtype(Sequence[int]))
assert not issubclass(list[bool], subtype(list[int]))
```

`issubclass` accepts a parameterized generic as its first argument, and compares type arguments by the variance of their type parameters.

```python
from typing import Self

assert issubclass(str, subtype(Sequence[Self]))
```

`subtype` binds `Self` to the class being checked, so `str` is checked against `Sequence[str]`.

```python
from typing import overload


class Decode(Protocol):
    @overload
    def __call__(self, data: bytes) -> str: ...
    @overload
    def __call__(self, data: None) -> None: ...


assert issubclass(Decode, subtype(Callable[[bytes], str]))
assert not issubclass(Decode, subtype(Callable[[str], str]))
```

A callback protocol is checked against `Callable[...]` like any other callable, overloads included.

## Installation

Install the package via `pip`:

```shell
pip install subtype
```
