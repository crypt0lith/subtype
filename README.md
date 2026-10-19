Provides a metaclass that extends runtime type checking to the full Python typing spec.

## Usage

### `isinstance`

`subtype.isinstance(obj, T)` answers whether `obj` is assignable to the type expression `T` under the typing spec, and narrows `obj` to `T` for a static checker when it is. Code can then enforce its own annotations on values that only exist at runtime, where a static checker can't see them.

```python
import subtype

assert subtype.isinstance([[1, 2], [3]], list[list[int]])
assert not subtype.isinstance([[1, 2], ["3"]], list[list[int]])
```

Where the builtin `isinstance` raises `TypeError` for a parameterized generic, `subtype.isinstance` checks the items of a collection against its type arguments, including nested collections.

```python
from collections.abc import Iterator

assert subtype.isinstance(zip(range(3), "abc"), Iterator[tuple[int, str]])
```

`subtype.isinstance` infers an iterator's item type from the objects the iterator was built from, without advancing it.

```python
from collections.abc import Callable
from typing import Protocol


class StrCall(Protocol):
    def __call__(self, o: object, /) -> str:
        return str(o)


assert subtype.isinstance(str, StrCall)
assert subtype.isinstance(str, Callable[[object], str])
```

A class matches a `__call__` protocol by its constructor, the same way it matches `Callable[...]`.

### `subtype`

`subtype(T)` turns the type expression `T` into a class that `issubclass` accepts, and answers whether a type is assignable to `T` under the typing spec.

```python
from collections.abc import Sequence

from subtype import subtype

assert issubclass(list[bool], subtype(Sequence[int]))
assert not issubclass(list[bool], subtype(list[int]))
```

`issubclass` takes a type expression as its first argument, and compares type arguments by the variance of their type parameters.

```python
from typing import Self

assert issubclass(str, subtype(Sequence[Self]))
```

`subtype` binds `Self` to the class being checked, so `str` is checked against `Sequence[str]`.

```python
from collections.abc import Callable
from typing import Protocol, overload


class Decode(Protocol):
    @overload
    def __call__(self, data: bytes) -> str: ...
    @overload
    def __call__(self, data: None) -> None: ...


assert issubclass(Decode, subtype(Callable[[bytes], str]))
```

Protocols are matched by the signatures of their members, not only by member names as with `runtime_checkable`. A protocol's `__call__` is checked against `Callable[...]` like any other callable, overloads included.

## Installation

Install the package via `pip`:

```shell
pip install subtype
```
