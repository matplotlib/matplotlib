from ._axes import Axes as Axes


# Backcompat.
Subplot = Axes

class _SubplotBaseMeta(type):
    def __instancecheck__(self, obj) -> bool: ...

class SubplotBase(metaclass=_SubplotBaseMeta): ...

def subplot_class_factory[T](cls: type[T]) -> type[T]: ...
