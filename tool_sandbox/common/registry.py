"""Registry mixin implementation."""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar, cast

T = TypeVar("T")


class RegistryMixin[T]:
    """Registry mixin that provides registry functionality for any class. Subclasses should define their own _registry."""

    _registry: ClassVar[dict[str, type[Any]]]

    def __init_subclass__(cls, **kwargs: int) -> None:
        """Called automatically when a class is defined."""
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "_registry"):
            cls._registry = {}
        cls._registry[cls.__name__] = cls

    @classmethod
    def get(cls, name: str) -> type[T]:
        """Get an registered instance of the class by name.

        Args:
            cls (Type[T]): The class to get the instance of
            name (str): The name of the class to get

        Returns:
            T: An instance of the class
        """
        if not hasattr(cls, "_registry"):
            raise ValueError(f"Class {cls.__name__} has no registry. Ensure it inherits from RegistryMixin.")

        registry = cast("dict[str, type[T]]", cls._registry)
        if name not in registry:
            raise ValueError(f"Class '{name}' not found in registry. Available classes: {cls.__name__}")
        return registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """Get a list of all available classes in the registry.

        Returns:
            list[str]: A list of all registered classes
        """
        if not hasattr(cls, "_registry"):
            raise ValueError(f"Class {cls.__name__} has no registry. Ensure it inherits from RegistryMixin.")

        return list(cls._registry.keys())
