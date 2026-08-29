from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import Callable, Dict, Any, Generic, TypeVar, Self, Union

from krrood.adapters.json_serializer import list_like_classes

T = TypeVar("T")


@dataclass
class HasFactoryAndKwargs(Generic[T]):
    """
    Mixing containing a hierarchy of factories and their keyword arguments.
    """

    factory: Callable[..., T]
    """
    The factory function to construct `T` with the keyword arguments.
    """

    kwargs: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    """
    The keyword arguments to pass to the factory.
    """

    def construct_instance(self):
        """
        Construct a python object from the CallableAndKwargs instance.

        ..note:: This method may work with ellipsis, but it's not guaranteed to work with all types.

        :return: The constructed object.
        """
        constructed_kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, list_like_classes):
                constructed_kwargs[key] = type(value)(
                    self._recurse_construct_instance_and_get_value(element)
                    for element in value
                )
            else:
                constructed_kwargs[key] = (
                    self._recurse_construct_instance_and_get_value(value)
                )
        return self.factory(**constructed_kwargs)

    def _recurse_construct_instance_and_get_value(self, value: Any):
        """
        Recursively construct an instance and return it.

        Most factories merely store whatever they are given, so a keyword argument
        still holding ``...`` -- something the caller has not determined yet --
        passes through unnoticed and construction succeeds, exactly as it always
        has. A factory that instead computes something from its arguments in its
        own constructor (for example a symbolic spatial type) cannot accept ``...``
        there and raises. Only then, and only when this value's own keyword
        arguments are the ones directly holding ``...`` (rather than some
        unrelated failure), is it left as ``None`` instead; any *other* value
        nested deeper in the same tree is unaffected and still constructed
        normally.

        :param value: The value to construct.
        :return: The constructed instance; ``None`` if constructing *value* failed
            because one of its own keyword arguments is still unresolved; or
            *value* unchanged if it is not itself constructible.
        :raises TypeError: If constructing *value* fails for a reason other than
            one of its own keyword arguments still being unresolved.
        """
        if isinstance(value, HasFactoryAndKwargs):
            try:
                return value.construct_instance()
            except TypeError:
                if any(kwarg_value is ... for kwarg_value in value.kwargs.values()):
                    return None
                raise
        return value

    def __deepcopy__(self, memo):
        return self.__class__(
            self.factory,
            kwargs={name: deepcopy(value) for name, value in self.kwargs.items()},
        )
