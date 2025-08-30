"""
core.utils.singleton_meta
=======================

Metaclass for implementing the Singleton pattern.
"""

class SingletonMeta(type):
    """
    Metaclass that implements the Singleton pattern.

    This metaclass ensures that only one instance of a class is created.
    All subsequent instantiations return the same instance.

    Example
    -------
    >>> class MyClass(metaclass=SingletonMeta):
    ...     pass
    >>> a = MyClass()
    >>> b = MyClass()
    >>> a is b  # True
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Return existing instance if it exists, otherwise create new instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
