from .core import SQLBear


try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
