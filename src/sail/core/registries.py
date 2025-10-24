from typing import Callable, Dict

class Registry:
    def __init__(self) -> None:
        self._f: Dict[str, Callable] = {}
    def register(self, name: str):
        def deco(fn: Callable):
            if name in self._f:
                raise ValueError(f"Duplicate registry name: {name}")
            self._f[name] = fn
            return fn
        return deco
    def get(self, name: str) -> Callable:
        if name not in self._f:
            raise KeyError(f"Not found: {name}. Available: {list(self._f)}")
        return self._f[name]

ModelRegistry = Registry()
DatasetRegistry = Registry()
ExplainerRegistry = Registry()
