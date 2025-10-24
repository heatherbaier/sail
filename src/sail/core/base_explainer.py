from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseExplainer(ABC):
    @abstractmethod
    def global_sensitivity(self, **kwargs) -> Dict[str, Any]: ...
    @abstractmethod
    def local_sensitivity(self, batch, **kwargs) -> Dict[str, Any]: ...
