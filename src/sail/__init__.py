from ._version import __version__

# Import subpackages so registries are populated on import simba
from . import models as _models  # noqa: F401
from . import data as _data      # noqa: F401
from . import explain as _expl   # noqa: F401
