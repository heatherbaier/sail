# src/simba/models/__init__.py
from ..core.registries import ModelRegistry

# Import concrete model classes
from .resnet18 import ResNet18Classifier
from .tang2015 import Tang2015Classifier  # or Tang2015Wrapper if that's your class name
from .coordRegressor import CoordResNetRegressor  # or Tang2015Wrapper if that's your class name
from .coordconv_resnet import CoordConvResNetRegressor  # noqa: F401
from .biasfield_resnet import BiasFieldResNetRegressor  # noqa: F401
from .geoconv import GeoConvNativeRegressor  # noqa: F401


# Register them under CLI-friendly names
ModelRegistry.register("resnet18")(ResNet18Classifier)
ModelRegistry.register("tang2015")(Tang2015Classifier)
ModelRegistry.register("r18_wc")(CoordResNetRegressor)
ModelRegistry.register("coordConv")(CoordConvResNetRegressor)
ModelRegistry.register("biasField")(BiasFieldResNetRegressor)
ModelRegistry.register("geoconv")(GeoConvNativeRegressor)