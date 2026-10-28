"""t1seg: plug-and-play myocardial T1 mapping analysis. Importing the package registers all built-ins."""
from . import align, data, losses, models, postprocess, segmenters, t1fit  # noqa: F401
from .registry import ALIGNERS, DATASETS, FITTERS, LOSSES, MODELS, POSTPROCESS, SEGMENTERS, describe  # noqa: F401
