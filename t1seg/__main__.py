"""python -m t1seg  -> list every registered component and the available devices."""
import sys

from . import describe
from .device import available_devices, resolve_device
from .registry import import_plugins

import_plugins(sys.argv[1:])
print(describe())
print(f"{'devices':12s} {available_devices()}  (auto -> {resolve_device('auto')})")
