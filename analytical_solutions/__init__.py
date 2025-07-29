# Analytical Solutions Package
# This package contains analytical and numerical solution generators for different equation types

from .hydrodynamics import LAX
from .burgers import burgers_analytical
from .wave import wave_analytical

__all__ = [
    'LAX',
    'burgers_analytical', 
    'wave_analytical'
] 