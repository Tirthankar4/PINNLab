# Visualisations package for PINN models
# This package contains visualization modules for different equation types

from .visualisation import plot_function, rel_misfit
from .visualisation_burgers import plot_burgers_solution, rel_misfit_burgers
from .visualisation_wave import plot_wave_solution, rel_misfit_wave, plot_wave_animation, plot_wave_energy_conservation

__all__ = [
    # Hydrodynamics visualizations
    'plot_function',
    'rel_misfit',
    
    # Burgers equation visualizations
    'plot_burgers_solution',
    'rel_misfit_burgers',
    
    # Wave equation visualizations
    'plot_wave_solution',
    'rel_misfit_wave',
    'plot_wave_animation',
    'plot_wave_energy_conservation'
] 