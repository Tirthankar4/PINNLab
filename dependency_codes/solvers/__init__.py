# Solvers module - modular solver implementations for different PDE types

from .base import closure_batched, train_batched_with_progress
from .hydro import (
    input_taker as hydro_input_taker,
    req_consts_calc as hydro_req_consts_calc,
    process_batch as hydro_process_batch,
    closure_batched_hydro,
    train_batched_with_progress_hydro
)
from .burgers import (
    burgers_initial_condition,
    process_batch as burgers_process_batch,
    closure_batched_burgers,
    train_batched_with_progress_burgers
)
from .wave import (
    process_batch as wave_process_batch,
    closure_batched_wave,
    train_batched_with_progress_wave
)
# Import wave initial conditions from losses module for consistency
from losses.wave import wave_initial_condition_sine, wave_initial_velocity_zero
