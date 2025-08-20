from flask import Flask, render_template, request, jsonify, Response
import torch
import numpy as np

# Configure matplotlib backend first with comprehensive warning suppression
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

# Ensure matplotlib is properly configured
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['backend'] = 'Agg'

# Comprehensive matplotlib warning suppression
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')
warnings.filterwarnings('ignore', message='.*and thus cannot be shown.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.backends.backend_agg')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.figure')

# Set matplotlib to be completely silent about backend issues
matplotlib.set_loglevel('error')

import io
import base64
import time
import json
import threading
from torch.autograd import Variable
from scipy.interpolate import interp1d
import os # Added for file existence check

# Import your existing code
from dependency_codes.model_architecture import PINN
from models.burgers import BurgersPINN, BurgersSimplePINN
from models.hydro import HydroPINN, HydroSimplePINN
from models.wave import WavePINN, WaveSimplePINN
from models.SHM import SHMPINN, SHMSimplePINN
from visualisations import plot_function, rel_misfit, plot_burgers_solution, rel_misfit_burgers, plot_wave_solution, rel_misfit_wave
from visualisations.visualisation_SHM import plot_SHM_solution, rel_misfit_SHM
from config import xmin, tmin, rho_o, HOST, PORT, DEBUG, LOG_LEVEL, LOG_FILE, MAX_FILE_SIZE
from analytical_solutions.hydrodynamics import LAX
from analytical_solutions.wave import wave_analytical_solution_dalembert
from losses.wave import wave_initial_condition_sine, wave_initial_condition_gaussian

app = Flask(__name__)

# Configure logging with a completely safe approach
import logging
import logging.handlers
import os
import sys
import threading

# Create logs directory if it doesn't exist
os.makedirs('logging/logs', exist_ok=True)

# Create a completely safe logging system
class SafeLogger:
    """A thread-safe logger that never fails due to buffer issues"""
    
    def __init__(self, name, log_file):
        self.name = name
        self.log_file = log_file
        self.lock = threading.Lock()
        
        # Try to set up file logging
        try:
            self.file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            self.file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        except Exception:
            self.file_handler = None
    
    def _safe_write(self, level, message):
        """Safely write to both file and console"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"{timestamp} - {self.name} - {level.upper()} - {message}"
        
        # Try file logging first
        if self.file_handler:
            try:
                with self.lock:
                    self.file_handler.emit(logging.LogRecord(
                        name=self.name,
                        level=getattr(logging, level.upper()),
                        pathname='',
                        lineno=0,
                        msg=message,
                        args=(),
                        exc_info=None
                    ))
            except Exception:
                pass  # File logging failed, continue to console
        
        # Always try console output
        try:
            print(formatted_message)
        except Exception:
            # Last resort - try stderr
            try:
                print(formatted_message, file=sys.stderr)
            except:
                pass  # If even stderr fails, we can't do anything
    
    def info(self, message, *args, **kwargs):
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self._safe_write('info', message)
    
    def warning(self, message, *args, **kwargs):
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self._safe_write('warning', message)
    
    def error(self, message, *args, **kwargs):
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self._safe_write('error', message)
    
    def debug(self, message, *args, **kwargs):
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self._safe_write('debug', message)

# Create our safe logger instance
logger = SafeLogger(__name__, LOG_FILE)

# Global exception handler to catch any remaining logging issues
import sys
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow keyboard interrupts to pass through
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # For any other exception, just print it safely
    try:
        print(f"Unhandled exception: {exc_type.__name__}: {exc_value}")
    except:
        pass  # If even print fails, we can't do anything

# Set the global exception handler
sys.excepthook = handle_exception

# Suppress Flask request logs but keep startup messages
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.setLevel(logging.WARNING)  # Keep WARNING and ERROR, suppress INFO (request logs)

# Suppress matplotlib warnings about non-interactive backend
matplotlib_log = logging.getLogger('matplotlib')
matplotlib_log.setLevel(logging.ERROR)  # Only show errors, suppress warnings

# Our SafeLogger handles all logging safely - no need for safe_log wrapper

# Also suppress the specific matplotlib warning about non-interactive backend
import warnings
warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')
warnings.filterwarnings('ignore', message='.*and thus cannot be shown.*')

# Progress tracking
training_progress = {}
progress_lock = threading.Lock()

# Training stop flags
training_stop_flags = {}
stop_flags_lock = threading.Lock()

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Available pretrained models with their parameter ranges
PRETRAINED_MODELS = {
    'hydro_case1': {
        'file': 'pretrained_models/hydrodynamics/Case1_final_part1.pth',
        'type': 'hydro',
        'description': 'Hydrodynamics Case 1',
        'lam': 7.0,
        'alpha_min': 0.01,
        'alpha_max': 0.08,
        'num_of_waves': 2
    },
    'hydro_case2': {
        'file': 'pretrained_models/hydrodynamics/Case2_final_part1.pth', 
        'type': 'hydro',
        'description': 'Hydrodynamics Case 2',
        'lam': 7.0,
        'alpha_min': 0.1,
        'alpha_max': 0.8,
        'num_of_waves': 2
    },
    'hydro_case3': {
        'file': 'pretrained_models/hydrodynamics/Case3_final_part1.pth',
        'type': 'hydro', 
        'description': 'Hydrodynamics Case 3',
        'lam': 5.0,
        'alpha_min': 0.01,
        'alpha_max': 0.08,
        'num_of_waves': 2
    },
    'mixed': {
        'file': 'pretrained_models/hydrodynamics/mixed_final_part1.pth',
        'type': 'mixed',
        'description': 'Mixed Model',
        'lam': 7.0,
        'alpha_min': 0.09,
        'alpha_max': 0.11,
        'num_of_waves': 2
    },
}

# Load models into memory
loaded_models = {}
loaded_models_info = {}

def load_pretrained_model(model_name):
    """Load a pretrained model from disk"""
    try:
        # Our SafeLogger is completely safe
        logger.info(f"Loading pretrained model: {model_name}")
        
        # Get model info from PRETRAINED_MODELS
        if model_name not in PRETRAINED_MODELS:
            # Our SafeLogger is completely safe
            logger.warning(f"Model {model_name} not found in PRETRAINED_MODELS")
            return None, f"Model {model_name} not found in PRETRAINED_MODELS"
        
        model_info = PRETRAINED_MODELS[model_name]
        model_path = model_info['file']
        
        if not os.path.exists(model_path):
            return None, f"Model file {model_path} not found"
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if checkpoint is just the state dict or has metadata
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            model_type = checkpoint.get('model_type', model_info['type'])
            config = checkpoint.get('config', {})
            state_dict = checkpoint['model_state_dict']
            use_legacy = False  # New models use new architecture
        else:
            # Old format - checkpoint is just the state dict (pretrained models)
            model_type = model_info['type']
            config = {}
            state_dict = checkpoint
            use_legacy = True  # Pretrained models use legacy architecture
        
        # Our SafeLogger is completely safe
        logger.info(f"Loading {model_type} model with config: {config} (legacy: {use_legacy})")
        
        # Create model
        model = create_new_model(model_type, config, use_legacy=use_legacy)
        if model is None:
            # Our SafeLogger is completely safe
            logger.error(f"Failed to create {model_type} model")
            return None, f"Failed to create {model_type} model"
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Model loaded successfully
        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        # Our SafeLogger is completely safe
        logger.info(f"Model loaded successfully with {total_params} trainable parameters")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model evaluation mode: {model.training}")
        
        return model, None
        
    except Exception as e:
        # Our SafeLogger is completely safe
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None, f"Error loading model: {str(e)}"

def create_new_model(model_type, config, use_legacy=False):
    """Create a new model with custom architecture"""
    # For SHM, use SimplePINN by default since it's a simpler equation
    if model_type == 'SHM':
        default_architecture = 'SimplePINN'
    else:
        default_architecture = 'P2PINN'  # Default to P2PINN for other equations
        
    architecture = config.get('architecture', default_architecture)
    
    if use_legacy:
        # Use legacy PINN architecture for pretrained models
        from dependency_codes.model_architecture import PINN
        model = PINN(num_neurons=config.get('num_neurons', 96))
    elif architecture == 'SimplePINN':
        # Use SimplePINN architecture
        if model_type == 'burgers':
            model = BurgersSimplePINN(
                num_neurons=config.get('num_neurons', 96),
                use_param_embedding=config.get('use_param_embedding', True),
                num_layers=config.get('main_pinn_layers', 5)
            )
        elif model_type == 'hydro':
            model = HydroSimplePINN(
                num_neurons=config.get('num_neurons', 96),
                use_param_embedding=config.get('use_param_embedding', True),
                num_layers=config.get('main_pinn_layers', 5)
            )
        elif model_type == 'wave':
            model = WaveSimplePINN(
                num_neurons=config.get('num_neurons', 96),
                use_param_embedding=config.get('use_param_embedding', True),
                num_layers=config.get('main_pinn_layers', 5)
            )
        elif model_type == 'SHM':
            model = SHMSimplePINN(
                num_neurons=config.get('num_neurons', 96),
                use_param_embedding=config.get('use_param_embedding', True),
                num_layers=config.get('main_pinn_layers', 5)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # Use P2PINN architecture (default)
        if model_type == 'burgers':
            model = BurgersPINN(
                num_neurons=config.get('num_neurons', 96),
                param_embed_activation=config.get('param_embed_activation', 'tanh'),
                input_embed_activation=config.get('input_embed_activation', 'sin'),
                main_pinn_activation=config.get('main_pinn_activation', 'sin'),
                param_embed_layers=config.get('param_embed_layers', 2),
                param_embed_neurons=config.get('param_embed_neurons', 32),
                input_embed_layers=config.get('input_embed_layers', 3),
                input_embed_neurons=config.get('input_embed_neurons', 64),
                main_pinn_layers=config.get('main_pinn_layers', 5)
            )
        elif model_type == 'hydro':
            model = HydroPINN(
                num_neurons=config.get('num_neurons', 96),
                param_embed_activation=config.get('param_embed_activation', 'tanh'),
                input_embed_activation=config.get('input_embed_activation', 'sin'),
                main_pinn_activation=config.get('main_pinn_activation', 'sin'),
                param_embed_layers=config.get('param_embed_layers', 2),
                param_embed_neurons=config.get('param_embed_neurons', 32),
                input_embed_layers=config.get('input_embed_layers', 3),
                input_embed_neurons=config.get('input_embed_neurons', 64),
                main_pinn_layers=config.get('main_pinn_layers', 5)
            )
        elif model_type == 'wave':
            model = WavePINN(
                num_neurons=config.get('num_neurons', 96),
                param_embed_activation=config.get('param_embed_activation', 'tanh'),
                input_embed_activation=config.get('input_embed_activation', 'sin'),
                main_pinn_activation=config.get('main_pinn_activation', 'sin'),
                param_embed_layers=config.get('param_embed_layers', 2),
                param_embed_neurons=config.get('param_embed_neurons', 32),
                input_embed_layers=config.get('input_embed_layers', 3),
                input_embed_neurons=config.get('input_embed_neurons', 64),
                main_pinn_layers=config.get('main_pinn_layers', 5)
            )
        elif model_type == 'SHM':
            model = SHMPINN(
                num_neurons=config.get('num_neurons', 96),
                param_embed_activation=config.get('param_embed_activation', 'tanh'),
                input_embed_activation=config.get('input_embed_activation', 'sin'),
                main_pinn_activation=config.get('main_pinn_activation', 'sin'),
                param_embed_layers=config.get('param_embed_layers', 2),
                param_embed_neurons=config.get('param_embed_neurons', 32),
                input_embed_layers=config.get('input_embed_layers', 3),
                input_embed_neurons=config.get('input_embed_neurons', 64),
                main_pinn_layers=config.get('main_pinn_layers', 5)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    return model

def train_model_with_progress(training_id, model_type, config, training_config, model_params, save_model=False):
    """Train a PINN model with progress tracking - using real solver modules"""
    
    # Our SafeLogger is completely safe
    logger.info(f"Starting real training for {model_type} model (ID: {training_id})")
    import time
    start_time = time.time()
    
    # Create model
    model = create_new_model(model_type, config)
    
    # Import the appropriate solver based on model type
    training_param_value = None
    if model_type == 'SHM':
        from dependency_codes.solvers.SHM import train_batched_with_progress_SHM
        from dependency_codes.solvers.SHM import process_batch
        from dependency_codes.solvers.SHM import closure_batched_SHM
        from losses.SHM import SHM_loss, SHM_initial_condition, SHM_initial_velocity
    elif model_type == 'hydro':
        from dependency_codes.solvers.hydro import train_batched_with_progress_hydro
        from dependency_codes.solvers.hydro import process_batch as process_batch_hydro
        from dependency_codes.solvers.hydro import req_consts_calc
        from dependency_codes.solvers.hydro import closure_batched_hydro
        from losses.hydrodynamic import hydrodynamical_equations_loss
    elif model_type == 'burgers':
        from dependency_codes.solvers.burgers import train_batched_with_progress_burgers
        from dependency_codes.solvers.burgers import process_batch as process_batch_burgers
        from dependency_codes.solvers.burgers import closure_batched_burgers
        from losses.burgers import burgers_loss
    elif model_type == 'wave':
        from dependency_codes.solvers.wave import train_batched_with_progress_wave
        from dependency_codes.solvers.wave import process_batch as process_batch_wave
        from dependency_codes.solvers.wave import closure_batched_wave
        from losses.wave import wave_loss
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Set up training parameters
    device = next(model.parameters()).device
    
    # Create collocation points based on model type
    if model_type == 'SHM':
        # SHM: time domain from 0 to tmax
        tmax = model_params.get('tmax', 2.0)
        N_0 = model_params.get('N_0', 100)
        N_r = model_params.get('N_r', 1000)
        
        # Create time coordinates
        t_domain = torch.linspace(0, tmax, N_r, device=device).reshape(-1, 1)
        # Initial condition should be enforced at t=0 only
        t_ic = torch.zeros(N_0, 1, device=device)
        
        collocation_domain = [t_domain]
        collocation_IC = [t_ic]
        
        # SHM parameter(s): spring constant c
        c_single_shm = model_params.get('c_single_shm', None)
        if c_single_shm is not None:
            try:
                c_value_shm = float(c_single_shm)
            except Exception:
                c_value_shm = 1.0
            alpha = torch.tensor([c_value_shm], device=device, dtype=torch.float32).view(-1, 1)
            training_param_value = float(c_value_shm)
        else:
            c_min_shm = float(model_params.get('c_min_shm', 1.0))
            c_max_shm = float(model_params.get('c_max_shm', 9.0))
            c_N_shm = int(model_params.get('c_N_shm', 5))
            if c_N_shm <= 1 or abs(c_max_shm - c_min_shm) < 1e-12:
                alpha = torch.tensor([c_min_shm], device=device, dtype=torch.float32).view(-1, 1)
            else:
                step = (c_max_shm - c_min_shm) / max(c_N_shm - 1, 1)
                eps = step / 2.0
                c_vals_shm = torch.arange(c_min_shm, c_max_shm + eps, step, device=device, dtype=torch.float32)
                if c_vals_shm.shape[0] > c_N_shm:
                    c_vals_shm = c_vals_shm[:c_N_shm]
                elif c_vals_shm.shape[0] < c_N_shm:
                    pad = torch.full((c_N_shm - c_vals_shm.shape[0],), c_max_shm, device=device, dtype=torch.float32)
                    c_vals_shm = torch.cat([c_vals_shm, pad], dim=0)
                alpha = c_vals_shm.view(-1, 1)
        
        # Use SHM-specific training function
        train_function = train_batched_with_progress_SHM
        process_batch_func = process_batch
        
    elif model_type == 'hydro':
        # Hydro: spatial domain from xmin to xmax, time from 0 to tmax
        xmin, xmax = 0.0, 14.0
        tmax = model_params.get('tmax', 2.0)
        N_0 = model_params.get('N_0', 100)
        N_b = model_params.get('N_b', 100)
        N_r = model_params.get('N_r', 1000)
        
        # Create spatial and time coordinates
        x_domain = torch.linspace(xmin, xmax, N_r, device=device).reshape(-1, 1)
        t_domain = torch.linspace(0, tmax, N_r, device=device).reshape(-1, 1)
        x_ic = torch.linspace(xmin, xmax, N_0, device=device).reshape(-1, 1)
        t_ic = torch.zeros(N_0, 1, device=device)
        x_bc = torch.tensor([xmin, xmax], device=device).reshape(-1, 1)
        t_bc = torch.linspace(0, tmax, N_b, device=device).reshape(-1, 1)
        
        collocation_domain = [x_domain, t_domain]
        collocation_IC = [x_ic, t_ic]
        collocation_BC = [x_bc, t_bc]
        
        # Hydro parameters: wavelength, Jeans length, velocity
        lam = 7.0  # Default wavelength
        jeans, alpha_const = req_consts_calc(lam)

        # Generate list of parameter values (alpha_list) and corresponding v_1 values
        alpha_single = model_params.get('alpha_single', None)
        if alpha_single is not None:
            try:
                alpha_value = float(alpha_single)
            except Exception:
                alpha_value = 0.05
            alpha_list = torch.tensor([alpha_value], device=device, dtype=torch.float32).view(-1, 1)
            training_param_value = float(alpha_value)
        else:
            alpha_min = float(model_params.get('alpha_min', 0.01))
            alpha_max = float(model_params.get('alpha_max', 0.1))
            alpha_N = int(model_params.get('alpha_N', 5))

            if alpha_N <= 1 or abs(alpha_max - alpha_min) < 1e-12:
                alpha_list = torch.tensor([alpha_min], device=device, dtype=torch.float32).view(-1, 1)
            else:
                step = (alpha_max - alpha_min) / max(alpha_N - 1, 1)
                # Add small epsilon to include alpha_max due to float precision
                eps = step / 2.0
                alpha_vals = torch.arange(alpha_min, alpha_max + eps, step, device=device, dtype=torch.float32)
                if alpha_vals.shape[0] > alpha_N:
                    alpha_vals = alpha_vals[:alpha_N]
                elif alpha_vals.shape[0] < alpha_N:
                    pad = torch.full((alpha_N - alpha_vals.shape[0],), alpha_max, device=device, dtype=torch.float32)
                    alpha_vals = torch.cat([alpha_vals, pad], dim=0)
                alpha_list = alpha_vals.view(-1, 1)

        # v_1 = (alpha/(rho_o * 2*pi/lam)) * rho_1, where rho_1 spans alpha_list
        factor = torch.tensor(alpha_const, device=device, dtype=torch.float32) / (rho_o * 2 * np.pi / lam)
        v_1 = factor * alpha_list  # Shape (N, 1)

        # Use alpha_list as the parameter set for training (shape (N,1))
        alpha = alpha_list
        
        # Use hydro-specific training function
        train_function = train_batched_with_progress_hydro
        process_batch_func = process_batch_hydro
        
    elif model_type == 'burgers':
        # Burgers: spatial domain from xmin to xmax, time from 0 to tmax
        xmax_user = model_params.get('xmax', 1.0)
        xmin, xmax = -float(xmax_user), float(xmax_user)
        tmax = model_params.get('tmax', 1.0)
        N_0 = model_params.get('N_0', 100)
        N_b = model_params.get('N_b', 100)
        N_r = model_params.get('N_r', 1000)
        
        # Create spatial and time coordinates
        x_domain = torch.linspace(xmin, xmax, N_r, device=device).reshape(-1, 1)
        t_domain = torch.linspace(0, tmax, N_r, device=device).reshape(-1, 1)
        x_ic = torch.linspace(xmin, xmax, N_0, device=device).reshape(-1, 1)
        t_ic = torch.zeros(N_0, 1, device=device)
        x_bc = torch.tensor([xmin, xmax], device=device).reshape(-1, 1)
        t_bc = torch.linspace(0, tmax, N_b, device=device).reshape(-1, 1)
        
        collocation_domain = [x_domain, t_domain]
        collocation_IC = [x_ic, t_ic]
        collocation_BC = [x_bc, t_bc]
        
        # Burgers parameter(s): viscosity nu
        nu_single = model_params.get('nu_single', None)
        if nu_single is not None:
            try:
                nu_value = float(nu_single)
            except Exception:
                nu_value = 0.01
            alpha = torch.tensor([nu_value], device=device, dtype=torch.float32).view(-1, 1)
            training_param_value = float(nu_value)
        else:
            nu_min = float(model_params.get('nu_min', 0.01))
            nu_max = float(model_params.get('nu_max', 0.05))
            nu_N = int(model_params.get('nu_N', 5))
            if nu_N <= 1 or abs(nu_max - nu_min) < 1e-12:
                alpha = torch.tensor([nu_min], device=device, dtype=torch.float32).view(-1, 1)
            else:
                step = (nu_max - nu_min) / max(nu_N - 1, 1)
                eps = step / 2.0
                nu_vals = torch.arange(nu_min, nu_max + eps, step, device=device, dtype=torch.float32)
                if nu_vals.shape[0] > nu_N:
                    nu_vals = nu_vals[:nu_N]
                elif nu_vals.shape[0] < nu_N:
                    pad = torch.full((nu_N - nu_vals.shape[0],), nu_max, device=device, dtype=torch.float32)
                    nu_vals = torch.cat([nu_vals, pad], dim=0)
                alpha = nu_vals.view(-1, 1)
        
        # Use burgers-specific training function
        train_function = train_batched_with_progress_burgers
        process_batch_func = process_batch_burgers
        
    elif model_type == 'wave':
        # Wave: spatial domain from xmin to xmax, time from 0 to tmax
        xmax_user = model_params.get('xmax', 1.0)
        xmin, xmax = 0.0, float(xmax_user)
        tmax = model_params.get('tmax', 1.0)
        N_0 = model_params.get('N_0', 100)
        N_b = model_params.get('N_b', 100)
        N_r = model_params.get('N_r', 1000)
        
        # Create spatial and time coordinates
        x_domain = torch.linspace(xmin, xmax, N_r, device=device).reshape(-1, 1)
        t_domain = torch.linspace(0, tmax, N_r, device=device).reshape(-1, 1)
        x_ic = torch.linspace(xmin, xmax, N_0, device=device).reshape(-1, 1)
        t_ic = torch.zeros(N_0, 1, device=device)
        x_bc = torch.tensor([xmin, xmax], device=device).reshape(-1, 1)
        t_bc = torch.linspace(0, tmax, N_b, device=device).reshape(-1, 1)
        
        collocation_domain = [x_domain, t_domain]
        collocation_IC = [x_ic, t_ic]
        collocation_BC = [x_bc, t_bc]
        
        # Wave parameter(s): wave speed c
        c_single = model_params.get('c_single', None)
        if c_single is not None:
            try:
                c_value = float(c_single)
            except Exception:
                c_value = 1.0
            alpha = torch.tensor([c_value], device=device, dtype=torch.float32).view(-1, 1)
            training_param_value = float(c_value)
        else:
            c_min = float(model_params.get('c_min', 0.5))
            c_max = float(model_params.get('c_max', 2.0))
            c_N = int(model_params.get('c_N', 5))
            if c_N <= 1 or abs(c_max - c_min) < 1e-12:
                alpha = torch.tensor([c_min], device=device, dtype=torch.float32).view(-1, 1)
            else:
                step = (c_max - c_min) / max(c_N - 1, 1)
                eps = step / 2.0
                c_vals = torch.arange(c_min, c_max + eps, step, device=device, dtype=torch.float32)
                if c_vals.shape[0] > c_N:
                    c_vals = c_vals[:c_N]
                elif c_vals.shape[0] < c_N:
                    pad = torch.full((c_N - c_vals.shape[0],), c_max, device=device, dtype=torch.float32)
                    c_vals = torch.cat([c_vals, pad], dim=0)
                alpha = c_vals.view(-1, 1)
        
        # Use wave-specific training function
        train_function = train_batched_with_progress_wave
        process_batch_func = process_batch_wave
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    optimizerL = torch.optim.LBFGS(model.parameters(), max_iter=20)
    
    # Create loss function
    mse_cost_function = torch.nn.MSELoss()
    
    # Progress update function with loss plotting
    def update_progress(training_id, progress_data):
        with progress_lock:
            if training_id in training_progress:
                training_progress[training_id].update(progress_data)
                
                # If loss info is available, log it for plotting
                if 'loss_info' in progress_data:
                    # Our SafeLogger is completely safe
                    logger.info(f"Training Loss Update - {progress_data['loss_info']}")
    
    # Start real training
    # Our SafeLogger is completely safe
    logger.info(f"Starting real {model_type} training with {training_config['iteration_adam']} Adam + {training_config['iteration_lbfgs']} L-BFGS iterations")
    
    try:
        # Call the appropriate training function based on model type
        if model_type == 'SHM':
            trained_model = train_function(
                training_id, model, model, alpha, collocation_domain, collocation_IC,
                optimizer, optimizerL, mse_cost_function,
                training_config['iteration_adam'], training_config['iteration_lbfgs'],
                training_config['batch_size'], training_config['num_batches'],
                update_progress
            )
            
        elif model_type == 'hydro':
            # Hydro needs additional parameters - use the tensor versions defined earlier
            # lam, jeans, alpha_val, and v_1 are already defined above
            
            trained_model = train_function(
                training_id, model, model, alpha, collocation_domain, collocation_IC,
                optimizer, optimizerL, mse_cost_function,
                training_config['iteration_adam'], training_config['iteration_lbfgs'],
                lam, jeans, v_1, training_config['batch_size'], training_config['num_batches'],
                update_progress
            )
            
        elif model_type == 'burgers':
            trained_model = train_function(
                training_id, model, model, alpha, collocation_domain, collocation_IC,
                optimizer, optimizerL, mse_cost_function,
                training_config['iteration_adam'], training_config['iteration_lbfgs'],
                training_config['batch_size'], training_config['num_batches'],
                update_progress
            )
            
        elif model_type == 'wave':
            trained_model = train_function(
                training_id, model, model, alpha, collocation_domain, collocation_IC,
                optimizer, optimizerL, mse_cost_function,
                training_config['iteration_adam'], training_config['iteration_lbfgs'],
                training_config['batch_size'], training_config['num_batches'],
                update_progress
            )
            
        else:
            # Generic training for other model types
            trained_model = train_function(
                training_id, model, model, alpha, collocation_domain, collocation_IC,
                optimizer, optimizerL, mse_cost_function,
                training_config['iteration_adam'], training_config['iteration_lbfgs'],
                training_config['batch_size'], training_config['num_batches'],
                process_batch_func, model_type, update_progress
            )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save model if requested
        model_id = None
        model_path = None
        if save_model:
            model_id = f"{model_type}_{int(time.time())}"
            model_path = f"pretrained_models/{model_type}/{model_id}.pth"
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(f"pretrained_models/{model_type}", exist_ok=True)
            
            # Save the trained model
            torch.save(trained_model.state_dict(), model_path)
            # Our SafeLogger is completely safe
            logger.info(f"Model saved to {model_path}")
        
        training_info = {
            'training_time_seconds': training_time,
            'model_type': model_type,
            'model_id': model_id,
            'model_path': model_path,
            'use_param_embedding_requested': config.get('use_param_embedding', True),
            'param_value': training_param_value
        }
        
        # Our SafeLogger is completely safe
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        return trained_model, training_info
        
    except Exception as e:
        # Our SafeLogger is completely safe
        logger.error(f"Training failed: {str(e)}")
        raise e

# Utility functions to create base64 plots from visualization modules
def create_plot_from_visualization_module(module_func, model, time_array, initial_params, N, **kwargs):
    """Generic function to create base64 plot from visualization module functions"""
    try:
        plt.close('all')
        
        # Debug logging - our SafeLogger is completely safe
        logger.info(f"Calling visualization function: {module_func.__name__}")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Time array: {time_array}")
        logger.info(f"Initial params: {initial_params}")
        logger.info(f"Additional kwargs: {kwargs}")
        
        # Call the visualization module function
        if 'show' in kwargs:
            kwargs['show'] = True  # Force show=True to generate the plot
        if 'isplot' in kwargs:
            kwargs['isplot'] = True  # Force isplot=True to generate the plot
        
        # Suppress matplotlib warnings during function call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = module_func(model, time_array, initial_params, N, **kwargs)
        
        # Get the current figure (should be the one created by rel_misfit)
        current_fig = plt.gcf()
        
        # Check if we have an active figure
        if not plt.get_fignums():
            # No figure was created, create a default one
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No plot data available', ha='center', va='center', transform=plt.gca().transAxes)
            current_fig = plt.gcf()
        
        # Convert the current matplotlib figure to base64
        buffer = io.BytesIO()
        current_fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(current_fig)
        
        return plot_base64
    except Exception as e:
        plt.close('all')
        # Our SafeLogger is completely safe
        logger.error(f"Error in create_plot_from_visualization_module: {str(e)}")
        logger.error(f"Function: {module_func.__name__}")
        logger.error(f"Arguments: model={type(model)}, time_array={time_array}, initial_params={initial_params}, N={N}, kwargs={kwargs}")
        raise Exception(f"Error in create_plot_from_visualization_module: {str(e)}")

def create_simple_plot_from_data(result_data, plot_type, equation_type, time_value, **params):
    """Create a simple plot from data returned by visualization modules"""
    try:
        plt.close('all')
        
        if equation_type == 'hydro':
            X, rho_pred0, vx_pred0, phi_pred0, rho_max_PN, rho_theory = result_data
            
            if plot_type == 'all':
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Ensure axes is always a list/array
                if not hasattr(axes, '__len__'):
                    axes = [axes]
                
                axes[0].plot(X, rho_pred0)
                axes[0].set_title('Density')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('rho')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(X, vx_pred0)
                axes[1].set_title('Velocity')
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('v')
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(X, phi_pred0)
                axes[2].set_title('Phi')
                axes[2].set_xlabel('x')
                axes[2].set_ylabel('phi')
                axes[2].grid(True, alpha=0.3)
            else:
                plt.figure(figsize=(10, 6))
                alpha_value = params.get('alpha_value', 0.05)
                if plot_type == 'density':
                    plt.plot(X, rho_pred0)
                    plt.ylabel('rho')
                    plt.title(f'Density (alpha={alpha_value:.3f})')
                elif plot_type == 'velocity':
                    plt.plot(X, vx_pred0)
                    plt.ylabel('v')
                    plt.title(f'Velocity (alpha={alpha_value:.3f})')
                elif plot_type == 'phi':
                    plt.plot(X, phi_pred0)
                    plt.ylabel('phi')
                    plt.title(f'Phi (alpha={alpha_value:.3f})')
                
                plt.xlabel('x')
                plt.grid(True, alpha=0.3)
                
        elif equation_type == 'burgers':
            X, u_pred = result_data
            plt.figure(figsize=(10, 6))
            plt.plot(X, u_pred, linewidth=2, label='PINN Solution')
            plt.xlabel('x')
            plt.ylabel('u(x, t)')
            plt.title(f'Burgers Equation Solution at t={time_value}')
            plt.grid(True, alpha=0.3)   
            plt.legend()
            
        elif equation_type == 'wave':
            results = result_data
            X = results['X']
            u_pinn = results['u_pinn']
            c = params.get('c_value', 1.0)
            
            plt.figure(figsize=(10, 6))
            if len(u_pinn) == 1:
                plt.plot(X.flatten(), u_pinn[0, :], linewidth=2, label='PINN Solution')
                plt.xlabel('x')
                plt.ylabel('u(x, t)')
                plt.title(f'Wave Equation Solution at t={time_value} (c={c})')
            else:
                plt.contourf(X.flatten(), [time_value], u_pinn, levels=50, cmap='viridis')
                plt.xlabel('x')
                plt.ylabel('t')
                plt.title(f'Wave Equation Solution (c={c})')
                plt.colorbar()
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        elif equation_type == 'SHM':
            results = result_data
            T = results['T']
            x_pinn = results['x_pinn']
            c = params.get('c_value', 1.0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(T.flatten(), x_pinn[0, :], linewidth=2, label='PINN Solution')
            plt.xlabel('t')
            plt.ylabel('x(t)')
            plt.title(f'SHM Solution (c={c}, ω={np.sqrt(c):.2f})')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64
    except Exception as e:
        plt.close('all')
        # Our SafeLogger is completely safe
        logger.error(f"Error in create_simple_plot_from_data: {str(e)}")
        raise Exception(f"Error in create_simple_plot_from_data: {str(e)}")

def generate_hydro_plot_with_comparison(model, t_value, x_min, x_max, plot_type, show_comparison=False, model_info=None, alpha_value=0.05):
    """Generate plot for Hydrodynamics equation using visualization modules"""
    
    # Get parameters from model_info
    lam = model_info['lam']
    num_of_waves = model_info['num_of_waves']
    N = 1000
    
    # Calculate parameters
    from dependency_codes.solver import req_consts_calc
    jeans, alpha_val = req_consts_calc(lam)
    v_1 = ((alpha_val/(rho_o*2*np.pi/lam)) * alpha_value)
    tmax = 2.0
    
    # Create initial params tuple
    initial_params = (x_min, x_max, alpha_value, alpha_value, v_1, jeans, lam, tmax, device)
    time_array = np.array([t_value])
    
    if show_comparison:
        # Use rel_misfit for comparison plots
        nu = 0.5  # courant number for LAX solver
        # Our SafeLogger is completely safe
        logger.info(f"Calling rel_misfit with nu={nu}, num_of_waves={num_of_waves}, rho_1={alpha_value}")
        return create_plot_from_visualization_module(
            rel_misfit, model, time_array, initial_params, N, 
            nu=nu, num_of_waves=num_of_waves, rho_1=alpha_value
        )
    else:
        # Use plot_function for simple plots
        result_data = plot_function(model, time_array, initial_params, N, velocity=True, isplot=False, animation=True)
        return create_simple_plot_from_data(result_data, plot_type, 'hydro', t_value, alpha_value=alpha_value)

def generate_burgers_plot_with_comparison(model, t_value, x_min, x_max, plot_type, show_comparison=False, nu_value=0.02):
    """Generate plot for Burgers equation using visualization modules"""
    
    N = 1000
    nu = nu_value  # Use provided viscosity value
    initial_params = (x_min, x_max, t_value, device)  # Use t_value instead of 2.0
    time_array = np.array([t_value])
    
    if show_comparison:
        # Use rel_misfit_burgers for comparison plots
        from analytical_solutions.burgers.burgers_analytical import burgers_analytical_solution
        def true_solution_fn(x, t, nu):
            return burgers_analytical_solution(x, t, nu, initial_condition='sin')
        
        return create_plot_from_visualization_module(
            rel_misfit_burgers, model, time_array, initial_params, N,
            nu=nu, true_solution_fn=true_solution_fn
        )
    else:
        # Use plot_burgers_solution for simple plots
        result_data = plot_burgers_solution(model, time_array, initial_params, N, nu, isplot=False)
        return create_simple_plot_from_data(result_data, plot_type, 'burgers', t_value)

def generate_wave_plot_with_comparison(model, t_value, x_min, x_max, plot_type, show_comparison=False, c_value=1.0):
    """Generate plot for Wave equation using visualization modules"""
    
    N = 1000
    initial_params = (x_min, x_max, t_value, device)  # Use t_value instead of 2.0
    time_array = np.array([t_value])
    
    if show_comparison:
        # Use rel_misfit_wave for comparison plots
        def analytical_solution_fn(x, t, c):
            from analytical_solutions.wave.wave_analytical import wave_analytical_solution_exact
            return wave_analytical_solution_exact(x, t, c, initial_condition_type='sine', 
                                                wavenumber=2*np.pi, amplitude=1.0)
        
        return create_plot_from_visualization_module(
            rel_misfit_wave, model, time_array, initial_params, N,
            c=c_value, analytical_solution_fn=analytical_solution_fn
        )
    else:
        # Use plot_wave_solution for simple plots
        result_data = plot_wave_solution(model, time_array, initial_params, N, c_value, isplot=False)
        return create_simple_plot_from_data(result_data, plot_type, 'wave', t_value, c_value=c_value)

def generate_SHM_plot_with_comparison(model, t_value, x_min, x_max, plot_type, show_comparison=False, c_value=1.0):
    """Generate plot for SHM equation using visualization modules"""
    
    N = 1000
    initial_params = (0.0, t_value, device)  # (tmin, tmax, device) for SHM
    time_array = np.array([t_value])
    
    if show_comparison:
        # Use rel_misfit_SHM for comparison plots
        def analytical_solution_fn(t, c, amplitude=1.0, initial_velocity=0.0):
            from analytical_solutions.SHM.SHM_analytical import SHM_analytical_solution
            return SHM_analytical_solution(t, c, amplitude, initial_velocity)
        
        return create_plot_from_visualization_module(
            rel_misfit_SHM, model, time_array, initial_params, N,
            c=c_value, analytical_solution_fn=analytical_solution_fn
        )
    else:
        # Use plot_SHM_solution for simple plots
        result_data = plot_SHM_solution(model, time_array, initial_params, N, c_value, isplot=False)
        return create_simple_plot_from_data(result_data, plot_type, 'SHM', t_value, c_value=c_value)

@app.route('/')
def index():
    return render_template('index.html', pretrained_models=PRETRAINED_MODELS)

@app.route('/list_models', methods=['GET'])
def list_models():
    """List all available models (pretrained and newly trained)"""
    try:
        # Get pretrained models
        pretrained_list = list(PRETRAINED_MODELS.keys())
        
        # Get newly trained models
        trained_list = list(loaded_models.keys())
        
        return jsonify({
            'success': True,
            'pretrained_models': pretrained_list,
            'trained_models': trained_list,
            'total_models': len(pretrained_list) + len(trained_list)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/progress/<training_id>', methods=['GET'])
def get_progress(training_id):
    """Get training progress for a specific training session"""
    try:
        with progress_lock:
            if training_id in training_progress:
                progress_data = training_progress[training_id]
                return jsonify(progress_data)
            else:
                return jsonify({'error': 'Training ID not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_training/<training_id>', methods=['POST'])
def stop_training(training_id):
    """Stop training for a specific training session"""
    try:
        print(f"DEBUG: Stop training requested for training_id: {training_id}")
        
        # Set both app-level and global stop flags
        with stop_flags_lock:
            training_stop_flags[training_id] = True
            print(f"DEBUG: App stop flag set to True for {training_id}")
        
        # Also set global stop flag for cross-thread access
        try:
            from dependency_codes.solvers.base import _global_stop_flags, _global_stop_lock
            with _global_stop_lock:
                _global_stop_flags[training_id] = True
                print(f"DEBUG: Global stop flag set to True for {training_id}")
        except ImportError as e:
            print(f"DEBUG: Could not import global stop flags: {e}")
        
        with progress_lock:
            if training_id in training_progress:
                training_progress[training_id].update({
                    'status': 'stopping',
                    'message': 'Stopping training...'
                })
        
        return jsonify({'success': True, 'message': 'Training stop requested'})
    except Exception as e:
        print(f"DEBUG: Error in stop_training: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        # Get parameters from form
        model_type = request.form['model_type']  # 'pretrained' or 'new'
        show_comparison = request.form.get('show_comparison', 'false') == 'true'
        
        if model_type == 'pretrained':
            model_name = request.form['pretrained_model']
            model, error = load_pretrained_model(model_name)
            if model is None:
                return jsonify({'success': False, 'error': error})
            model_info = PRETRAINED_MODELS[model_name]
        else:
            # For newly trained models
            model_id = request.form['trained_model_id']
            if model_id not in loaded_models:
                return jsonify({'success': False, 'error': 'Model not found. Please train a model first.'})
            
            model = loaded_models[model_id]
            # Create a basic model_info for newly trained models
            model_info = {
                'type': request.form.get('trained_model_type', 'hydro'),  # Default to hydro
                'lam': float(request.form.get('lam', 7.0)),
                'num_of_waves': int(request.form.get('num_of_waves', 2))
            }
            # Try to retrieve stored embedding info
            try:
                info = loaded_models_info.get(model_id, {})
                model_info['use_param_embedding_requested'] = info.get('use_param_embedding_requested', True)
                model_info['trained_param_value'] = info.get('param_value', None)
            except Exception:
                model_info['use_param_embedding_requested'] = True
                model_info['trained_param_value'] = None
        
        # Get visualization parameters
        t_value = float(request.form['time'])
        x_min = float(request.form['x_min'])
        x_max = float(request.form['x_max'])
        plot_type = request.form.get('plot_type', 'density')
        # Determine parameter value: if training used single value (embedding disabled), use stored value
        if 'trained_model_id' in request.form:
            use_embed_flag = model_info.get('use_param_embedding_requested', True)
            trained_param = model_info.get('trained_param_value', None)
            if not use_embed_flag and trained_param is not None:
                alpha_value = float(trained_param)
            else:
                alpha_value = float(request.form.get('alpha', 0.05))
        else:
            alpha_value = float(request.form.get('alpha', 0.05))
        
        # Generate plot based on model type
        if model_info['type'] == 'hydro':
            plot_data = generate_hydro_plot_with_comparison(model, t_value, x_min, x_max, 
                                                            plot_type, show_comparison, model_info, alpha_value)
        elif model_info['type'] == 'burgers':
            nu_val = alpha_value
            plot_data = generate_burgers_plot_with_comparison(model, t_value, x_min, x_max, 
                                                              plot_type, show_comparison, nu_val)
        elif model_info['type'] == 'wave':
            c_value = alpha_value
            plot_data = generate_wave_plot_with_comparison(model, t_value, x_min, x_max, 
                                                           plot_type, show_comparison, c_value)
        elif model_info['type'] == 'SHM':
            c_value = alpha_value
            plot_data = generate_SHM_plot_with_comparison(model, t_value, x_min, x_max, 
                                                          plot_type, show_comparison, c_value)
        else:
            plot_data = generate_hydro_plot_with_comparison(model, t_value, x_min, x_max, 
                                                            plot_type, show_comparison, model_info, alpha_value)
        
        return jsonify({'success': True, 'plot': plot_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train', methods=['POST'])
def train():
    try:
        # Get training parameters from form
        model_type = request.form['model_type']
        save_model = request.form.get('save_model', 'false') == 'true'
        
        # Generate unique training ID
        training_id = f"training_{int(time.time())}"
        
        # Initialize progress tracking
        with progress_lock:
            training_progress[training_id] = {
                'status': 'starting',
                'phase': 'initialization',
                'progress': 0,
                'message': 'Initializing training...',
                'current_iteration': 0,
                'total_iterations': 0
            }
        
        # Initialize stop flag
        with stop_flags_lock:
            training_stop_flags[training_id] = False
        
        # Training configuration
        config = {
            'architecture': request.form.get('architecture', 'P2PINN'),
            'num_neurons': int(request.form.get('num_neurons', 96)),
            'use_param_embedding': request.form.get('use_param_embedding', 'true') == 'true',
            'param_embed_activation': request.form.get('param_embed_activation', 'tanh'),
            'input_embed_activation': request.form.get('input_embed_activation', 'sin'),
            'main_pinn_activation': request.form.get('main_pinn_activation', 'sin'),
            'param_embed_layers': int(request.form.get('param_embed_layers', 2)),
            'param_embed_neurons': int(request.form.get('param_embed_neurons', 32)),
            'input_embed_layers': int(request.form.get('input_embed_layers', 3)),
            'input_embed_neurons': int(request.form.get('input_embed_neurons', 64)),
            'main_pinn_layers': int(request.form.get('main_pinn_layers', 5))
        }
        
        # Training hyperparameters
        training_config = {
            'iteration_adam': int(request.form.get('iteration_adam', 1000)),
            'iteration_lbfgs': int(request.form.get('iteration_lbfgs', 100)),
            'learning_rate': float(request.form.get('learning_rate', 0.001)),
            'batch_size': int(request.form.get('batch_size', 1000)),
            'num_batches': int(request.form.get('num_batches', 5))
        }
        
        # Model-specific parameters from form with sensible defaults
        def _get_int(name, default):
            try:
                return int(request.form.get(name, default))
            except Exception:
                return default
        def _get_float(name, default):
            try:
                return float(request.form.get(name, default))
            except Exception:
                return default

        tmax_val = _get_float('tmax', 2.0)
        # Prefer generic names; fall back to equation-specific field names
        N_0_val = _get_int('N_0', _get_int('N_0_burgers', _get_int('N_0_wave', 100)))
        N_b_val = _get_int('N_b', _get_int('N_b_burgers', _get_int('N_b_wave', 100)))
        N_r_val = _get_int('N_r', _get_int('N_r_burgers', _get_int('N_r_wave', 1000)))

        # Optional spatial xmax for burgers/wave
        xmax_val = request.form.get('xmax')
        try:
            xmax_val = float(xmax_val) if xmax_val is not None else None
        except Exception:
            xmax_val = None

        model_params = {
            'tmax': tmax_val,
            'N_0': N_0_val,
            'N_b': N_b_val,
            'N_r': N_r_val
        }
        if xmax_val is not None:
            model_params['xmax'] = xmax_val

        # Collect model-specific parameter inputs
        if model_type == 'hydro':
            if 'alpha_single' in request.form:
                model_params.update({ 'alpha_single': _get_float('alpha_single', 0.05) })
            else:
                model_params.update({
                    'alpha_min': _get_float('alpha_min', 0.01),
                    'alpha_max': _get_float('alpha_max', 0.1),
                    'alpha_N': _get_int('alpha_N', 5)
                })
        elif model_type == 'burgers':
            if 'nu_single' in request.form:
                model_params.update({ 'nu_single': _get_float('nu_single', 0.02) })
            else:
                model_params.update({
                    'nu_min': _get_float('nu_min', 0.01),
                    'nu_max': _get_float('nu_max', 0.05),
                    'nu_N': _get_int('nu_N', 5)
                })
        elif model_type == 'wave':
            if 'c_single' in request.form:
                model_params.update({ 'c_single': _get_float('c_single', 1.0) })
            else:
                model_params.update({
                    'c_min': _get_float('c_min', 0.5),
                    'c_max': _get_float('c_max', 2.0),
                    'c_N': _get_int('c_N', 5)
                })
        elif model_type == 'SHM':
            if 'c_single_shm' in request.form:
                model_params.update({ 'c_single_shm': _get_float('c_single_shm', 4.0),
                                      'tmax': _get_float('tmax_shm', tmax_val),
                                      'N_0': _get_int('N_0_shm', N_0_val),
                                      'N_r': _get_int('N_r_shm', N_r_val) })
            else:
                model_params.update({
                    'c_min_shm': _get_float('c_min_shm', 1.0),
                    'c_max_shm': _get_float('c_max_shm', 9.0),
                    'c_N_shm': _get_int('c_N_shm', 5),
                    'tmax': _get_float('tmax_shm', tmax_val),
                    'N_0': _get_int('N_0_shm', N_0_val),
                    'N_r': _get_int('N_r_shm', N_r_val)
                })
        
        # Return training ID immediately so frontend can start polling
        response_data = {
            'success': True, 
            'message': f'{model_type.capitalize()} training started',
            'training_id': training_id,
            'model_type': model_type
        }
        
        # Start training in a separate thread
        def train_in_background():
            try:
                # Create and train the model
                model, training_info = train_model_with_progress(
                    training_id=training_id,
                    model_type=model_type,
                    config=config,
                    training_config=training_config,
                    model_params=model_params,
                    save_model=save_model
                )
                
                # Store the trained model
                model_id = f"{model_type}_{int(time.time())}"
                loaded_models[model_id] = model
                # Track whether embedding was requested and the param used (if single value)
                loaded_models_info[model_id] = {
                    'model_type': model_type,
                    'use_param_embedding_requested': config.get('use_param_embedding', True),
                    'param_value': training_info.get('param_value')
                }
                
                # Update progress
                with progress_lock:
                    training_progress[training_id].update({
                        'status': 'completed',
                        'phase': 'completed',
                        'progress': 100,
                        'message': f'{model_type.capitalize()} model trained successfully',
                        'model_id': model_id,
                        'use_param_embedding_requested': config.get('use_param_embedding', True),
                        'param_value': training_info.get('param_value')
                    })
                
                # Clean up stop flag
                with stop_flags_lock:
                    if training_id in training_stop_flags:
                        del training_stop_flags[training_id]
                    
            except Exception as e:
                # Update progress to error
                with progress_lock:
                    training_progress[training_id].update({
                        'status': 'error',
                        'message': f'Training failed: {str(e)}'
                    })
                
                # Clean up stop flag
                with stop_flags_lock:
                    if training_id in training_stop_flags:
                        del training_stop_flags[training_id]
        
        # Start training in background thread
        training_thread = threading.Thread(target=train_in_background)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Configure Flask app with better timeout and error handling
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    # Set encoding to UTF-8 to prevent character encoding issues
    import sys
    import codecs
    # Only detach if not already detached to avoid errors
    try:
        if not sys.stdout.isatty():
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        if not sys.stderr.isatty():
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass  # If detaching fails, continue without it
    
    # Add error handlers
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    
    # Check if user wants to use Waitress for external access
    import sys
    use_waitress = '--waitress' in sys.argv or '--external' in sys.argv
    
    if use_waitress:
        try:
            from waitress import serve
            print("=" * 60)
            print("🚀 PINNLab Waitress Server Starting...")
            print("📊 PINN Trainer and Visualizer (External Access)")
            print(f"🌐 Server will be accessible from any device on port {PORT}")
            print(f" Local access: http://127.0.0.1:{PORT}/")
            print(f"🌐 Network access: http://YOUR_IP_ADDRESS:{PORT}/")
            print("=" * 60)
            
            # Use Waitress for external access
            serve(app, host='0.0.0.0', port=PORT, threads=4)
        except ImportError:
            print("❌ Waitress not installed. Install with: pip install waitress")
            print("🔄 Falling back to Flask development server...")
            use_waitress = False
    
    if not use_waitress:
        # Print startup message for Flask development server
        print("=" * 60)
        print("🚀 PINNLab Flask Application Starting...")
        print("📊 PINN Trainer and Visualizer (Development Mode)")
        print(f"🌐 Access the application at: http://127.0.0.1:{PORT}/")
        print(f"🌐 Or from other devices at: http://{HOST}:{PORT}/")
        print(f"🔧 Debug mode: {DEBUG}")
        print(f"📝 Log level: {LOG_LEVEL}")
        print(f"📁 Log file: {LOG_FILE}")
        print("💡 To enable external access, run: python app.py --waitress")
        print("=" * 60)
        
        # Run with increased timeout and threaded mode
        app.run(
            debug=DEBUG,  # Use config value
            host=HOST,    # Use config value
            port=PORT,    # Use config value
            threaded=True,  # Enable threading for concurrent requests
            use_reloader=False  # Disable auto-reloader
        )