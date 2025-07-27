from flask import Flask, render_template, request, jsonify, Response
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI warnings
import matplotlib.pyplot as plt
import io
import base64
import time
import json
import threading
from torch.autograd import Variable
from scipy.interpolate import interp1d

# Import your existing code
from model_architecture import PINN
from models.burgers import BurgersPINN
from models.hydro import HydroPINN
from visualisation_burgers import plot_burgers_solution, rel_misfit_burgers
from visualisation import plot_function, rel_misfit
from config import xmin, tmin, rho_o
from LAX import lax_solution1D

app = Flask(__name__)

# Progress tracking
training_progress = {}
progress_lock = threading.Lock()

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Available pretrained models with their parameter ranges
PRETRAINED_MODELS = {
    'hydro_case1': {
        'file': 'Case1_final_part1.pth',
        'type': 'hydro',
        'description': 'Hydrodynamics Case 1',
        'lam': 7.0,
        'alpha_min': 0.01,
        'alpha_max': 0.08,
        'num_of_waves': 2
    },
    'hydro_case2': {
        'file': 'Case2_final_part1.pth', 
        'type': 'hydro',
        'description': 'Hydrodynamics Case 2',
        'lam': 7.0,
        'alpha_min': 0.1,
        'alpha_max': 0.8,
        'num_of_waves': 2
    },
    'hydro_case3': {
        'file': 'Case3_final_part1.pth',
        'type': 'hydro', 
        'description': 'Hydrodynamics Case 3',
        'lam': 5.0,
        'alpha_min': 0.01,
        'alpha_max': 0.08,
        'num_of_waves': 2
    },
    'mixed': {
        'file': 'mixed_final_part1.pth',
        'type': 'mixed',
        'description': 'Mixed Model',
        'lam': 7.0,
        'alpha_min': 0.09,
        'alpha_max': 0.11,
        'num_of_waves': 2
    },

    # Placeholder for Burgers models (add when available)
    # 'burgers_case1': {
    #     'file': 'burgers_case1.pth',
    #     'type': 'burgers',
    #     'description': 'Burgers Case 1'
    # }
}

# Load models into memory
loaded_models = {}

def load_pretrained_model(model_name):
    """Load a pretrained model"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_info = PRETRAINED_MODELS[model_name]
    model_file = model_info['file']
    model_type = model_info['type']
    
    try:
        # Use PINN class for hydrodynamics pretrained models (compatible with saved models)
        if model_type in ['hydro', 'mixed']:
            model = PINN()
        else:
            # For Burgers and future models, use the new architecture classes
            if model_type == 'burgers':
                model = BurgersPINN()
            else:
                # Default to HydroPINN for unknown types (future compatibility)
                model = HydroPINN()
        
        # Load weights
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        
        loaded_models[model_name] = model
        return model
    except Exception as e:
        raise Exception(f"Error loading model {model_name}: {str(e)}")

def create_new_model(model_type, config):
    """Create a new model with custom architecture"""
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    return model

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

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        # Get parameters from form
        model_type = request.form['model_type']  # 'pretrained' or 'new'
        show_comparison = request.form.get('show_comparison', 'false') == 'true'
        
        if model_type == 'pretrained':
            model_name = request.form['pretrained_model']
            model = load_pretrained_model(model_name)
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
        
        # Get visualization parameters
        t_value = float(request.form['time'])
        x_min = float(request.form['x_min'])
        x_max = float(request.form['x_max'])
        plot_type = request.form.get('plot_type', 'density')
        alpha_value = float(request.form.get('alpha', 0.05))  # Get alpha from form
        
        # Generate plot based on model type
        if model_info['type'] == 'hydro':
            plot_data = generate_hydro_plot_with_comparison(model, t_value, x_min, x_max, 
                                                            plot_type, show_comparison, model_info, alpha_value)
        elif model_info['type'] == 'burgers':
            # For Burgers, use the selected plot type (should be 'solution')
            plot_data = generate_burgers_plot_with_comparison(model, t_value, x_min, x_max, 
                                                              plot_type, show_comparison)
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
        
        # Training configuration
        config = {
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
            'iteration_adam': int(request.form.get('iteration_adam', 500)),
            'iteration_lbfgs': int(request.form.get('iteration_lbfgs', 50)),
            'learning_rate': float(request.form.get('learning_rate', 0.001)),
            'batch_size': int(request.form.get('batch_size', 1000)),
            'num_batches': int(request.form.get('num_batches', 5))
        }
        
        # Model-specific parameters
        if model_type == 'hydro':
            # Hydrodynamics parameters
            lam = float(request.form.get('lam', 7.0))
            rho_1 = float(request.form.get('rho_1', 0.03))
            num_of_waves = int(request.form.get('num_of_waves', 2))
            tmax = float(request.form.get('tmax', 1.5))
            alpha_min = float(request.form.get('alpha_min', 0.01))
            alpha_max = float(request.form.get('alpha_max', 0.1))
            alpha_N = int(request.form.get('alpha_N', 5))
            # Domain configuration
            N_0 = int(request.form.get('N_0', 100))
            N_b = int(request.form.get('N_b', 100))
            N_r = int(request.form.get('N_r', 1000))
        elif model_type == 'burgers':
            # Burgers parameters - viscosity range for training (like alpha in hydrodynamics)
            nu_min = float(request.form.get('nu_min', 0.01))
            nu_max = float(request.form.get('nu_max', 0.05))
            nu_N = int(request.form.get('nu_N', 5))
            tmax = float(request.form.get('tmax', 1.0))
            # Domain configuration
            N_0 = int(request.form.get('N_0', 100))
            N_b = int(request.form.get('N_b', 100))
            N_r = int(request.form.get('N_r', 1000))
        else:
            return jsonify({'success': False, 'error': f'Unknown model type: {model_type}'})
        
        # Update progress
        with progress_lock:
            training_progress[training_id].update({
                'status': 'training',
                'phase': 'setup',
                'progress': 5,
                'message': f'Setting up {model_type} training...',
                'total_iterations': training_config['iteration_adam'] + training_config['iteration_lbfgs']
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
                    model_params={
                        'lam': lam if model_type == 'hydro' else None,
                        'rho_1': rho_1 if model_type == 'hydro' else None,
                        'num_of_waves': num_of_waves if model_type == 'hydro' else None,
                        'tmax': tmax,
                        'alpha_min': alpha_min if model_type == 'hydro' else None,
                        'alpha_max': alpha_max if model_type == 'hydro' else None,
                        'alpha_N': alpha_N if model_type == 'hydro' else None,
                        'nu_min': nu_min if model_type == 'burgers' else None,
                        'nu_max': nu_max if model_type == 'burgers' else None,
                        'nu_N': nu_N if model_type == 'burgers' else None,
                        'N_0': N_0,
                        'N_b': N_b,
                        'N_r': N_r
                    }
                )
                
                # Store the trained model (in a real app, you'd save it to disk)
                model_id = f"{model_type}_{int(time.time())}"
                loaded_models[model_id] = model
                
                # Update progress to completed
                with progress_lock:
                    training_progress[training_id].update({
                        'status': 'completed',
                        'phase': 'completed',
                        'progress': 100,
                        'message': f'{model_type.capitalize()} model trained successfully',
                        'current_iteration': training_config['iteration_adam'] + training_config['iteration_lbfgs'],
                        'model_id': model_id
                    })
                    
            except Exception as e:
                # Update progress to error
                with progress_lock:
                    training_progress[training_id].update({
                        'status': 'error',
                        'message': f'Training failed: {str(e)}'
                    })
        
        # Start training in background thread
        training_thread = threading.Thread(target=train_in_background)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify(response_data)
        
    except Exception as e:
        # Update progress to error
        with progress_lock:
            if training_id in training_progress:
                training_progress[training_id].update({
                    'status': 'error',
                    'message': f'Training failed: {str(e)}'
                })
        return jsonify({'success': False, 'error': str(e)})

def train_model_with_progress(training_id, model_type, config, training_config, model_params):
    """Train a PINN model with progress tracking"""
    import time
    import warnings
    
    print(f"Starting training for {model_type} model")
    
    # Suppress specific warnings during training
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.loss")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    from data_generator import alpha_generator
    from solver import input_taker, req_consts_calc, closure_batched, train_batched_with_progress
    from losses.losses import ASTPN
    
    start_time = time.time()
    
    # Set MODEL_TYPE based on model_type for proper loss function selection
    import config as config_module
    config_module.MODEL_TYPE = model_type
    
    # Also update MODEL_TYPE in solver module
    import solver
    solver.MODEL_TYPE = model_type
    
    print(f"MODEL_TYPE set to {model_type}")
    
    # Create progress update function
    def update_progress(training_id, progress_data):
        with progress_lock:
            if training_id in training_progress:
                training_progress[training_id].update(progress_data)
    
    # Update progress
    update_progress(training_id, {
        'phase': 'model_creation',
        'progress': 10,
        'message': f'Creating {model_type} model...'
    })
    
    # Create model
    model = create_new_model(model_type, config)
    
    # Setup training data and parameters
    if model_type == 'hydro':
        # Hydrodynamics training setup
        lam = model_params['lam']
        rho_1 = model_params['rho_1']
        num_of_waves = model_params['num_of_waves']
        tmax = model_params['tmax']
        alpha_min = model_params['alpha_min']
        alpha_max = model_params['alpha_max']
        alpha_N = model_params['alpha_N']
        
        # Generate alpha values for training
        alpha_min, alpha_max, alpha_list = alpha_generator(
            alpha_min=alpha_min, 
            alpha_max=alpha_max, 
            N=alpha_N
        )
        alpha_list = alpha_list.to(device)
        
        # Calculate required constants
        jeans, alpha = req_consts_calc(lam)
        v_1 = ((alpha/(rho_o*2*np.pi/lam)) * alpha_list)
        
        # Domain setup - use same logic as train.py
        xmax = xmin + lam * num_of_waves
        N_0 = model_params['N_0']
        N_b = model_params['N_b']
        N_r = model_params['N_r']
        
    elif model_type == 'burgers':
        # Burgers training setup - viscosity as parameter (like alpha in hydrodynamics)
        nu_min = model_params['nu_min']
        nu_max = model_params['nu_max']
        nu_N = model_params['nu_N']
        tmax = model_params['tmax']
        
        # Generate viscosity values for training (like alpha generation in hydrodynamics)
        nu_values = torch.linspace(nu_min, nu_max, nu_N, device=device, dtype=torch.float32)
        alpha_list = nu_values  # Use nu values as alpha_list for consistency
        
        # Domain setup for Burgers: [0,1] x [0,tmax]
        xmax = 1.0  # Burgers typically uses [0,1] domain
        N_0 = model_params['N_0']
        N_b = model_params['N_b']
        N_r = model_params['N_r']
        
        # Placeholder values for hydro-specific variables (not used in Burgers)
        lam, rho_1, num_of_waves = 1.0, 1.0, 1
        jeans, alpha = 1.0, 1.0
        v_1 = alpha_list  # Use nu values as v_1 for Burgers
        
        print(f"Burgers training setup:")
        print(f"  Domain: x in [0, {xmax}], t in [0, {tmax}]")
        print(f"  Viscosity range: [{nu_min}, {nu_max}] with {nu_N} values")
        print(f"  Collocation points: N_0={N_0}, N_b={N_b}, N_r={N_r}")
    
    # Update progress
    update_progress(training_id, {
        'phase': 'data_setup',
        'progress': 20,
        'message': f'Setting up training data for {model_type}...'
    })
    
    # Create training model and collocation points
    model_1D = ASTPN(
        rmin=[xmin, tmin], 
        rmax=[xmax, tmax], 
        N_0=N_0, 
        N_b=N_b, 
        N_r=N_r, 
        dimension=1
    )
    
    collocation_domain_1D = model_1D.geo_time_coord(option="Domain")
    collocation_IC_1D = model_1D.geo_time_coord(option="IC")
    
    # Ensure collocation points have requires_grad=True for differentiation
    collocation_domain_1D = [coord.requires_grad_(True) for coord in collocation_domain_1D]
    collocation_IC_1D = [coord.requires_grad_(True) for coord in collocation_IC_1D]
    
    # Setup optimizers
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate']
    )
    optimizerL = torch.optim.LBFGS(
        model.parameters(), 
        line_search_fn='strong_wolfe'
    )
    
    # Update progress
    update_progress(training_id, {
        'phase': 'training',
        'progress': 30,
        'message': f'Starting {model_type} training...'
    })
    
    # Train the model with progress tracking
    train_batched_with_progress(
        training_id=training_id,
        net=model,
        model=model_1D,
        alpha=alpha_list,
        collocation_domain=collocation_domain_1D,
        collocation_IC=collocation_IC_1D,
        optimizer=optimizer,
        optimizerL=optimizerL,
        closure=closure_batched,
        mse_cost_function=mse_cost_function,
        iteration_adam=training_config['iteration_adam'],
        iterationL=training_config['iteration_lbfgs'],
        lam=lam,
        jeans=jeans,
        v_1=v_1,
        batch_size=training_config['batch_size'],
        num_batches=training_config['num_batches'],
        progress_update_fn=update_progress
    )
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_built():
        torch.mps.empty_cache()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    training_info = {
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'model_type': model_type,
        'config': config,
        'training_config': training_config,
        'model_params': model_params
    }
    
    return model, training_info

def generate_hydro_plot_with_comparison(model, t_value, x_min, x_max, plot_type, show_comparison=False, model_info=None, alpha_value=0.05):
    """Generate plot for Hydrodynamics equation with optional LT/FD comparison"""
    
    if show_comparison:
        # Use the rel_misfit logic for comparison plots
        # Get parameters from model_info
        lam = model_info['lam']
        num_of_waves = model_info['num_of_waves']
        nu = 0.5  # courant number
        N = 8000  # number of grid points (same as inference.py)
        alpha = alpha_value  # Use the alpha value directly
        
        # Calculate parameters exactly like visualisation.py and inference.py
        from solver import req_consts_calc
        jeans, alpha_val = req_consts_calc(lam)
        v_1 = ((alpha_val/(rho_o*2*np.pi/lam)) * alpha)
        tmax = 2.0
        
        # Create initial params tuple for rel_misfit (matching the expected format)
        initial_params = (x_min, x_max, alpha, alpha, v_1, jeans, lam, tmax, device)
        
        # Create a single time array for the specific time
        time_array = np.array([t_value])
        
        # Generate comparison plot using rel_misfit logic
        plot_data = generate_hydro_comparison_plot(model, time_array, initial_params, N, nu, num_of_waves, alpha, plot_type)
    else:
        # Simple plot without comparison
        plot_data = generate_hydro_simple_plot(model, t_value, x_min, x_max, plot_type, alpha_value)
    
    return plot_data

def generate_hydro_comparison_plot(model, time_array, initial_params, N, nu, num_of_waves, alpha, plot_type='density'):
    """Generate hydrodynamics plot with LT/FD comparison using rel_misfit logic"""
    try:
        # Clear any existing matplotlib figures to free memory
        plt.close('all')
        
        xmin, xmax, rho_1, alpha, v_1, jeans, lam, tmax, device = initial_params
        
        t = time_array[0]
        
        # Use the exact same approach as visualisation.py rel_misfit function
        # Get FD and LT solutions using lax_solution1D
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max, v_LT = lax_solution1D(
            t, N, nu, lam, num_of_waves, alpha, gravity=True, isplot=False, comparison=True
        )
        
        # Get PINN solution using plot_function (exact same as visualisation.py)
        X, rho_pred0, v_pred0, phi_pred0, rho_max_PN, rho_theory = plot_function(
            model, t, initial_params, N, velocity=True, isplot=False, animation=True
        )
        
        # Interpolate FD and LT solutions to PINN grid (X) to avoid broadcasting issues
        from scipy.interpolate import interp1d
        interp_rho = interp1d(x, rho, kind='cubic', fill_value='extrapolate')
        interp_v = interp1d(x, v, kind='cubic', fill_value='extrapolate')
        rho_fd_on_X = interp_rho(X[:, 0])
        v_fd_on_X = interp_v(X[:, 0])
        
        if alpha < 0.1:
            interp_rho_LT = interp1d(x, rho_LT, kind='cubic', fill_value='extrapolate')
            interp_v_LT = interp1d(x, v_LT, kind='cubic', fill_value='extrapolate')
            rho_lt_on_X = interp_rho_LT(X[:, 0])
            v_lt_on_X = interp_v_LT(X[:, 0])
        
        # Create the same plotting structure as visualisation.py rel_misfit
        plt.style.use('default')
        plt.rc('grid', linestyle='-', color='black', linewidth=0.05)
        
        if plot_type == 'all':
            # Create 4x1 subplot structure like rel_misfit
            fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
            plt.subplots_adjust(wspace=0.12, hspace=0.1)
            
            # Density comparison (top plot)
            axes[0].plot(X, rho_pred0, color='c', linewidth=5, label="PN")
            if alpha < 0.1:
                axes[0].plot(X, rho_lt_on_X, linestyle='dashed', color='firebrick', linewidth=3, label="LT")
            axes[0].plot(X, rho_fd_on_X, linestyle='solid', color='black', linewidth=1.0, label="FD")
            axes[0].set_xlim(xmin, xmax)
            axes[0].set_title(f"Time={round(t, 2)}")
            axes[0].set_ylabel(r"$\rho$", fontsize=18)
            axes[0].grid("True")
            axes[0].minorticks_on()
            axes[0].tick_params(labelsize=10)
            axes[0].legend(loc='best', fancybox=False, shadow=False, ncol=3, fontsize=10)
            
            # Density misfit (second plot)
            axes[1].plot(X, (rho_pred0[:, 0] - rho_fd_on_X) / ((rho_pred0[:, 0] + rho_fd_on_X) / 2) * 100, 
                        color='black', linewidth=1, label="FD")
            if alpha < 0.1:
                axes[1].plot(X, (rho_pred0[:, 0] - rho_lt_on_X) / ((rho_pred0[:, 0] + rho_lt_on_X) / 2) * 100, 
                            color='b', linewidth=1, label="LT")
            axes[1].set_xlabel("x", fontsize=18)
            axes[1].grid("True")
            axes[1].minorticks_on()
            axes[1].tick_params(labelsize=10)
            axes[1].legend(loc='best', fancybox=False, shadow=False, ncol=3, fontsize=10)
            axes[1].set_ylim(-5.0, 5.0)
            axes[1].set_xlim(xmin, xmax)
            axes[1].set_ylabel(r"Rel misfit $\%$ ", fontsize=14)
            
            # Velocity comparison (third plot)
            axes[2].plot(X, v_pred0, color='c', linewidth=5, label="PN")
            if alpha < 0.1:
                axes[2].plot(X, v_lt_on_X, linestyle='dashed', color='firebrick', linewidth=3, label="LT")
            axes[2].plot(X, v_fd_on_X, linestyle='solid', color='black', linewidth=1.0, label="FD")
            axes[2].set_ylabel(r"$v$", fontsize=18)
            axes[2].grid("True")
            axes[2].minorticks_on()
            axes[2].tick_params(labelsize=8)
            axes[2].legend(loc='best', fancybox=False, shadow=False, ncol=3, fontsize=10)
            
            # Velocity misfit (fourth plot)
            axes[3].plot(X, (v_pred0[:, 0] + 1 - (v_fd_on_X + 1)) / ((v_pred0[:, 0] + 1 + v_fd_on_X + 1) / 2) * 100, 
                        color='black', linewidth=1, label="FD")
            if alpha < 0.1:
                axes[3].plot(X, (v_pred0[:, 0] + 1 - (v_lt_on_X + 1)) / ((v_pred0[:, 0] + 1 + (v_lt_on_X + 1)) / 2) * 100, 
                            color='b', linewidth=1, label="LT")
            axes[3].set_xlabel("x", fontsize=18)
            axes[3].grid("True")
            axes[3].minorticks_on()
            axes[3].tick_params(labelsize=8)
            axes[3].legend(loc='best', fancybox=False, shadow=False, ncol=3, fontsize=10)
            axes[3].set_ylabel(r"$\epsilon$ ", fontsize=18)
            axes[3].set_ylim(-5.0, 5.0)
            axes[3].set_xlim(xmin, xmax)
            
        elif plot_type == 'velocity':
            # Create 2x1 subplot for velocity comparison
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Velocity comparison
            axes[0].plot(X, v_pred0, color='c', linewidth=3, label="PINN")
            if alpha < 0.1:
                axes[0].plot(X, v_lt_on_X, linestyle='dashed', color='firebrick', linewidth=2, label="LT")
            axes[0].plot(X, v_fd_on_X, linestyle='solid', color='black', linewidth=1.0, label="FD")
            axes[0].set_xlim(xmin, xmax)
            axes[0].set_title(f"Velocity Solution at t={t} (alpha={alpha:.3f}, lam={lam})")
            axes[0].set_ylabel("v", fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Velocity misfit
            axes[1].plot(X, (v_pred0[:, 0] + 1 - (v_fd_on_X + 1)) / ((v_pred0[:, 0] + 1 + v_fd_on_X + 1) / 2) * 100, 
                        color='black', linewidth=1, label="FD")
            if alpha < 0.1:
                axes[1].plot(X, (v_pred0[:, 0] + 1 - (v_lt_on_X + 1)) / ((v_pred0[:, 0] + 1 + (v_lt_on_X + 1)) / 2) * 100, 
                            color='blue', linewidth=1, label="LT")
            axes[1].set_xlabel("x", fontsize=14)
            axes[1].set_ylabel("Velocity Relative Misfit (%)", fontsize=14)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_ylim(-5.0, 5.0)
            axes[1].set_xlim(xmin, xmax)
            
        else:  # density (default)
            # Create 2x1 subplot (solution and error) for density
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot solutions
            axes[0].plot(X, rho_pred0, color='c', linewidth=3, label="PINN")
            if alpha < 0.1:
                axes[0].plot(X, rho_lt_on_X, linestyle='dashed', color='firebrick', linewidth=2, label="LT")
            axes[0].plot(X, rho_fd_on_X, linestyle='solid', color='black', linewidth=1.0, label="FD")
            axes[0].set_xlim(xmin, xmax)
            axes[0].set_title(f"Hydrodynamics Solution at t={t} (alpha={alpha:.3f}, lam={lam})")
            axes[0].set_ylabel(r"$\rho$", fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Plot relative misfit
            axes[1].plot(X, (rho_pred0[:, 0] - rho_fd_on_X) / ((rho_pred0[:, 0] + rho_fd_on_X) / 2) * 100, 
                        color='black', linewidth=1, label="FD")
            if alpha < 0.1:
                axes[1].plot(X, (rho_pred0[:, 0] - rho_lt_on_X) / ((rho_pred0[:, 0] + rho_lt_on_X) / 2) * 100, 
                            color='blue', linewidth=1, label="LT")
            axes[1].set_xlabel("x", fontsize=14)
            axes[1].set_ylabel("Relative Misfit (%)", fontsize=14)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_ylim(-5.0, 5.0)
            axes[1].set_xlim(xmin, xmax)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Clear memory
        del x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max, v_LT
        del X, rho_pred0, v_pred0, phi_pred0, rho_max_PN, rho_theory
        del rho_fd_on_X, v_fd_on_X
        if alpha < 0.1:
            del rho_lt_on_X, v_lt_on_X
        
        return plot_data
    except Exception as e:
        plt.close('all')  # Clean up any remaining figures
        raise Exception(f"Error in generate_hydro_comparison_plot: {str(e)}")

def generate_hydro_simple_plot(model, t_value, x_min, x_max, plot_type, alpha_value=0.05):
    """Generate simple hydrodynamics plot without comparison"""
    try:
        # Clear any existing matplotlib figures to free memory
        plt.close('all')
        
        res = 1000
        X = np.linspace(x_min, x_max, res).reshape(res, 1)
        t_ = t_value * np.ones(res).reshape(res, 1)
        
        # Use exact same approach as visualisation.py
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        
        # Use the same alpha_coor creation as visualisation.py
        alpha_coor = torch.cuda.FloatTensor([alpha_value]).repeat(res, 1).to(device) if torch.cuda.is_available() else torch.full((res, 1), alpha_value, device=device, dtype=torch.float32)
        test_coor = [pt_x_collocation, pt_t_collocation, alpha_coor]
        
        with torch.no_grad():
            output_0 = model(test_coor)
            rho_pred0 = output_0[:, 0:1].data.cpu().numpy()
            vx_pred0 = output_0[:, 1:2].data.cpu().numpy()
            phi_pred0 = output_0[:, 2:3].data.cpu().numpy()
        
        # Create subplots based on plot_type
        if plot_type == 'all':
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
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
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Clear memory
        del X, t_, pt_x_collocation, pt_t_collocation, alpha_coor, test_coor
        del rho_pred0, vx_pred0, phi_pred0
        
        return plot_data
    except Exception as e:
        plt.close('all')  # Clean up any remaining figures
        raise Exception(f"Error in generate_hydro_simple_plot: {str(e)}")

def generate_burgers_plot_with_comparison(model, t_value, x_min, x_max, plot_type, show_comparison=False):
    """Generate plot for Burgers equation with optional true solution comparison"""
    
    if show_comparison:
        # Use the rel_misfit_burgers logic for comparison plots
        N = 1000
        nu = 0.02  # Default viscosity
        initial_params = (x_min, x_max, 2.0, device)
        
        # Generate comparison plot
        plot_data = generate_burgers_comparison_plot(model, [t_value], initial_params, N, nu, None)
    else:
        # Simple plot without comparison
        plot_data = generate_burgers_simple_plot(model, t_value, x_min, x_max, plot_type)
    
    return plot_data

def generate_burgers_comparison_plot(model, time_array, initial_params, N, nu, true_solution_fn):
    """Generate Burgers plot with true solution comparison using rel_misfit_burgers logic"""
    xmin, xmax, tmax, device = initial_params
    
    # For single time, create a 2x1 subplot (solution and error)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    t = time_array[0]
    res = N
    X = np.linspace(xmin, xmax, res).reshape(res, 1)
    t_ = t * np.ones(res).reshape(res, 1)
    
    pt_x_collocation = torch.from_numpy(X).float().to(device)
    pt_t_collocation = torch.from_numpy(t_).float().to(device)
    
    # Check if network uses parameter embedding
    if hasattr(model, 'use_param_embedding') and model.use_param_embedding:
        nu_input = torch.full((res, 1), nu, device=device, dtype=torch.float32)
        net_input = [pt_x_collocation, pt_t_collocation, nu_input]
    else:
        net_input = [pt_x_collocation, pt_t_collocation]
    
    with torch.no_grad():
        u_pred = model(net_input).cpu().numpy().flatten()
    
    # Use analytical solution for comparison
    from burgers_analytical import burgers_analytical_solution
    u_true = burgers_analytical_solution(X.flatten(), t, nu, initial_condition='sin')
    
    # Plot solutions
    axes[0].plot(X, u_pred, linewidth=2, label="PINN")
    axes[0].plot(X, u_true, '--', linewidth=2, label="Analytical")
    axes[0].set_title(f"Burgers' Equation Solution at t={t}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot absolute error
    abs_err = u_pred - u_true
    axes[1].plot(X, abs_err, color='black', linewidth=1)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title(f"Absolute Error at t={t}")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def generate_burgers_simple_plot(model, t_value, x_min, x_max, plot_type):
    """Generate simple Burgers plot without comparison"""
    res = 1000
    X = np.linspace(x_min, x_max, res).reshape(res, 1)
    t_ = t_value * np.ones(res).reshape(res, 1)
    
    pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
    
    # Always use parameter embedding for Burgers (like alpha in hydrodynamics)
    nu = 0.02  # Default viscosity
    nu_input = torch.full((res, 1), nu, device=device, dtype=torch.float32)
    net_input = [pt_x_collocation, pt_t_collocation, nu_input]
    
    with torch.no_grad():
        output = model(net_input)
        u_pred = output.cpu().numpy()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, u_pred, linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'Burgers Equation Solution at t={t_value}')
    plt.grid(True, alpha=0.3)
    plt.legend(['PINN Solution'])
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

if __name__ == '__main__':
    # Configure Flask app with better timeout and error handling
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Set encoding to UTF-8 to prevent character encoding issues
    import sys
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    # Add error handlers
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    
    # Run with increased timeout and threaded mode
    app.run(
        debug=False,  # Disable debug mode to prevent auto-restart
        host='0.0.0.0', 
        port=5000,
        threaded=True,  # Enable threading for concurrent requests
        use_reloader=False  # Disable auto-reloader
    ) 