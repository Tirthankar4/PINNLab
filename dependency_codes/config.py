MODEL_TYPE = 'hydro'  # Options: 'hydro', 'burgers', 'wave'

tmin = 0.
xmin = 0.
ymin= 0.
zmin= 0.

cs = 1.0
rho_o = 1
const = 1
G = 1


iteration_adam_1D = 500
iteration_lbgfs_1D = 51

iteration_adam_2D = 1000
iteration_lbgfs_2D = 201

iteration_adam_3D = 10
iteration_lbgfs_3D = 2

# Activation function options (set to None to use defaults in code)
PARAM_EMBED_ACTIVATION = None  # e.g., 'tanh', 'relu', 'sin', etc.
INPUT_EMBED_ACTIVATION = None  # e.g., 'sin', 'relu', etc.
MAIN_PINN_ACTIVATION = None    # e.g., 'sin', 'relu', etc.

# Deployment Configuration
import os

# Server settings with environment variable fallbacks
HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
PORT = int(os.environ.get('FLASK_PORT', 5000))
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = os.environ.get('LOG_FILE', 'app.log')

# Model storage settings
MODEL_STORAGE_PATH = os.environ.get('MODEL_STORAGE_PATH', 'pretrained_models')
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB default
