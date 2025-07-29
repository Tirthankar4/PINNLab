# PINNLab - PINN Trainer and Visualizer

A comprehensive web application for training, visualizing, and comparing Physics-Informed Neural Networks (PINNs) for solving partial differential equations. This project supports both hydrodynamics and Burgers' equation models with real-time training progress tracking and interactive visualization capabilities.

## 🚀 Features

### Core Functionality
- **Multi-Model Support**: Train and visualize PINNs for hydrodynamics and Burgers' equations
- **Real-time Training**: Live progress tracking with detailed metrics during model training
- **Interactive Visualization**: Generate plots with optional comparison to analytical/numerical solutions
- **Pretrained Models**: Access to pre-trained models for immediate visualization
- **Parameter Embedding**: Advanced neural network architectures with parameter embedding capabilities
- **Web Interface**: Modern, responsive web UI built with Flask and HTML/CSS/JavaScript

### Model Types
- **Hydrodynamics Models**: Solve fluid dynamics equations with customizable parameters
- **Burgers' Equation Models**: Solve the viscous Burgers' equation with analytical solution comparison
- **Mixed Models**: Combined architectures for enhanced performance

### Visualization Options
- **Density Plots**: Visualize fluid density distributions
- **Velocity Plots**: Show velocity field solutions
- **Comparison Plots**: Compare PINN solutions with Finite Difference (FD) and Linear Theory (LT) solutions
- **Error Analysis**: Relative misfit calculations and error visualization

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 4GB+ RAM recommended

## 🛠️ Installation

1. **Clone the repository**
   ```bash
       git clone https://github.com/yourusername/pinnlab.git
    cd pinnlab
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Start exploring**
   - Use pretrained models for immediate visualization
   - Train new models with custom parameters
   - Generate comparison plots with analytical solutions

## 📖 Usage Guide

### Using Pretrained Models

1. **Select a Model**: Choose from available pretrained models in the dropdown
2. **Set Parameters**: Configure time, spatial domain, and model-specific parameters
3. **Generate Plot**: Click "Visualize" to create plots
4. **Enable Comparison**: Toggle comparison with analytical/numerical solutions

### Training New Models

1. **Choose Model Type**: Select between hydrodynamics, Burgers' equation, or wave equation
2. **Configure Architecture**: Set neural network parameters (layers, neurons, activations)
3. **Set Training Parameters**: Configure learning rates, iterations, and batch sizes
4. **Choose Save Option**: 
   - **Save Model**: Check to save the trained model to disk for later use
   - **Session Only**: Leave unchecked to use the model only for the current session
5. **Start Training**: Monitor real-time progress with detailed metrics
6. **Visualize Results**: Generate plots from your trained model

### Command Line Training

You can also train models from the command line:

```bash
# Train without saving (default)
python train.py

# Train and save the model
python train.py --save_model
```

The command-line training will:
- Use the model type specified in `config.py`
- Save models to `pretrained_models/{model_type}/` folder if `--save_model` is used
- Generate plots and comparison results automatically

### Advanced Configuration

#### Model Architecture Parameters
- **Neural Network Layers**: Configure depth and width of networks
- **Activation Functions**: Choose from tanh, sin, relu, and other activations
- **Parameter Embedding**: Enable/disable parameter embedding layers
- **Input Embedding**: Configure input coordinate embedding

#### Training Parameters
- **Adam Optimizer**: Learning rate and iteration count
- **L-BFGS Optimizer**: Fine-tuning iterations
- **Batch Training**: Configure batch size and number of batches
- **Domain Configuration**: Set spatial and temporal domains

## 🏗️ Project Structure

```
PINNLab/
├── app.py                          # Main Flask application
├── train.py                        # Training script for command line
├── inference.py                    # Inference script
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Neural network architectures
│   ├── base.py                     # Base PINN architecture
│   ├── hydro.py                    # Hydrodynamics PINN
│   ├── burgers.py                  # Burgers equation PINN
│   └── wave.py                     # Wave equation PINN
│
├── losses/                         # Loss functions
│   ├── base.py                     # Base loss functions
│   ├── hydrodynamic.py             # Hydrodynamics losses
│   ├── burgers.py                  # Burgers equation losses
│   ├── wave.py                     # Wave equation losses
│   ├── losses.py                   # Loss registry and utilities
│   └── registry.py                 # Loss function registry
│
├── visualisations/                 # Visualization modules
│   ├── __init__.py                 # Package initialization
│   ├── visualisation.py            # Hydrodynamics visualizations
│   ├── visualisation_burgers.py    # Burgers equation visualizations
│   └── visualisation_wave.py       # Wave equation visualizations
│
├── pretrained_models/              # Organized pretrained models
│   ├── hydrodynamics/              # Hydrodynamics models
│   │   ├── Case1_final_part1.pth   # Case 1 model
│   │   ├── Case2_final_part1.pth   # Case 2 model
│   │   ├── Case3_final_part1.pth   # Case 3 model
│   │   └── mixed_final_part1.pth   # Mixed model
│   ├── burgers/                    # Burgers equation models
│   │   └── (future models)
│   └── wave/                       # Wave equation models
│       └── (future models)
│
├── templates/                      # HTML templates
│   └── index.html                  # Main web interface
│
├── analytical_solutions/           # Analytical and numerical solution generators
│   ├── __init__.py                 # Package initialization
│   ├── hydrodynamics/              # Hydrodynamics solutions
│   │   ├── __init__.py             # Hydrodynamics package init
│   │   └── LAX.py                  # Lax-Wendroff Finite Difference solver
│   ├── burgers/                    # Burgers equation solutions
│   │   ├── __init__.py             # Burgers package init
│   │   └── burgers_analytical.py   # Burgers equation analytical solutions
│   └── wave/                       # Wave equation solutions
│       ├── __init__.py             # Wave package init
│       └── wave_analytical.py      # Wave equation analytical solutions
│
└── legacy/                         # Legacy code
    ├── model_architecture.py       # Legacy PINN architecture
    └── legacy_pinn.py              # Legacy PINN class
```

## 🔧 Configuration

### Model Configuration (`