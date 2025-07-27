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

1. **Choose Model Type**: Select between hydrodynamics or Burgers' equation
2. **Configure Architecture**: Set neural network parameters (layers, neurons, activations)
3. **Set Training Parameters**: Configure learning rates, iterations, and batch sizes
4. **Start Training**: Monitor real-time progress with detailed metrics
5. **Visualize Results**: Generate plots from your trained model

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
├── train.py                        # Standalone training script
├── config.py                       # Configuration parameters
├── requirements.txt                 # Python dependencies
├── models/                         # Neural network architectures
│   ├── base.py                     # Base model class
│   ├── hydro.py                    # Hydrodynamics PINN
│   └── burgers.py                  # Burgers' equation PINN
├── losses/                         # Loss function implementations
│   ├── base.py                     # Base loss class
│   ├── hydro.py                    # Hydrodynamics losses
│   ├── burgers.py                  # Burgers' losses
│   └── registry.py                 # Loss function registry
├── templates/                      # Web interface templates
│   └── index.html                  # Main web interface
├── visualisation.py                # Hydrodynamics plotting functions
├── visualisation_burgers.py        # Burgers' equation plotting
├── data_generator.py               # Training data generation
├── solver.py                       # Training algorithms
├── LAX.py                         # Lax-Wendroff numerical solver
├── burgers_analytical.py           # Analytical Burgers' solutions
└── *.pth                          # Pretrained model files
```

## 🔧 Configuration

### Model Configuration (`config.py`)
```python
MODEL_TYPE = 'hydro'  # 'hydro' or 'burgers'
tmin = 0.0           # Minimum time
xmin = 0.0           # Minimum x-coordinate
rho_o = 1            # Reference density
```

### Training Parameters
- **Adam Iterations**: 500 (default)
- **L-BFGS Iterations**: 50 (default)
- **Learning Rate**: 0.001 (default)
- **Batch Size**: 1000 (default)

## 🧪 Model Types

### Hydrodynamics Models
- **Case 1**: Low amplitude waves (α: 0.01-0.08)
- **Case 2**: High amplitude waves (α: 0.1-0.8)
- **Case 3**: Different wavelength configuration
- **Mixed Model**: Combined architecture

### Burgers' Equation Models
- **Viscosity Range**: Configurable viscosity parameter
- **Analytical Comparison**: Compare with exact solutions
- **Parameter Embedding**: Viscosity as embedded parameter

## 📊 Visualization Features

### Plot Types
- **Density**: Fluid density distribution over space
- **Velocity**: Velocity field visualization
- **All Variables**: Combined density, velocity, and potential plots
- **Comparison**: PINN vs analytical/numerical solutions

### Error Analysis
- **Relative Misfit**: Percentage error calculations
- **Absolute Error**: Direct difference measurements
- **Statistical Analysis**: Error distribution and statistics

## 🔬 Technical Details

### Neural Network Architecture
- **Multi-layer Perceptron**: Configurable depth and width
- **Parameter Embedding**: Separate networks for parameter encoding
- **Input Embedding**: Coordinate transformation layers
- **Activation Functions**: Flexible activation function selection

### Training Algorithm
- **Adam Optimizer**: Initial training phase
- **L-BFGS Optimizer**: Fine-tuning phase
- **Batch Training**: Memory-efficient training
- **Progress Tracking**: Real-time training metrics

### Numerical Methods
- **Finite Difference**: Lax-Wendroff scheme for comparison
- **Linear Theory**: Analytical approximations
- **Collocation Points**: Physics-informed training data

## 🚀 Performance

### Hardware Acceleration
- **CUDA Support**: GPU acceleration for training
- **MPS Support**: Apple Silicon optimization
- **CPU Fallback**: Automatic CPU usage when GPU unavailable

### Memory Management
- **Batch Processing**: Efficient memory usage
- **Gradient Accumulation**: Large model training support
- **Memory Cleanup**: Automatic GPU memory management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Physics-Informed Neural Networks (PINNs) methodology
- PyTorch for deep learning framework
- Flask for web application framework
- Matplotlib for scientific visualization

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

## 🔄 Version History

- **v1.0.0**: Initial release with hydrodynamics and Burgers' equation support
- **v1.1.0**: Added real-time training progress tracking
- **v1.2.0**: Enhanced visualization with comparison plots
- **v1.3.0**: Improved web interface and parameter embedding

---

**Note**: PINNLab is designed for research and educational purposes. For production use, additional testing and validation are recommended. 