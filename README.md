# PINNLab - Basic PINN Training and Visualization

A simple web application for training and visualizing Physics-Informed Neural Networks (PINNs) for basic differential equations.

## What it does

- Train PINN models for simple equations (Burgers, Wave, SHM, Hydrodynamics)
- Basic visualization of training results
- Simple web interface for model training

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Basic mathematical libraries (numpy, scipy, matplotlib)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tirthankar4/PINNLab
   cd Modular_Code_Files_Running_Final
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Run the application**
   ```bash
   # Basic local access
   python app.py
   
   # For external network access (from other devices)
   python app.py --waitress
   ```

2. **Open your browser**
   - Local access: `http://localhost:5000`
   - External access: `http://YOUR_IP_ADDRESS:5000` (when using --waitress)

3. **Train a model**
   - Select equation type
   - Set basic parameters
   - Start training

## Supported Equations

- **Burgers Equation**: Basic fluid dynamics
- **Wave Equation**: Simple wave propagation  
- **SHM**: Simple harmonic motion
- **Hydrodynamics**: Extended GRINN case