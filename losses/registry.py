"""
losses/registry.py
Central registry for PINN loss functions. Provides a decorator to register loss functions 
and a dispatcher to call them by name.
"""

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

def loss_fn(name, *args, **kwargs):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss function '{name}' not found in registry.")
    return LOSS_REGISTRY[name](*args, **kwargs) 