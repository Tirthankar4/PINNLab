import torch

# General derivative calculator using autograd
def diff(y, x, order=1):
    """
    Compute the nth derivative of y with respect to x using autograd.
    y: output tensor
    x: input tensor
    order: order of the derivative
    """
    for _ in range(order):
        grads = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        y = grads
    return y

# General MSE loss (wrapper for torch.nn.functional.mse_loss)
def mse_loss(pred, target):
    return torch.nn.functional.mse_loss(pred, target) 