# Learning rate schedulers
from torch.optim.lr_scheduler import StepLR


def get_scheduler(optimizer, config):
    """Return a simple StepLR scheduler."""
    step_size = config.get('step_size', 10)
    gamma = config.get('gamma', 0.1)
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
