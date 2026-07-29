import torch


def get_device(requested_device: str = "auto") -> torch.device:
    """
    Returns the appropriate PyTorch device based on user preference and hardware support.
    
    Args:
        requested_device (str): 'auto', 'cuda', 'mps', or 'cpu'.
    
    Returns:
        torch.device: Torch device object.
    """
    if requested_device != "auto":
        return torch.device(requested_device)
        
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
