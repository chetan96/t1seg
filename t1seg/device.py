"""Hardware selection. 'auto' prefers CUDA, then Apple MPS, then CPU."""
import torch


def available_devices():
    out = []
    if torch.cuda.is_available():
        out += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        out.append("mps")
    return out + ["cpu"]


def resolve_device(spec="auto"):
    """spec: 'auto' | 'cuda' | 'cuda:1' | 'mps' | 'cpu'. An explicit but unavailable device is an error,
    so a run never silently lands on the CPU."""
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and (not torch.cuda.is_available() or (dev.index or 0) >= torch.cuda.device_count()):
        raise RuntimeError(f"device '{spec}' requested but not available; have {available_devices()}")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"device 'mps' requested but not available; have {available_devices()}")
    return dev


def loader_kwargs(device):
    """DataLoader settings that depend on the accelerator."""
    return dict(pin_memory=device.type == "cuda")
