import torch

class DeviceSelector:
    def __init__(self):
        self.device = self._select_device()

    def _select_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self._configure_cuda()
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            self._warn_about_mps()
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device

    def _configure_cuda(self):
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _warn_about_mps(self):
        print("\nSupport for MPS devices is preliminary...")
