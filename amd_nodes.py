"""Small AMD loader adapter; upstream nodes and all MultiView nodes stay intact."""
import copy
import torch

from .amd_runtime import BACKENDS, select_backend
from .nodes import Trellis2LoadModel


class Trellis2LoadModelAMD(Trellis2LoadModel):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        for name in ("backend", "sparse_backend"):
            inputs["required"][name] = (list(BACKENDS), {"default": "auto"})
        return inputs

    def process(self, modelname, backend, device, low_vram, keep_models_loaded,
                conv_backend, sparse_backend, use_reconviagen,
                pixal3d_multiview=False):
        rocm = bool(torch.version.hip)
        if rocm:
            if device != "cuda":
                raise ValueError("Select device=cuda for AMD ROCm: PyTorch uses the "
                                 "torch.cuda API for HIP GPUs too.")
            if not torch.cuda.is_available():
                raise RuntimeError("ROCm PyTorch cannot see the GPU. Run "
                                   "python tools/check_amd.py in this environment.")
            if conv_backend != "flex_gemm":
                raise ValueError("This fork ships ROCm FlexGEMM wheels. Select "
                                 "conv_backend=flex_gemm; spconv/torchsparse are not bundled.")
        return super().process(
            modelname=modelname,
            backend=select_backend(backend, rocm=rocm),
            device=device, low_vram=low_vram, keep_models_loaded=keep_models_loaded,
            conv_backend=conv_backend,
            sparse_backend=select_backend(sparse_backend, rocm=rocm),
            use_reconviagen=use_reconviagen, pixal3d_multiview=pixal3d_multiview,
        )
