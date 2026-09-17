from .nodes import NODE_CLASS_MAPPINGS as _UPSTREAM_NODES
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _UPSTREAM_NAMES
from .amd_nodes import Trellis2LoadModelAMD

# Keep the upstream node ID so existing single-view and MultiView graphs load.
NODE_CLASS_MAPPINGS = dict(_UPSTREAM_NODES)
NODE_DISPLAY_NAME_MAPPINGS = dict(_UPSTREAM_NAMES)
NODE_CLASS_MAPPINGS["Trellis2LoadModel"] = Trellis2LoadModelAMD
NODE_DISPLAY_NAME_MAPPINGS["Trellis2LoadModel"] = "Trellis2 - LoadModel (AMD / ROCm)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
