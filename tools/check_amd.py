"""Read-only environment/import check; --smoke also executes small GPU kernels.

This is not a full TRELLIS.2 generation benchmark and downloads no model weights.
"""
import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

# Works both as a standalone script and as an import in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from install_amd import validate_stack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "errors": [], "gpu_smoke": "not run"}
    if sys.version_info[:2] != (3, 12):
        report["errors"].append("Bundled wheels require CPython 3.12.")
    try:
        import torch
        report.update(torch=str(torch.__version__), hip=torch.version.hip)
        validate_stack(torch)
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, arch=getattr(props, "gcnArchName", "unknown"),
                      vram_gib=round(props.total_memory / 2**30, 2))
    except (ImportError, RuntimeError) as exc:
        report["errors"].append(str(exc))
        print(json.dumps(report, indent=2))
        return 1
    for name in ("triton", "cumesh", "flex_gemm.ops.grid_sample", "o_voxel", "nvdiffrast.torch", "aule"):
        try:
            importlib.import_module(name)
            report[name] = "import OK"
        except Exception as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    if args.smoke and not report["errors"]:
        try:
            from aule import flash_attention_triton
            from nvdiffrast import torch as dr
            with torch.inference_mode():
                q, k, v = [torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                actual = flash_attention_triton(q, k, v, causal=False)
                torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
                # Exercise a native ROCm rasterizer kernel, not just its Python import.
                context = dr.RasterizeCudaContext(device=torch.device("cuda"))
                vertices = torch.tensor([[[-0.8, -0.8, 0., 1.], [0.8, -0.8, 0., 1.],
                                          [0., 0.8, 0., 1.]]], device="cuda")
                triangles = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
                raster, _ = dr.rasterize(context, vertices, triangles, resolution=[32, 32])
                if not bool((raster[..., 3] > 0).any()):
                    raise RuntimeError("Rasterizer produced no triangle coverage.")
                torch.cuda.synchronize()
                report["gpu_smoke"] = "Aule vs SDPA + nvdiffrast triangle: PASS"
        except Exception as exc:
            report["errors"].append(f"GPU smoke: {type(exc).__name__}: {exc}")
            report["gpu_smoke"] = "FAIL"
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
