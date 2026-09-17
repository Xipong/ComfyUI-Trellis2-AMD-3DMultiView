# TRELLIS.2 AMD + 3D MultiView for ComfyUI

This fork combines the newer **visualbruno/Xipong MultiView wrapper** with
**dmonkman's ROCm native wheels and Aule attention integration**. It is a ComfyUI
custom node, not a fork of ComfyUI itself.

**Validation boundary:** the AMD donor reports end-to-end testing on an RX 6800 XT
(`gfx1030`), Windows 11 and Ubuntu 24.04. This merged fork has CPU regression tests,
but has **not yet been end-to-end tested on an AMD GPU**. A passing CPU test or import
check is not proof that every native GPU kernel or workflow works. Pixal3D/NATTEN,
FP8 and DCx paths are not claimed as AMD-validated.

## What is combined

- All wrapper source and workflows through `14597418bbe33a440ead4667e2966408f0524a21`,
  including TRELLIS.2 MultiView nodes, newer Pixal3D MultiView code and the new
  `model_3d` export. These additions are preserved, not replaced with older nodes.
- ROCm CuMesh, FlexGEMM, O-Voxel and nvdiffrast wheels from
  `dmonkman/ComfyUI-Trellis2-AMD@3aa728ad71b6f892a4c3074e5a38c17dd9f0eec5`.
- Dense and variable-length Aule attention from the AMD donor, plus completed
  Aule/SDPA **windowed self- and cross-attention**. Cross-attention pairs occupied
  windows by coordinates rather than assuming their ordinal positions match.
- An AMD-aware loader under the existing `Trellis2LoadModel` node ID. `auto` prefers
  Aule on ROCm. Old workflows referencing missing NVIDIA attention packages are
  redirected before loading models, with a warning.
- A guarded wheel installer, environment/GPU smoke checker, and CPU regression tests.

The donor's dangling `trellis2_profiler` import is deliberately not carried over.
Neither the NVIDIA wheels nor the donor's pinned ComfyUI frontend/manager packages
are installed by this fork. Exact provenance is recorded in [UPSTREAMS.json](UPSTREAMS.json).
The original MIT license and upstream attribution remain in place.

## Target runtime

Use **CPython 3.12 x64 + PyTorch 2.13.0 + ROCm 10.0** for the bundled native wheels.
Do not put these wheels into an existing CUDA, DirectML, ZLUDA, ROCm 7.x or
ABI-incompatible PyTorch environment. The donor targets Windows 11 and Ubuntu
24.04; other Linux distributions/WSL are not validated here. This integration does
not add missing `gfx1030` support to a PyTorch build: the matching device runtime
must already work.

The repository retains the donor's Python 3.13/3.14 wheels for provenance, but the
installer deliberately selects **3.12 only** because the full dependency stack
is not validated for the other interpreters.

## Installation: Windows / RX 6800 XT

Use an isolated ComfyUI checkout/environment. Do not install both this fork and
another TRELLIS.2 wrapper into the same `custom_nodes` folder: their node IDs collide.
Run everything with the **same Python that starts ComfyUI**, not a different global Python.

Create and activate a clean Python 3.12 venv, then install the donor's pinned runtime:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "torch[device-gfx1030]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1030]==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0" `
  "rocm[libraries,device-gfx1030]==10.0.0"
```

For a different GPU, choose its supported `device-gfx...` extras; do not spoof the
architecture. Availability of a wheel is not a validation claim for that GPU.
Verify the runtime **before** installing any native extensions:

```powershell
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Clone the custom node into an existing ComfyUI checkout:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/Xipong/ComfyUI-Trellis2-AMD-3DMultiView.git
cd ComfyUI-Trellis2-AMD-3DMultiView
# In a new ComfyUI venv, install ComfyUI's own requirements without changing this stack:
python -m pip install -c constraints-rocm.txt -r ..\..\requirements.txt
python tools/install_amd.py
python tools/check_amd.py --smoke
```

`install_amd.py` checks the Python/Torch/HIP ABI and GPU visibility, installs only
the four platform-matched ROCm wheels with `--no-deps`, then installs Python
requirements under constraints for the existing Torch stack. Use `--dry-run`
to inspect the plan without installing anything.

`check_amd.py --smoke` tests imports, a small Aule-vs-SDPA calculation, and an
nvdiffrast triangle rasterization. It downloads **no model weights**. CuMesh,
FlexGEMM and O-Voxel are import-checked, not fully kernel-tested by this command.

## Linux

Use Python 3.12 on the donor's Ubuntu 24.04 target. Create a venv with
`python3.12 -m venv venv` and activate it with `source venv/bin/activate`.
Use the same runtime pip command (replace PowerShell backticks with shell `\`
line continuations). After cloning, run:

```bash
python -m pip install -c constraints-rocm.txt -r ../../requirements.txt
python tools/install_amd.py
python tools/check_amd.py --smoke
```

The installer selects `wheels/Linux/Python3.12` automatically. Do not install a
CUDA Triton package over the ROCm Triton provided by the runtime.

## First generation on a 16 GB RX 6800 XT

Start with the **ordinary TRELLIS.2 model**, not FP8 or Pixal3D:

| LoadModel setting | Start with |
| --- | --- |
| modelname | `microsoft/TRELLIS.2-4B` |
| backend / sparse_backend | `auto` (normally selects `aule` on ROCm) |
| conv_backend | `flex_gemm` |
| device | `cuda` — this is also the correct PyTorch device name for AMD HIP |
| low_vram | `true` |
| pixal3d_multiview | `false` — this flag is for Pixal3D weights, not ordinary TRELLIS.2 MultiView |

Start at 512 resolution with a single-view example, then switch to the TRELLIS.2
MultiView workflows in `example_workflows/`. Use consistent front/side/back images
of the **same object**. MultiView adds conditioning; it is not a guarantee of
perfect reconstruction or four times the quality.

Obtain the required TRELLIS.2 and DINOv3 weights according to their upstream access
and license terms. DINOv3 belongs under
`ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m/`. Upstream loaders may
download missing model files on the first run; allow sufficient disk space.

### Boundaries and troubleshooting

- **No HIP/GPU in the checker:** fix the ROCm PyTorch/device runtime first. Changing
  a ComfyUI node cannot fix an unsupported driver, missing architecture or CPU Torch.
- **Native import/ABI errors:** use the exact stack above; installing CUDA wheels
  with the same Python tag does not make them ROCm-compatible.
- **Aule missing:** install `requirements-amd.txt` in the correct environment.
  Explicit `aule` requests fail clearly instead of silently using a high-memory path.
- **SDPA runs out of VRAM:** use Aule and low-VRAM mode; on hardware without a fused
  SDPA kernel its memory use can become quadratic. No automatic VRAM capacity claim
  is made for high-resolution or multi-view jobs.
- **Pixal3D MultiView:** source and workflows are retained, but its NATTEN dependency
  is not AMD-validated here. Do not interpret source preservation as working NATTEN.
- **DCx / FP8 / projection extras:** retained upstream nodes may require additional
  native extensions or unsupported GPU operations. No AMD DCx wheel is bundled.

## Development and verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

CPU tests cover backend selection, loader forwarding, native-wheel ABI guards,
window pairing, empty inputs, gradient propagation, and dense/sparse attention
packing. Aule calls are mocked in CPU tests; only the optional GPU checker executes
real Aule kernels. The full-checkout tests verify preserved/imported Git blob hashes.

Credits: [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2),
[visualbruno's wrapper](https://github.com/visualbruno/ComfyUI-Trellis2),
[dmonkman's AMD port](https://github.com/dmonkman/ComfyUI-Trellis2-AMD), and their
respective dependencies/contributors. Historical READMEs are in `docs/`; their
installation commands may not apply to this combined fork.
