"""CPU regression tests; mocked Aule dispatch is NOT a GPU kernel validation."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "_trellis_amd_tests"


def package(name, path):
    obj = types.ModuleType(name)
    obj.__path__ = [str(path)]
    sys.modules[name] = obj
    return obj


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


package(PREFIX, ROOT)
runtime = load(PREFIX + ".amd_runtime", ROOT / "amd_runtime.py")
package(PREFIX + ".trellis2", ROOT / "trellis2")
package(PREFIX + ".trellis2.modules", ROOT / "trellis2/modules")
attention_pkg = package(PREFIX + ".trellis2.modules.attention", ROOT / "trellis2/modules/attention")
sparse_pkg = package(PREFIX + ".trellis2.modules.sparse", ROOT / "trellis2/modules/sparse")
package(PREFIX + ".trellis2.modules.sparse.attention", ROOT / "trellis2/modules/sparse/attention")
sparse_cfg = load(sparse_pkg.__name__ + ".config", ROOT / "trellis2/modules/sparse/config.py")
dense_cfg = load(attention_pkg.__name__ + ".config", ROOT / "trellis2/modules/attention/config.py")
sparse_pkg.config = sparse_cfg
attention_pkg.config = dense_cfg


class SparseFixture:
    """Only the tensor/cache protocol; no native convolution extension needed."""
    def __init__(self, feats, coords, batch_size=2, cache=None):
        self.feats, self.coords = feats, coords
        self.batch_size = batch_size
        self.cache = {} if cache is None else cache

    @property
    def shape(self):
        return (self.batch_size, *self.feats.shape[1:])

    @property
    def device(self):
        return self.feats.device

    def get_spatial_cache(self, key):
        return self.cache.get(key)

    def register_spatial_cache(self, key, value):
        self.cache[key] = value

    def replace(self, feats):
        return SparseFixture(feats, self.coords, self.batch_size, self.cache)


sparse_pkg.SparseTensor = SparseFixture
window = load(sparse_pkg.__name__ + ".attention.windowed_attn",
              ROOT / "trellis2/modules/sparse/attention/windowed_attn.py")


def fake_aule(q, k, v, causal=False):
    assert not causal
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def reference(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None],
        v.transpose(0, 1)[None])[0].transpose(0, 1)


class BackendTests(unittest.TestCase):
    def test_rocm_auto_aule(self):
        self.assertEqual(runtime.select_backend("auto", rocm=True, available=lambda _: True), "aule")

    def test_auto_without_extras(self):
        for rocm in (True, False):
            self.assertEqual(runtime.select_backend("auto", rocm=rocm, available=lambda _: False), "sdpa")

    def test_nvidia_auto(self):
        self.assertEqual(runtime.select_backend("auto", rocm=False, available=lambda _: True), "flash_attn")

    def test_legacy_workflow_migrates_to_aule(self):
        for backend in ("flash_attn", "xformers", "flash_attn_3"):
            self.assertEqual(runtime.select_backend(backend, rocm=True,
                             available=lambda name: name == "aule"), "aule")

    def test_explicit_installed_flash_preserved(self):
        self.assertEqual(runtime.select_backend("flash_attn", rocm=True,
                         available=lambda _: True), "flash_attn")

    def test_explicit_missing_aule_errors(self):
        with self.assertRaisesRegex(RuntimeError, "requirements-amd"):
            runtime.select_backend("aule", rocm=True, available=lambda _: False)

    def test_unknown_backend_errors(self):
        with self.assertRaises(ValueError):
            runtime.select_backend("typo", rocm=True)

    def test_sdpa_is_explicitly_supported_in_both_configs(self):
        sparse_cfg.set_attn_backend("sdpa")
        dense_cfg.set_backend("sdpa")
        self.assertEqual((sparse_cfg.ATTN, dense_cfg.BACKEND), ("sdpa", "sdpa"))

    def test_config_validation(self):
        for setter in (sparse_cfg.set_attn_backend, sparse_cfg.set_conv_backend, dense_cfg.set_backend):
            with self.assertRaises(ValueError):
                setter("not-a-backend")


class WindowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        sparse_cfg.set_attn_backend("sdpa")
        self.coords = torch.tensor([[0, 4, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0],
                                    [0, 1, 0, 0], [1, 1, 0, 0], [0, 5, 0, 0]], dtype=torch.int32)

    def fixture(self, n=None, packed=3):
        n = len(self.coords) if n is None else n
        return SparseFixture(torch.randn(n, packed, 2, 8), self.coords[:n])

    def test_self_sdpa_matches_isolated_windows(self):
        x = self.fixture()
        result = window.sparse_windowed_scaled_dot_product_self_attention(x, 2)
        expected = torch.empty(6, 2, 8)
        for indices in ([0, 5], [1, 4], [2, 3]):
            expected[indices] = reference(*x.feats[indices].unbind(1))
        torch.testing.assert_close(result.feats, expected)
        self.assertIs(result.coords, x.coords)

    def test_aule_dispatch_and_backend_switch_reuses_partition(self):
        x = self.fixture()
        expected = window.sparse_windowed_scaled_dot_product_self_attention(x, 2).feats
        cache_value = next(iter(x.cache.values()))
        sparse_cfg.set_attn_backend("aule")
        with patch.dict(sys.modules, {"aule": types.SimpleNamespace(flash_attention_triton=fake_aule)}):
            got = window.sparse_windowed_scaled_dot_product_self_attention(x, 2).feats
        torch.testing.assert_close(got, expected)
        self.assertIs(next(iter(x.cache.values())), cache_value)

    def test_self_empty(self):
        x = self.fixture(0)
        out = window.sparse_windowed_scaled_dot_product_self_attention(x, 2)
        self.assertEqual(tuple(out.feats.shape), (0, 2, 8))

    def test_empty_partition(self):
        fwd, bwd, lengths, args = window.calc_window_partition(self.fixture(0), 2)
        self.assertEqual((len(fwd), len(bwd), len(lengths), args["max_seqlen"]), (0, 0, 0, 0))

    def test_window_size_validation(self):
        for size in (0, -1, (1, 2)):
            with self.assertRaises(ValueError):
                window.calc_window_partition(self.fixture(), size)

    def test_shift_and_negative_coordinates(self):
        x = self.fixture()
        fwd, bwd, lengths, _ = window.calc_window_partition(x, 2, (-3, 0, 0))
        self.assertTrue(torch.equal(fwd[bwd], torch.arange(6)))
        self.assertEqual(int(lengths.sum()), 6)
        self.assertEqual(len(lengths), 6)

    def test_cross_pairs_by_coordinates_not_nonempty_ordinal(self):
        q = SparseFixture(torch.randn(3, 2, 8), torch.tensor([[0, 0, 0, 0], [0, 4, 0, 0], [1, 0, 0, 0]]))
        kv = SparseFixture(torch.randn(3, 2, 2, 8), torch.tensor([[0, 2, 0, 0], [0, 4, 0, 0], [1, 0, 0, 0]]))
        out = window.sparse_windowed_scaled_dot_product_cross_attention(q, kv, 2, 2)
        expected = torch.zeros_like(q.feats)
        expected[1:2] = reference(q.feats[1:2], *kv.feats[1:2].unbind(1))
        expected[2:3] = reference(q.feats[2:3], *kv.feats[2:3].unbind(1))
        torch.testing.assert_close(out.feats, expected)
        self.assertIs(out.coords, q.coords)

    def test_cross_different_window_scales(self):
        q = SparseFixture(torch.randn(2, 2, 8), torch.tensor([[0, 2, 0, 0], [0, 0, 0, 0]]))
        kv = SparseFixture(torch.randn(2, 2, 2, 8), torch.tensor([[0, 0, 0, 0], [0, 4, 0, 0]]))
        out = window.sparse_windowed_scaled_dot_product_cross_attention(q, kv, 2, 4)
        expected = torch.stack((kv.feats[1, 1], kv.feats[0, 1]))
        torch.testing.assert_close(out.feats, expected)

    def test_cross_aule(self):
        qkv = self.fixture()
        q = qkv.replace(qkv.feats[:, 0])
        kv = qkv.replace(qkv.feats[:, 1:])
        expected = window.sparse_windowed_scaled_dot_product_cross_attention(q, kv, 2, 2).feats
        sparse_cfg.set_attn_backend("aule")
        with patch.dict(sys.modules, {"aule": types.SimpleNamespace(flash_attention_triton=fake_aule)}):
            out = window.sparse_windowed_scaled_dot_product_cross_attention(q, kv, 2, 2)
        torch.testing.assert_close(out.feats, expected)

    def test_cross_empty_kv(self):
        q = self.fixture().replace(torch.randn(6, 2, 8))
        kv = SparseFixture(torch.empty(0, 2, 2, 8), torch.empty(0, 4, dtype=torch.int32))
        out = window.sparse_windowed_scaled_dot_product_cross_attention(q, kv, 2, 2)
        self.assertEqual(int(out.feats.count_nonzero()), 0)

    def test_cross_empty_q(self):
        q = SparseFixture(torch.empty(0, 2, 8), torch.empty(0, 4, dtype=torch.int32))
        kv = self.fixture(packed=2)
        out = window.sparse_windowed_scaled_dot_product_cross_attention(q, kv, 2, 2)
        self.assertEqual(tuple(out.feats.shape), (0, 2, 8))

    def test_gradients_are_preserved(self):
        x = self.fixture()
        x.feats.requires_grad_(True)
        out = window.sparse_windowed_scaled_dot_product_self_attention(x, 2)
        out.feats.square().sum().backward()
        self.assertTrue(torch.isfinite(x.feats.grad).all())

    def test_unknown_dispatch_errors(self):
        sparse_cfg.ATTN = "typo"
        with self.assertRaises(ValueError):
            window.sparse_windowed_scaled_dot_product_self_attention(self.fixture(), 2)


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Parent:
            schema = {"required": {"backend": (["flash_attn"], {"default": "flash_attn"}),
                                   "sparse_backend": (["flash_attn"], {"default": "flash_attn"}),
                                   "pixal3d_multiview": ("BOOLEAN", {"default": False})}}
            @classmethod
            def INPUT_TYPES(cls):
                return cls.schema
            def process(self, **kwargs):
                return kwargs
        cls.parent = Parent
        with patch.dict(sys.modules, {PREFIX + ".nodes": types.SimpleNamespace(Trellis2LoadModel=Parent)}):
            cls.module = load(PREFIX + ".amd_nodes", ROOT / "amd_nodes.py")

    def test_schema_retains_multiview_and_does_not_mutate_upstream(self):
        schema = self.module.Trellis2LoadModelAMD.INPUT_TYPES()
        self.assertIn("pixal3d_multiview", schema["required"])
        self.assertEqual(schema["required"]["backend"][1]["default"], "auto")
        self.assertEqual(self.parent.schema["required"]["backend"][1]["default"], "flash_attn")

    def args(self):
        return dict(modelname="microsoft/TRELLIS.2-4B", backend="sdpa", device="cuda",
                    low_vram=True, keep_models_loaded=True, conv_backend="flex_gemm",
                    sparse_backend="sdpa", use_reconviagen=False, pixal3d_multiview=True)

    def test_multiview_flag_passed_through(self):
        with patch.object(torch.version, "hip", "10.0"), patch.object(torch.cuda, "is_available", return_value=True):
            out = self.module.Trellis2LoadModelAMD().process(**self.args())
        self.assertTrue(out["pixal3d_multiview"])
        self.assertEqual(out["backend"], "sdpa")

    def test_rocm_wrong_device_rejected(self):
        args = self.args()
        args["device"] = "cpu"
        with patch.object(torch.version, "hip", "10.0"), self.assertRaisesRegex(ValueError, "device=cuda"):
            self.module.Trellis2LoadModelAMD().process(**args)

    def test_rocm_wrong_conv_rejected(self):
        args = self.args()
        args["conv_backend"] = "spconv"
        with patch.object(torch.version, "hip", "10.0"), patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "flex_gemm"):
                self.module.Trellis2LoadModelAMD().process(**args)

    def test_missing_gpu_rejected_before_download(self):
        with patch.object(torch.version, "hip", "10.0"), patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cannot see the GPU"):
                self.module.Trellis2LoadModelAMD().process(**self.args())


if __name__ == "__main__":
    unittest.main()
