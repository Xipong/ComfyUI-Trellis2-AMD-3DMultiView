"""Exercise the exact imported AMD attention source using real CPU PyTorch.

Aule is mocked with SDPA: these tests cover packing/dispatch/variable lengths,
not the native Aule kernel, ROCm, or end-to-end model inference.
"""
import types
import unittest
from unittest.mock import patch
import torch
from test_amd import ROOT, load, attention_pkg, sparse_pkg, dense_cfg, sparse_cfg, fake_aule, reference

FULL_CHECKOUT = (ROOT / "trellis2/modules/sparse/basic.py").is_file()


@unittest.skipUnless(FULL_CHECKOUT, "Full upstream/AMD checkout required")
class ImportedAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        basic = load(sparse_pkg.__name__ + ".basic", ROOT / "trellis2/modules/sparse/basic.py")
        sparse_pkg.VarLenTensor = basic.VarLenTensor
        cls.V = basic.VarLenTensor
        cls.dense = load(attention_pkg.__name__ + ".full_attn", ROOT / "trellis2/modules/attention/full_attn.py")
        cls.sparse = load(sparse_pkg.__name__ + ".attention.full_attn", ROOT / "trellis2/modules/sparse/attention/full_attn.py")

    def setUp(self):
        torch.manual_seed(19)
        dense_cfg.set_backend("sdpa")
        sparse_cfg.set_attn_backend("sdpa")

    def test_dense_packed_qkv(self):
        packed = torch.randn(2, 5, 3, 2, 8)
        q, k, v = packed.unbind(2)
        expected = torch.stack([reference(*xs) for xs in zip(q, k, v)])
        actual = self.dense.scaled_dot_product_attention(qkv=packed)
        torch.testing.assert_close(actual, expected)

    def test_dense_cross_both_signatures(self):
        q, kv = torch.randn(2, 3, 2, 8), torch.randn(2, 7, 2, 2, 8)
        k, v = kv.unbind(2)
        expected = torch.stack([reference(*xs) for xs in zip(q, k, v)])
        for actual in (self.dense.scaled_dot_product_attention(q, kv=kv),
                       self.dense.scaled_dot_product_attention(q, k, v)):
            torch.testing.assert_close(actual, expected)

    def test_dense_aule_dispatch(self):
        packed = torch.randn(2, 5, 3, 2, 8)
        expected = self.dense.scaled_dot_product_attention(packed)
        dense_cfg.set_backend("aule")
        with patch.dict("sys.modules", {"aule": types.SimpleNamespace(flash_attention_triton=fake_aule)}):
            actual = self.dense.scaled_dot_product_attention(packed)
        torch.testing.assert_close(actual, expected)

    def test_sparse_packed_qkv(self):
        groups = [torch.randn(n, 3, 2, 8) for n in (2, 5)]
        packed = self.V.from_tensor_list(groups)
        expected = torch.cat([reference(*group.unbind(1)) for group in groups])
        actual = self.sparse.sparse_scaled_dot_product_attention(packed)
        torch.testing.assert_close(actual.feats, expected)
        self.assertEqual(actual.layout, packed.layout)

    def test_sparse_q_dense_kv(self):
        groups = [torch.randn(n, 2, 8) for n in (2, 5)]
        q = self.V.from_tensor_list(groups)
        kv = torch.randn(2, 7, 2, 2, 8)
        expected = torch.cat([reference(qi, *kvi.unbind(1)) for qi, kvi in zip(groups, kv)])
        actual = self.sparse.sparse_scaled_dot_product_attention(q, kv)
        torch.testing.assert_close(actual.feats, expected)

    def test_dense_q_sparse_kv(self):
        q = torch.randn(2, 3, 2, 8)
        groups = [torch.randn(n, 2, 2, 8) for n in (2, 5)]
        kv = self.V.from_tensor_list(groups)
        expected = torch.stack([reference(qi, *kvi.unbind(1)) for qi, kvi in zip(q, groups)])
        actual = self.sparse.sparse_scaled_dot_product_attention(q=q, kv=kv)
        torch.testing.assert_close(actual, expected)

    def test_sparse_three_arguments(self):
        qs = [torch.randn(n, 2, 8) for n in (2, 5)]
        ks = [torch.randn(n, 2, 8) for n in (3, 4)]
        vs = [torch.randn(n, 2, 6) for n in (3, 4)]
        args = [self.V.from_tensor_list(groups) for groups in (qs, ks, vs)]
        actual = self.sparse.sparse_scaled_dot_product_attention(*args)
        expected = torch.cat([reference(*xs) for xs in zip(qs, ks, vs)])
        torch.testing.assert_close(actual.feats, expected)

    def test_sparse_aule_dispatch(self):
        packed = self.V.from_tensor_list([torch.randn(n, 3, 2, 8) for n in (2, 5)])
        expected = self.sparse.sparse_scaled_dot_product_attention(packed).feats
        sparse_cfg.set_attn_backend("aule")
        with patch.dict("sys.modules", {"aule": types.SimpleNamespace(flash_attention_triton=fake_aule)}):
            actual = self.sparse.sparse_scaled_dot_product_attention(packed).feats
        torch.testing.assert_close(actual, expected)
