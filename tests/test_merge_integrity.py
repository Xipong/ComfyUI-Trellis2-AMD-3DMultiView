"""Verify that integration did not discard newer MultiView code or alter binaries."""
import hashlib
import json
import subprocess
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "UPSTREAMS.json").read_text(encoding="utf-8"))


class MergeIntegrityTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "nodes.py").exists(), "Full upstream checkout required")
    def test_preserved_wrapper_blobs(self):
        for path, expected in MANIFEST["preserved_wrapper_blobs"].items():
            data = subprocess.check_output(["git", "cat-file", "blob", f"HEAD:{path}"], cwd=ROOT)
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            self.assertEqual(actual, expected, path)

    @unittest.skipUnless((ROOT / "trellis2/modules/attention/full_attn.py").exists(), "Full AMD source checkout required")
    def test_imported_amd_attention_blobs(self):
        for path, expected in MANIFEST["imported_amd_blobs"].items():
            data = subprocess.check_output(["git", "cat-file", "blob", f"HEAD:{path}"], cwd=ROOT)
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            self.assertEqual(actual, expected, path)

    @unittest.skipUnless((ROOT / "nodes.py").exists(), "Full upstream checkout required")
    def test_no_missing_profiler_import(self):
        self.assertNotIn("trellis2_profiler", (ROOT / "nodes.py").read_text(encoding="utf-8"))
        self.assertTrue((ROOT / "trellis2/utils/mv_camera.py").is_file())
