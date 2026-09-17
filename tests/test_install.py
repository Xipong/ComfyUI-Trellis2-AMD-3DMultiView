import importlib.util
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("installer_test", ROOT / "tools/install_amd.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def torch(self, version="2.13.0+rocm10.0.0", hip="10.0.12345", available=True):
        return types.SimpleNamespace(__version__=version, version=types.SimpleNamespace(hip=hip),
                                     cuda=types.SimpleNamespace(is_available=lambda: available))

    def test_supported_stack(self):
        installer.validate_stack(self.torch())

    def test_cpu_or_cuda_torch_rejected(self):
        for version in ("2.13.0+cpu", "2.13.0+cu130"):
            with self.assertRaisesRegex(RuntimeError, "not a ROCm"):
                installer.validate_stack(self.torch(version=version, hip=None))

    def test_abi_mismatch_rejected(self):
        for fake in (self.torch(version="2.12.0+rocm10.0.0"), self.torch(hip="7.1.0")):
            with self.assertRaisesRegex(RuntimeError, "ABIs"):
                installer.validate_stack(fake)

    def test_missing_gpu_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "cannot see"):
            installer.validate_stack(self.torch(available=False))

    def test_exact_wheel_plan(self):
        for system, tag in (("Windows", "win_amd64"), ("Linux", "linux_x86_64")):
            paths = installer.wheel_paths(system=system)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(f"cp312-cp312-{tag}.whl" in str(p) for p in paths))
            self.assertTrue(all("rocm" in p.name for p in paths))

    def test_unsupported_platform_rejected(self):
        with self.assertRaises(RuntimeError):
            installer.wheel_paths(system="Darwin")
