import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("stack", Path(__file__).parents[1] / "resolve_crystal_stack.py")
stack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack)


class StackTests(unittest.TestCase):
    def test_approved_image_tracks_native_stack(self):
        image = "lmsysorg/sglang:v0.5.23-rocm10-mi35x@sha256:" + "a" * 64
        self.assertEqual(
            stack.resolve({"sglang_image": image}, {"sglang_image": image}),
            {"image": image, "version": "0.5.23"},
        )

    def test_rejects_mutable_wrong_registry_or_changed_candidate_image(self):
        image = "lmsysorg/sglang:v0.5.23-rocm10-mi35x@sha256:" + "a" * 64
        for bad in ["lmsysorg/sglang:latest", image.replace("lmsysorg", "other"), "", None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stack.resolve({"sglang_image": bad}, {"sglang_image": bad})
        with self.assertRaises(ValueError):
            stack.resolve({"sglang_image": image}, {"sglang_image": image.replace("0.5.23", "0.5.24")})


if __name__ == "__main__":
    unittest.main()
