"""Run the dependency-free browser payload/streaming tests with Node.js."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js is required for playground JavaScript tests")
class PlaygroundTests(unittest.TestCase):
    def test_request_builders_streaming_audio_and_endpoint_coverage(self):
        result = subprocess.run([shutil.which("node"), "--test", str(Path(__file__).with_suffix(".js"))],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
