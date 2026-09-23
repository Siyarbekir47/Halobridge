"""Doctor tests: python -m unittest discover -s tests -v"""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import doctor
from settings import Config, DashboardConfig, ModelsConfig, SecurityConfig, UpdatesConfig


class DoctorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HALOGEN_TEST_TMP"))
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    async def test_no_models_is_a_failure(self):
        config = Config(
            models=ModelsConfig(auto_discover=False, explicit={}),
            dashboard=DashboardConfig(state_dir=self.root / "state"),
            updates=UpdatesConfig(enabled=False, backup_dir=self.root / "backup"),
        )
        out = io.StringIO()
        with redirect_stdout(out):
            code = await doctor.run(config)
        self.assertEqual(code, 1)
        self.assertIn("No models found", out.getvalue())

    async def test_explicit_model_and_token_passes(self):
        config = Config(
            models=ModelsConfig(
                auto_discover=False,
                explicit={"model-a": "svc-a.service"},
            ),
            dashboard=DashboardConfig(state_dir=self.root / "state"),
            updates=UpdatesConfig(enabled=False, backup_dir=self.root / "backup"),
            security=SecurityConfig(auth_token="secret"),
        )
        out = io.StringIO()
        with redirect_stdout(out):
            code = await doctor.run(config)
        self.assertEqual(code, 0)
        self.assertIn("Token gate active", out.getvalue())
        self.assertIn("model-a -> svc-a.service", out.getvalue())


if __name__ == "__main__":
    unittest.main()
