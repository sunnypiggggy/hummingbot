import tempfile
from pathlib import Path
from unittest import TestCase

import yaml

from scripts.install_binance_stocks_runtime import install


class StocksRuntimeInstallerTests(TestCase):
    def test_preserves_existing_compose_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compose = root / "docker-compose.yml"
            compose.write_text(
                "services:\n  hummingbot-api:\n    image: keep-me\n  hummingbot-mcp:\n    image: keep-mcp\n"
                "secrets:\n  dca_binance_emergency_credentials:\n    file: /keep/secret\n",
                encoding="utf-8",
            )
            (root / ".env.control").write_text("EXISTING_VALUE=unchanged\n", encoding="utf-8")
            install(root)
            first = compose.read_text(encoding="utf-8")
            install(root)
            second = compose.read_text(encoding="utf-8")
            self.assertEqual(first, second)
            self.assertIn("image: keep-me", second)
            self.assertIn("file: /keep/secret", second)
            parsed = yaml.safe_load(second)
            self.assertIn("binance-stocks-runtime", parsed["services"])
            self.assertIn("binance_stocks_credentials", parsed["secrets"])
            env = (root / ".env.control").read_text(encoding="utf-8")
            self.assertEqual(1, env.count("BINANCE_STOCKS_LIVE_AUTHORIZED=false"))
