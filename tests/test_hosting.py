import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from selesta.config import Config


class HostingTests(unittest.TestCase):
    def test_bothost_bot_token_and_owner_id(self):
        with patch.dict(os.environ, {
            'BOT_TOKEN': 'test-token', 'LLM_API_KEY': 'test-api',
            'USER_ID': '12345',
        }, clear=True), patch('selesta.config.load_dotenv'):
            config = Config.load()
        self.assertEqual(config.telegram_token, 'test-token')
        # Bothost USER_ID must not silently grant Telegram access to someone else.
        self.assertIsNone(config.owner_id)

    def test_file_entrypoint_calls_shared_runner(self):
        path = Path(__file__).resolve().parent.parent / 'main.py'
        with patch('selesta.__main__.run') as run:
            runpy.run_path(str(path), run_name='__main__')
        run.assert_called_once_with()
