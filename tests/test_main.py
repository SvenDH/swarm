"""Combined worker/REPL lifecycle, without contacting a model provider."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from agents import WorkerClient


class MainTests(unittest.IsolatedAsyncioTestCase):
    async def test_combined_mode_runs_real_repl_and_releases_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / 'agents.db')
            argv = ['main.py', '--with-worker', '--database', database]
            with patch('sys.argv', argv), patch('builtins.input', return_value='/quit'):
                await main.main()
            # A fresh worker can claim the same database after the REPL exits.
            async with main.Worker(database):
                sessions = await WorkerClient(database).request('sessions')
                self.assertEqual(len(sessions), 1)

    async def test_combined_mode_closes_worker_on_repl_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / 'agents.db')
            with patch('sys.argv', ['main.py', '--with-worker', '--database', database]):
                with patch('main.repl', side_effect=RuntimeError('repl failed')):
                    with self.assertRaisesRegex(RuntimeError, 'repl failed'):
                        await main.main()
            async with main.Worker(database):
                self.assertEqual(await WorkerClient(database).request('sessions'), [])
