"""Per-session budgets: real aggregate heap limits and concurrent native callbacks."""
import asyncio
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agents import Runtime, Worker, WorkerClient, ExecutionError, HELP
from wrappers.registry import available

MiB = 1024**2


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database = self.tmp.name + '/agents.db'

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_aggregate_heap_budget_rollback_and_reclaim(self):
        async with Runtime(self.database, memory_budget=4*MiB) as runtime:
            agent = await runtime.agent()
            self.assertEqual(await agent.execute("one=b'a'*600000; len(one)"), '600000')
            self.assertEqual(await agent.execute("two=b'b'*600000; len(two)"), '600000')
            with self.assertRaises(ExecutionError):
                await agent.execute("scratch=[b'c'*(1500000+i) for i in range(2)]; scratch=None")
            self.assertEqual(await agent.execute('(len(one),len(two))'), '(600000, 600000)')
            self.assertIsNone((await agent.state()).pending)
            self.assertEqual(await agent.execute("one=None; two=None; scratch=[b'c'*(1500000+i) for i in range(2)]; total=sum(len(x) for x in scratch); scratch=None; total"), '3000001')

    @unittest.skipUnless('numpy' in available(), 'NumPy not installed')
    async def test_callback_budgets_are_isolated_and_persist_after_restart(self):
        async with Runtime(self.database, workers=2) as runtime:
            small = await runtime.agent(memory_budget=2*MiB)
            large = await runtime.agent(memory_budget=16*MiB)
            code = 'import numpy as np; data=np.zeros(300000); data.nbytes'
            results = await asyncio.gather(small.execute(code), large.execute(code), return_exceptions=True)
            self.assertIsInstance(results[0], ExecutionError)
            self.assertIn('Unable to allocate', str(results[0]))
            self.assertEqual(results[1], '2400000')
            # More than a million elements is allowed when it fits in memory.
            self.assertEqual(await large.execute('data=None; a=np.zeros(1200000,dtype=np.uint8); a.nbytes'), '1200000')
            ids = small.id, large.id
        async with Runtime(self.database, memory_budget=32*MiB) as runtime:
            small = await runtime.agent(session_id=ids[0])
            large = await runtime.agent(session_id=ids[1])
            self.assertEqual((small.memory_budget,large.memory_budget), (2*MiB,16*MiB))
            self.assertEqual(await large.execute('a.nbytes'), '1200000')
            self.assertEqual((await runtime.agent()).memory_budget, 32*MiB)

    @unittest.skipUnless('PIL' in available(), 'Pillow not installed')
    async def test_images_no_longer_have_a_fixed_pixel_limit(self):
        async with Runtime(self.database, memory_budget=64*MiB) as runtime:
            agent = await runtime.agent()
            self.assertEqual(await agent.execute("from PIL import Image; image=Image.new('RGB',(2100,2000)); image.size"), '(2100, 2000)')
            with self.assertRaisesRegex(ExecutionError, '^MemoryError$'):
                await agent.execute("Image.new('RGBA',(10000,10000))")

    async def test_worker_accepts_a_new_session_budget(self):
        async with Worker(self.database) as worker:
            client = WorkerClient(self.database)
            info = await client.request('agent', memory_budget=8*MiB)
            agent = await worker.runtime.agent(session_id=info['id'])
            self.assertEqual(agent.memory_budget, 8*MiB)
            with self.assertRaisesRegex(RuntimeError, 'new sessions'):
                await client.request('agent', session_id=info['id'], memory_budget=16*MiB)

    async def test_existing_database_is_upgraded_without_losing_sessions(self):
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE sessions(id INTEGER PRIMARY KEY,name TEXT UNIQUE NOT NULL,cwd TEXT NOT NULL,monty BLOB,messages TEXT NOT NULL DEFAULT '[]')")
            db.execute("INSERT INTO sessions(id,name,cwd) VALUES (0,'old','/home/old')")
        async with Runtime(self.database, memory_budget=8*MiB) as runtime:
            agent = await runtime.agent(session_id=0)
            self.assertEqual(agent.memory_budget, 8*MiB)
            self.assertEqual(agent.name, 'old')

    async def test_invalid_budgets_and_suspension_prompt(self):
        with patch.dict('os.environ') as environment:
            environment.pop('SWARM_MEMORY_BYTES', None)
            self.assertEqual(Runtime(self.database).memory_budget, 1024*MiB)
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                Runtime(self.database, memory_budget=budget)
        self.assertNotIn('suspension', HELP.lower())
        self.assertIn('too many library calls per code block', HELP)
        self.assertIn('vectorized', HELP)
        self.assertLess(len(HELP), 650)
