"""Worker-wide budgets, restart invariants, and rejection of old schemas."""
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from agents import Runtime, Worker, WorkerClient, ExecutionError, HELP

MiB = 1024**2


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database = self.tmp.name + '/agents.db'

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_heap_budget_rollback_and_reclaim(self):
        async with Runtime(self.database, memory_budget=4*MiB) as runtime:
            agent = await runtime.agent()
            await agent.execute("one=b'a'*600000; two=b'b'*600000")
            with self.assertRaises(ExecutionError):
                await agent.execute("scratch=[b'c'*(1500000+i) for i in range(2)]")
            self.assertEqual(await agent.execute('(len(one),len(two))'), '(600000, 600000)')
            self.assertEqual(await agent.execute("one=None; two=None; 42"), '42')

    async def test_budget_is_worker_configuration_and_survives_restart(self):
        async with Runtime(self.database, memory_budget=8*MiB) as runtime:
            a,b = await runtime.agent(), await runtime.agent()
            self.assertEqual((a.memory_budget,b.memory_budget), (8*MiB,8*MiB))
            await a.execute('value=42')
        async with Runtime(self.database, memory_budget=8*MiB) as runtime:
            self.assertEqual(await (await runtime.agent(session_id=a.id)).execute('value'), '42')
        with self.assertRaisesRegex(ValueError, 'memory budget changed'):
            async with Runtime(self.database, memory_budget=16*MiB):
                pass

    async def test_worker_rejects_session_budget_overrides(self):
        async with Worker(self.database) as worker:
            client = WorkerClient(self.database)
            with self.assertRaisesRegex(RuntimeError, 'shared worker budget'):
                await client.request('agent', memory_budget=8*MiB)

    async def test_old_schema_is_rejected_without_mutation(self):
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE sessions(id INTEGER PRIMARY KEY,name TEXT UNIQUE NOT NULL,cwd TEXT NOT NULL,monty BLOB,messages TEXT NOT NULL DEFAULT '[]')")
            db.execute("INSERT INTO sessions(id,name,cwd) VALUES (0,'old','/home/old')")
        with self.assertRaisesRegex(RuntimeError, 'unsupported database schema'):
            async with Runtime(self.database):
                pass
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute('SELECT name FROM sessions').fetchone(), ('old',))
            self.assertEqual(len(db.execute('PRAGMA table_info(sessions)').fetchall()), 5)

    async def test_default_budget_and_small_usage_prompt(self):
        with patch.dict('os.environ') as environment:
            environment.pop('SWARM_MEMORY_BYTES', None)
            self.assertEqual(Runtime(self.database).memory_budget, 1024*MiB)
        for budget in (0,-1,True,1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                Runtime(self.database, memory_budget=budget)
        self.assertNotIn('suspension', HELP.lower())
        self.assertIn('too many library calls per code block', HELP)
        self.assertIn('MEMORY.md', HELP)
        self.assertLess(len(HELP), 650)
        for text in ('monty', 'virtual', 'sandbox', '1,000,000', '4,000,000'):
            self.assertNotIn(text, HELP.lower())
