import asyncio
import os
import pty
import tempfile
import termios
import unittest
from unittest.mock import patch

import httpx

import main
from agents import Worker, WorkerClient
from test_agents import model_client, stream_response


class StopTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_is_durable_and_explicit_resume_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            path = folder + '/agent.db'
            entered = asyncio.Event()
            cancelled = asyncio.Event()
            async def respond(request):
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
            client = model_client(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
            async with Worker(path, client=client) as worker:
                remote = WorkerClient(path)
                info = await remote.request('agent')
                run = asyncio.create_task(remote.request('run', session_id=info['id'], prompt='work'))
                await asyncio.wait_for(entered.wait(), 5)
                self.assertTrue(await remote.request('stop', session_id=info['id']))
                self.assertIsNone(await asyncio.wait_for(run, 5))
                self.assertTrue(cancelled.is_set())
                self.assertNotIn(info['id'], worker._running)
            requests = []
            def resumed(request):
                requests.append(request)
                return stream_response({'role': 'assistant', 'content': 'finished'})
            client = model_client(httpx.AsyncClient(transport=httpx.MockTransport(resumed)))
            async with Worker(path, client=client) as worker:
                await asyncio.sleep(0.05)
                self.assertEqual(requests, [])
                self.assertEqual(worker._running, {})
                remote = WorkerClient(path)
                self.assertTrue((await remote.request('agent', session_id=info['id']))['paused'])
                with patch('sys.argv', ['main.py', '--database', path, '--session', str(info['id'])]), \
                        patch('builtins.input', return_value='/quit'):
                    await main.main()
                self.assertEqual(requests, [])
                self.assertEqual(await remote.request('run', session_id=info['id']), 'finished')
                self.assertEqual(len(requests), 1)

    async def test_escape_helper_sends_stop_instead_of_detaching(self):
        entered, escape, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class Client:
            async def request(self, action, **fields):
                if action == 'run':
                    entered.set()
                    await stopped.wait()
                    return None
                self_test.assertEqual(action, 'stop')
                self_test.assertEqual(fields['session_id'], 7)
                stopped.set()
                return True
        self_test = self
        with patch.object(main, 'wait_for_escape', escape.wait):
            run = asyncio.create_task(main.run_with_escape(Client(), 7, 'work'))
            await entered.wait()
            escape.set()
            self.assertIsNone(await asyncio.wait_for(run, 5))
            self.assertTrue(stopped.is_set())

    async def test_real_terminal_escape_arrow_keys_and_restore(self):
        master, slave = pty.openpty()
        try:
            with os.fdopen(os.dup(slave), 'r') as terminal, patch.object(main.sys, 'stdin', terminal):
                original = termios.tcgetattr(slave)
                waiting = asyncio.create_task(main.wait_for_escape())
                await asyncio.sleep(0)
                os.write(master, b'\x1b[A')
                await asyncio.sleep(0.08)
                self.assertFalse(waiting.done())
                os.write(master, b'\x1b')
                await asyncio.wait_for(waiting, 1)
                self.assertEqual(termios.tcgetattr(slave), original)
                waiting = asyncio.create_task(main.wait_for_escape())
                await asyncio.sleep(0)
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                self.assertEqual(termios.tcgetattr(slave), original)
        finally:
            os.close(master)
            os.close(slave)
