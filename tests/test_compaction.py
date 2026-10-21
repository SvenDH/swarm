"""Automatic compaction through real SDK streams with a mocked provider."""
import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agents import CodeAgent, ContextLimit, Worker, WorkerClient, message_tokens, tokens, summary_source, tokenizer
from test_agents import model_client, stream_response, sse


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agent.db'
        self.agent = await CodeAgent.open(self.path, context_tokens=4096)
        self.session_id = self.agent.id
        self.requests = []

    async def asyncTearDown(self):
        await self.agent.close()
        self.tmp.cleanup()

    async def seed(self):
        await self.agent.execute("counter=1; open('saved.txt','w').write('saved')")
        await self.agent.append_message({'role': 'user', 'content': 'old task ' + 'x ' * 6000})
        await self.agent.append_message({'role': 'assistant', 'content': 'Old task complete.'})

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        return stream_response({'role': 'assistant', 'content':
            '## Task\nOld task complete.\n## Constraints\nNone.\n## State\ncounter=1; saved.txt contains saved.\n## Progress\nSaved.\n## Next\nFollow latest request.\n## Risks\nNone.' if 'tools' not in body else 'continued'})

    async def test_automatic_compaction_is_durable_and_keeps_interpreter_and_files(self):
        await self.seed()
        observed = []
        self.agent.on_output = lambda kind, text: observed.append((kind, text))
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run('next task'), 'continued')
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn('tools', self.requests[0])
        self.assertIn('max_completion_tokens', self.requests[0])
        self.assertEqual(self.requests[1]['messages'][-1]['content'], 'next task')
        self.assertTrue(any(kind == 'compaction' for kind, _ in observed))
        history = (await self.agent.state()).messages
        self.assertEqual(len(history), 3)
        self.assertIn('Session continuation memory', history[0]['content'])
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_tokens=4096)
        self.assertEqual((await self.agent.state()).messages, history)
        self.assertEqual(await self.agent.execute("(counter,open('saved.txt').read())"), "(1, 'saved')")

    async def test_pending_tools_finish_before_compacting_without_splitting_batch(self):
        await self.seed()
        await self.agent.append_message({'role': 'user', 'content': 'increment twice'})
        calls = [{'id': str(i), 'type': 'function', 'function': {
            'name': 'python', 'arguments': json.dumps({'code': 'counter += 1'})}} for i in range(2)]
        await self.agent.append_message({'role': 'assistant', 'tool_calls': calls})
        await self.agent.execute('counter += 1', tool_id='0')
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run(), 'continued')
        sent = self.requests[-1]['messages']
        tool_calls = [c['id'] for m in sent for c in m.get('tool_calls', [])]
        results = [m['tool_call_id'] for m in sent if m['role'] == 'tool']
        self.assertEqual(tool_calls, ['0', '1'])
        self.assertEqual(results, tool_calls)
        self.assertEqual(await self.agent.execute('counter'), '3')

    async def test_deepseek_disables_thinking_only_for_summaries(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            for model in ('deepseek-flash', 'deepseek-reasoner', 'other-model'):
                with patch.object(self.agent, 'model', model):
                    await self.agent._response(model_client(client), [], summary_tokens=512)
                    await self.agent._response(model_client(client), [])
                summary, normal = self.requests[-2:]
                if model.startswith('deepseek-'):
                    self.assertEqual(summary['thinking'], {'type': 'disabled'})
                else:
                    self.assertNotIn('thinking', summary)
                self.assertNotIn('thinking', normal)

    async def test_failed_or_cancelled_summary_keeps_original_history_and_retries(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                await self.seed()
                await self.agent.append_message({'role': 'user', 'content': 'continue'})
                before = (await self.agent.state()).messages
                async def fail(request):
                    if cancel:
                        raise asyncio.CancelledError()
                    return httpx.Response(503, json={'error': {'message': 'unavailable'}})
                async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                    with patch.object(self.agent.runtime, 'client', model_client(client)):
                        with self.assertRaises(asyncio.CancelledError if cancel else Exception):
                            await self.agent.run()
                self.assertEqual((await self.agent.state()).messages, before)
                await self.agent.close()
                self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_tokens=4096)
                async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
                    with patch.object(self.agent.runtime, 'client', model_client(client)):
                        self.assertEqual(await self.agent.run(), 'continued')

    async def test_small_history_and_disabled_compaction_make_no_summary_requests(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                await self.agent.run('hi')
                self.agent.runtime.context_tokens = None
                await self.agent.run('x ' * 6000)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all('tools' in body for body in self.requests))

    async def test_restart_after_summary_commit_does_not_summarize_again(self):
        await self.seed()
        async def respond(request):
            if 'tools' in json.loads(request.content):
                return httpx.Response(503, json={'error': {'message': 'interrupted after commit'}})
            return self.respond(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                with self.assertRaises(Exception):
                    await self.agent.run('continue')
        saved = (await self.agent.state()).messages
        self.assertEqual(len(saved), 2)
        self.assertIn('Session continuation memory', saved[0]['content'])
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_tokens=4096)
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run(), 'continued')
        self.assertEqual(len(self.requests), 2)
        self.assertIn('tools', self.requests[-1])

    async def test_repeated_compaction_folds_old_memory_and_is_session_local(self):
        await self.seed()
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                await self.agent.run('first')
                await self.seed()
                await self.agent.run('third')
        summaries = [body for body in self.requests if 'tools' not in body]
        self.assertEqual(len(summaries), 2)
        self.assertIn('Session continuation memory', summaries[-1]['messages'][-1]['content'])
        self.assertEqual((await self.agent.state()).messages[-2]['content'], 'third')
        other = await self.agent.runtime.agent()
        try:
            self.assertEqual((await other.state()).messages, [])
        finally:
            await other.close()

    async def test_invalid_summaries_do_not_replace_history(self):
        await self.seed()
        await self.agent.append_message({'role': 'user', 'content': 'continue'})
        before = (await self.agent.state()).messages
        for message in ({'role': 'assistant', 'content': ''},
                        {'role': 'assistant', 'refusal': 'No summary'},
                        {'role': 'assistant', 'tool_calls': [{'id': 'blocked', 'type': 'function',
                          'function': {'name': 'python', 'arguments': '{"code":"counter=0"}'}}]}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                    stream_response(message))) as client:
                with patch.object(self.agent.runtime, 'client', model_client(client)):
                    with self.assertRaisesRegex(RuntimeError, 'summary'):
                        await self.agent.run()
            self.assertEqual((await self.agent.state()).messages, before)
        with self.assertRaisesRegex(RuntimeError, 'did not reduce'):
            await self.agent.runtime.database_call(self.agent._save_compaction, before, before)

    async def test_finished_summaries_allow_different_headings_and_token_estimates(self):
        for checkpoint in ('### Task\nDone.\n### Next\nContinue with saved.txt.',
                           '**Progress:** Saved saved.txt; counter=1. Follow the latest request.',
                           'Verified state: ' + 'detail ' * 500 + 'verified.'):
            with self.subTest(checkpoint=checkpoint[:30]):
                await self.seed()
                def respond(request):
                    body = json.loads(request.content)
                    return stream_response({'role': 'assistant', 'content': checkpoint if 'tools' not in body else 'continued'})
                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    with patch.object(self.agent.runtime, 'client', model_client(client)):
                        self.assertEqual(await self.agent.run('next request'), 'continued')
                history = (await self.agent.state()).messages
                self.assertTrue(any(checkpoint in message.get('content', '') for message in history))
                self.assertEqual(history[-2]['content'], 'next request')
                self.assertEqual(await self.agent.execute('counter'), '1')

    async def test_empty_or_truncated_summary_recovers_with_one_retry(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                await self.seed()
                requests = []
                def respond(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    if len(requests) == 1:
                        if truncated:
                            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                content=sse({'role': 'assistant', 'content': 'partial'}, 'length') + b'data: [DONE]\n\n')
                        return stream_response({'role': 'assistant', 'content': ''})
                    return self.respond(request)
                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    with patch.object(self.agent.runtime, 'client', model_client(client)):
                        self.assertEqual(await self.agent.run('continue'), 'continued')
                self.assertEqual(len(requests), 3)
                self.assertGreater(requests[1]['max_completion_tokens'], requests[0]['max_completion_tokens'])
                self.assertIn('tools', requests[2])

    async def test_finished_summary_that_cannot_fit_preserves_history(self):
        await self.seed()
        await self.agent.append_message({'role': 'user', 'content': 'continue'})
        before = (await self.agent.state()).messages
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                stream_response({'role': 'assistant', 'content': '長' * 10000}))) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                with self.assertRaises(ContextLimit):
                    await self.agent.run()
        self.assertEqual((await self.agent.state()).messages, before)
        self.assertEqual((await self.agent.runtime.query('SELECT count(*) AS n FROM compactions'))[0]['n'], 0)

    async def test_pruning_avoids_summary_call_and_archived_output_is_searchable(self):
        await self.agent.append_message({'role': 'user', 'content': 'Inspect the log.'})
        await self.agent.append_message({'role': 'assistant', 'tool_calls': [{'id': 'read-log', 'type': 'function',
            'function': {'name': 'python', 'arguments': '{"code":"print(log)"}'}}]})
        output = 'ARCHIVE_NEEDLE_123 ' + 'detail ' * 6000
        await self.agent.append_message({'role': 'tool', 'tool_call_id': 'read-log', 'content': output})
        await self.agent.append_message({'role': 'assistant', 'content': 'Inspection complete.'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run('next task'), 'continued')
        self.assertEqual(len(self.requests), 1)
        self.assertIn('tools', self.requests[0])
        recovered = await self.agent.execute("session_search('ARCHIVE_NEEDLE_123')[0]['text']")
        self.assertIn('ARCHIVE_NEEDLE_123', recovered)
        self.assertEqual(await self.agent.execute("session_search(\"' OR 1=1 --\")"), '[]')
        other = await self.agent.runtime.agent()
        try:
            self.assertEqual(await other.execute("session_search('ARCHIVE_NEEDLE_123')"), '[]')
        finally:
            await other.close()
        archive = await self.agent.runtime.database_call(lambda: self.agent.db.execute(
            'SELECT messages FROM compactions WHERE session_id=?', (self.agent.id,)).fetchone()[0])
        self.assertEqual(json.loads(archive)[2]['content'], output)

    async def test_latest_user_and_short_original_constraints_survive_verbatim(self):
        first = 'Never delete protected.txt.'
        task = 'Create report.csv without changing source.csv.'
        for message in ({'role': 'user', 'content': first}, {'role': 'assistant', 'content': 'Understood.'},
                        {'role': 'user', 'content': task}, {'role': 'assistant', 'content': 'progress ' * 6000},
                        {'role': 'assistant', 'content': 'Checkpoint reached.'}):
            await self.agent.append_message(message)
        original = (await self.agent.state()).messages
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                compacted = await self.agent._compact(original)
        self.assertEqual(compacted[0]['content'], first)
        self.assertIn(task, [message.get('content') for message in compacted])
        self.assertEqual(compacted[-1], original[-1])

    async def test_usage_anchor_survives_restart_and_invalidates_on_model_or_notes_change(self):
        await self.agent.append_message({'role': 'user', 'content': 'hello'})
        original = (await self.agent.state()).messages
        basis = self.agent._context_basis(await self.agent._memory())
        await self.agent.runtime.database_call(self.agent._save_usage, original, basis, 2000)
        next_message = {'role': 'assistant', 'content': 'hello back'}
        await self.agent.append_message(next_message)
        expected = 2000 + message_tokens(next_message, self.agent.model)
        self.assertEqual(await self.agent._context_cost((await self.agent.state()).messages), expected)
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_tokens=4096)
        messages = (await self.agent.state()).messages
        self.assertEqual(await self.agent._context_cost(messages), expected)
        model = self.agent.model
        self.agent.model = 'different-model'
        self.assertNotEqual(await self.agent._context_cost(messages), expected)
        self.agent.model = model
        await self.agent.execute("open('MEMORY.md','w').write('New persistent note.')")
        self.assertNotEqual(await self.agent._context_cost(messages), expected)

    async def test_provider_usage_is_requested_and_saved_from_real_sdk_stream(self):
        def respond(request):
            body = json.loads(request.content)
            self.assertTrue(body['stream_options']['include_usage'])
            self.assertEqual(body['max_completion_tokens'], 1024)
            usage = {'id': 'chat-test', 'object': 'chat.completion.chunk', 'created': 0,
                     'model': 'test', 'choices': [],
                     'usage': {'prompt_tokens': 2000, 'completion_tokens': 5, 'total_tokens': 2005}}
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=
                sse({'role': 'assistant', 'content': 'done'}, 'stop') +
                ('data: ' + json.dumps(usage) + '\n\n' + 'data: [DONE]\n\n').encode())
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run('hello'), 'done')
                history = (await self.agent.state()).messages
                self.assertEqual(await self.agent._context_cost(history), 2000 + message_tokens(history[-1], self.agent.model))

    async def test_provider_overflow_gets_one_compaction_retry(self):
        for i in range(15):
            await self.agent.append_message({'role': 'user', 'content': f'Previous task {i}.'})
            await self.agent.append_message({'role': 'assistant', 'content': f'Completed task {i} with verified result.'})
        main_requests = 0
        def respond(request):
            nonlocal main_requests
            if 'tools' in json.loads(request.content):
                main_requests += 1
                if main_requests == 1:
                    return httpx.Response(400, json={'error': {'message': 'maximum context length exceeded',
                                                             'code': 'context_length_exceeded'}})
            return self.respond(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run('continue'), 'continued')
        self.assertEqual(main_requests, 2)
        self.assertEqual(sum('tools' not in body for body in self.requests), 1)

    async def test_summary_input_and_image_estimates_are_bounded(self):
        text = summary_source([{'role': 'user', 'content': '長' * 10000},
                               {'role': 'tool', 'content': 'TOKEN_SENTINEL'}], self.agent.model, 256)
        self.assertLessEqual(tokens(text, self.agent.model), 256)
        self.assertIn('omitted', text)
        self.assertIn('TOKEN_SENTINEL', text)
        small = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:small'}}]}
        big = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:' + 'x'*100000}}]}
        self.assertEqual(message_tokens(small, self.agent.model), message_tokens(big, self.agent.model))

    async def test_failed_compaction_commit_does_not_archive_or_replace_history(self):
        await self.seed()
        await self.agent.append_message({'role': 'user', 'content': 'continue'})
        original = (await self.agent.state()).messages
        def fail():
            self.agent.db.execute('''CREATE TRIGGER reject_compaction BEFORE UPDATE OF messages ON sessions
                BEGIN SELECT RAISE(ABORT, 'checkpoint failed'); END''')
        await self.agent.runtime.database_call(fail)
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                with self.assertRaises(Exception):
                    await self.agent._compact(original)
        self.assertEqual((await self.agent.state()).messages, original)
        count = await self.agent.runtime.database_call(lambda: self.agent.db.execute('SELECT count(*) FROM compactions').fetchone()[0])
        self.assertEqual(count, 0)

    async def test_offline_tokenizer_uses_conservative_fallback(self):
        tokenizer.cache_clear()
        try:
            with patch('agents.tiktoken.encoding_for_model', side_effect=OSError('offline')):
                self.assertEqual(tokens('héllo', 'offline-model'), len('héllo'.encode()))
                self.assertLessEqual(len(summary_source([{'content': 'large '*100}], 'offline-model', 100).encode()), 100)
        finally:
            tokenizer.cache_clear()

    async def test_oversized_latest_request_is_not_silently_summarized(self):
        text = 'keep this entire request ' * 5000
        with patch.object(self.agent, '_response', side_effect=AssertionError('oversized request sent')):
            with self.assertRaises(ContextLimit):
                await self.agent.run(text)
        self.assertEqual((await self.agent.state()).messages[-1]['content'], text)

    async def test_overflow_retry_is_bounded_and_non_context_errors_do_not_compact(self):
        from openai import BadRequestError
        for i in range(15):
            await self.agent.append_message({'role': 'user', 'content': f'Task {i}: preserve the exact identifier ID_{i}.'})
            await self.agent.append_message({'role': 'assistant', 'content': f'Task {i} completed and verified.'})
        requests = []
        def respond(request):
            body = json.loads(request.content)
            if 'tools' not in body:
                return self.respond(request)
            requests.append(body)
            return httpx.Response(400, json={'error': {'message': 'maximum context length exceeded',
                                                      'code': 'context_length_exceeded'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                with self.assertRaises(BadRequestError):
                    await self.agent.run('continue')
        self.assertEqual(len(requests), 2)
        before = (await self.agent.state()).messages
        requests.clear()
        def rejected(request):
            requests.append(json.loads(request.content))
            return httpx.Response(400, json={'error': {'message': 'invalid reasoning field', 'code': 'invalid_request_error'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(rejected)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                with self.assertRaises(BadRequestError):
                    await self.agent.run()
        self.assertEqual(len(requests), 1)
        self.assertEqual((await self.agent.state()).messages, before)

    async def test_search_prefers_original_output_to_later_pruned_copies(self):
        original = [{'role': 'tool', 'tool_call_id': 'needle-id', 'content': 'UNIQUE_NEEDLE original details'}]
        stub = [{'role': 'tool', 'tool_call_id': 'needle-id', 'content': 'UNIQUE_NEEDLE [Earlier output omitted; recover with session_search.]'}]
        def seed():
            self.agent.db.executemany('INSERT INTO compactions(session_id,messages) VALUES (?,?)',
                [(self.agent.id, json.dumps(original)), (self.agent.id, json.dumps(stub))])
        await self.agent.runtime.database_call(seed)
        self.assertIn('original details', await self.agent.execute("session_search('UNIQUE_NEEDLE',limit=1)[0]['text']"))

    async def test_truncated_summary_stops_worker_retry_and_keeps_history(self):
        await self.seed()
        before = (await self.agent.state()).messages
        await self.agent.close()
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content=sse({'role': 'assistant', 'content': '## Task\npartial'}, 'length') + b'data: [DONE]\n\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            async with Worker(self.path, context_tokens=4096, retry_delay=.01, client=model_client(client)) as worker:
                remote = WorkerClient(self.path)
                with self.assertRaisesRegex(RuntimeError, 'Incomplete model response') as failure:
                    await remote.request('run', session_id=self.session_id, prompt='continue')
                self.assertIn('history unchanged', str(failure.exception))
                self.assertNotIn('fix configuration/history', str(failure.exception))
                await asyncio.sleep(.05)
                self.assertEqual(len(requests), 2)
                agent = await worker.runtime.agent(session_id=self.session_id)
                self.assertEqual((await agent.state()).messages, [*before, {'role': 'user', 'content': 'continue'}])
                await agent.close()

