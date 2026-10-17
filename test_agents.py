import io
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx
from pypdf import PdfWriter

from agents import CodeAgent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agent.db'
        self.agent = CodeAgent(self.path)

    def tearDown(self):
        self.agent.close()
        self.tmp.cleanup()

    def test_restart_and_atomic_rollback(self):
        self.agent.execute("mkdir('/w'); cd('/w'); write('a', 'original'); x=42")
        with self.assertRaises(Exception):
            self.agent.execute("write('a','broken'); x=0; raise ValueError('abort')")
        self.agent.close()
        self.agent = CodeAgent(self.path)
        self.assertEqual(self.agent.execute("(x, read('a'), pwd())"), "(42, '1: original', '/w')")
        self.assertEqual(self.agent.execute("from pathlib import Path; Path('a').read_text()"), "'original'")

    def test_native_fs_and_tools(self):
        self.agent.execute("from pathlib import Path; Path('/w').mkdir(); Path('/w/a.py').write_text('a\\nb\\na'); Path('/empty').mkdir()")
        out = self.agent.execute("grep('a', '/w', limit=1)")
        self.assertIn("'truncated': True", out)
        self.assertIn("'line': 3", self.agent.execute("grep('a', '/w', offset=1)"))
        self.agent.execute("mv('/w','/new'); edit('/new/a.py', 'b', 'c')")
        self.agent.execute("mv('/new/a.py','/new/b.py'); rm('/new/b.py'); write('/new/a.py','c')")
        self.assertIn('/new/a.py', self.agent.execute("glob('*.py')"))
        self.assertEqual(self.agent.execute("Path('/empty').is_dir()"), 'True')
        self.assertEqual(self.agent.execute("Path('/etc/passwd').exists()"), 'False')
        self.agent.execute("rm('/new', recursive=True)")
        self.assertEqual(self.agent.execute("ls('/')"), "['/empty/']")

    def test_library_roundtrips(self):
        self.assertEqual(self.agent.execute("numpy('dot', [1,2], other=[3,4])"), '11')
        self.agent.execute("pandas('to_csv', [{'a': 2}], path='/a.csv')")
        self.assertEqual(self.agent.execute("pandas('read_csv', path='/a.csv')"), "[{'a': 2}]")
        self.agent.execute("pandas('to_excel', [{'a': 2}], path='/a.xlsx')")
        self.assertEqual(self.agent.execute("pandas('read_excel', path='/a.xlsx')"), "[{'a': 2}]")
        self.agent.execute("openpyxl('append','/a.xlsx',rows=[[3]])")
        self.assertIn('[3]', self.agent.execute("openpyxl('read','/a.xlsx')"))
        self.agent.execute("PIL('new',output='/a.png',size=[8,8]); PIL('resize',path='/a.png',output='/b.png',size=[2,2])")
        self.assertIn('[2, 2]', self.agent.execute("PIL('info',path='/b.png')"))
        self.agent.execute("docx('write','/a.docx',paragraphs=['hello']); docx('append','/a.docx',paragraphs=['world'])")
        self.assertEqual(self.agent.execute("docx('read','/a.docx')"), "['hello', 'world']")
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        out = io.BytesIO()
        writer.write(out)
        self.agent.db.execute('INSERT INTO files VALUES (?,?)', ('/a.pdf', out.getvalue()))
        self.agent.execute("pypdf('select','/a.pdf',output='/b.pdf',pages=[0])")
        self.assertEqual(self.agent.execute("pypdf('read','/b.pdf')"), "['']")

    def test_resume_partial_batch_without_repeating_effects(self):
        calls = [{'id': str(i), 'type': 'function', 'function': {'name': 'python',
                  'arguments': json.dumps({'code': code})}} for i, code in enumerate([
                      "write('/a', 'once')", "edit('/a', 'once', 'twice')"])]
        messages = [{'role': 'user', 'content': 'edit a'},
                    {'role': 'assistant', 'tool_calls': calls}]
        self.agent.put('messages', json.dumps(messages))
        self.agent.execute("write('/a','once')", tool_id='0', messages=messages)
        self.agent.close()
        self.agent = CodeAgent(self.path)
        def respond(request):
            history = json.loads(request.content)['messages']
            self.assertEqual([m['tool_call_id'] for m in history if m['role'] == 'tool'], ['0', '1'])
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]})
        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch('agents.httpx.Client', return_value=client):
            self.assertEqual(self.agent.run(), 'done')
        self.assertEqual(self.agent.execute("read('/a')"), "'1: twice'")

    def test_retry_pending_user_prompt(self):
        self.agent.put('messages', json.dumps([{'role': 'user', 'content': 'hi'}]))
        requests = []
        def respond(request):
            requests.append(json.loads(request.content)['messages'])
            return httpx.Response(200, json={'choices': [{'message': {
                'role': 'assistant', 'content': 'hello'}}]})
        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch('agents.httpx.Client', return_value=client):
            self.assertEqual(self.agent.run('hi'), 'hello')
        self.assertEqual(len(requests), 1)
        history = json.loads(self.agent.get('messages'))
        self.assertEqual([m['role'] for m in history], ['user', 'assistant'])

    def test_new_prompt_after_interrupted_turn(self):
        self.agent.put('messages', json.dumps([{'role': 'user', 'content': 'old'}]))
        requests = []
        def respond(request):
            history = json.loads(request.content)['messages']
            requests.append(history)
            return httpx.Response(200, json={'choices': [{'message': {
                'role': 'assistant', 'content': 'answer ' + history[-1]['content']}}]})
        clients = [httpx.Client(transport=httpx.MockTransport(respond)) for _ in range(2)]
        with patch('agents.httpx.Client', side_effect=clients):
            self.assertEqual(self.agent.run('new'), 'answer new')
        self.assertEqual(len(requests), 2)
        self.assertEqual([m['content'] for m in json.loads(self.agent.get('messages'))],
                         ['old', 'answer old', 'new', 'answer new'])


if __name__ == '__main__':
    unittest.main()
