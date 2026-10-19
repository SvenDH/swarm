"""Connect a REPL to a worker, or host both with --with-worker."""
import argparse
import asyncio
import os
import sys
import termios
import tty
from agents import Worker, WorkerClient
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


class ReplOutput:
    """One live Rich panel per streamed block; completed blocks stay in scrollback."""

    def __init__(self, console=None):
        self.console = console or Console()
        self.kind, self.text, self.live = None, '', None

    def panel(self):
        dark = self.kind in ('python', 'stdout', 'result')
        background = '#11161d'
        title = {'assistant': 'Assistant', 'reasoning': 'Reasoning', 'python': 'Python', 'stdout': 'Output · pending commit',
                 'result': 'Result', 'error': 'Error', 'compaction': 'Context memory', 'status': 'Status'}[self.kind]
        if self.kind == 'python':
            body = Syntax(self.text, 'python', theme='monokai', background_color=background, word_wrap=True)
        elif self.kind in ('assistant', 'compaction', 'reasoning'):
            body = Markdown(self.text, style='dim' if self.kind == 'reasoning' else 'none')
        else:
            content = self.text or ('Completed' if self.kind == 'result' else '')
            body = Text(content, style='#b8c0cc' if dark else 'red' if self.kind == 'error' else 'cyan')
        return Panel(body, title=title, title_align='left', padding=(0, 1),
                     style=f'on {background}' if dark else '',
                     border_style=('#30363d' if dark else 'red' if self.kind == 'error'
                                   else 'magenta' if self.kind == 'reasoning' else 'cyan'))

    def __call__(self, kind, text):
        if kind != self.kind or self.live is None:
            self.finish()
            self.kind, self.text = kind, ''
            self.live = Live(get_renderable=self.panel, console=self.console,
                             refresh_per_second=15, vertical_overflow='visible')
            self.live.start(refresh=False)
        self.text += text

    def finish(self):
        if self.live is not None:
            self.live.stop()
            self.live = None

    def reset(self):
        self.finish()
        self.kind, self.text = None, ''


async def wait_for_escape():
    """Watch a real terminal only while running; always restore its settings."""
    if not sys.stdin.isatty():
        await asyncio.Future()
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    pressed = asyncio.Event()
    timer = None
    def read():
        nonlocal timer
        if timer is not None:
            timer.cancel()
        data = os.read(fd, 1024)
        # Arrow/function keys start with ESC too. Only a standalone ESC stops.
        if data == b'\x1b':
            timer = loop.call_later(0.05, pressed.set)
    try:
        tty.setcbreak(fd)
        loop.add_reader(fd, read)
        await pressed.wait()
    finally:
        if timer is not None:
            timer.cancel()
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


async def run_with_escape(client, session_id, prompt):
    run = asyncio.create_task(client.request('run', session_id=session_id, prompt=prompt))
    escape = asyncio.create_task(wait_for_escape())
    try:
        done, _ = await asyncio.wait((run, escape), return_when=asyncio.FIRST_COMPLETED)
        if escape in done and not run.done():
            await escape
            await client.request('stop', session_id=session_id)
        return await run
    finally:
        for task in (run, escape):
            if not task.done():
                task.cancel()
        await asyncio.gather(run, escape, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--worker', action='store_true', help='run the session worker instead of the REPL')
    mode.add_argument('--with-worker', action='store_true', help='run the worker and REPL in this process')
    parser.add_argument('--socket', help='local worker socket (defaults to DATABASE.sock)')
    parser.add_argument('--session', type=int, help='reopen a saved session ID')
    parser.add_argument('--database', default='agent.db')
    parser.add_argument('--workspaces')
    parser.add_argument('--template', help='starter files copied into each new agent home')
    parser.add_argument('--workers', type=int, default=4, help='shared Monty workers in this host process')
    parser.add_argument('--model-concurrency', type=int, default=128)
    parser.add_argument('--context-chars', type=int, default=120_000, help='conversation compaction threshold')
    parser.add_argument('--memory-mb', type=int, help='memory budget per new session in MiB (default: 50)')
    args = parser.parse_args()
    if args.worker or args.with_worker:
        async with Worker(args.database, socket=args.socket, workspaces=args.workspaces,
                          workers=args.workers, model_concurrency=args.model_concurrency,
                          template=args.template, context_chars=args.context_chars,
                          memory_budget=None if args.memory_mb is None else args.memory_mb * 1024**2) as worker:
            print(f'Monty worker listening on {worker.socket}', flush=True)
            if args.with_worker:
                await repl(args)
            else:
                await worker.serve()
        return
    await repl(args)


async def repl(args):
    output = ReplOutput()
    client = WorkerClient(args.database, socket=args.socket, on_output=output)
    try:
        agent = await client.request('agent', session_id=args.session)
        print('Monty agent — /new, /sessions, /use ID, /resume, /quit')
        resume = args.session is not None and not agent.get('paused', False)
        try:
            while True:
                try:
                    if resume:
                        resume = False
                        prompt = '/resume'
                    else:
                        prompt = (await asyncio.to_thread(input, f"{agent['id']}:{agent['name']}> ")).strip()
                    if prompt == '/quit':
                        break
                    if prompt == '/sessions':
                        for session in await client.request('sessions'):
                            print(f"{session['id']}: {session['name']}")
                        continue
                    if prompt == '/new' or prompt.startswith('/use '):
                        agent = await client.request('agent',
                            session_id=None if prompt == '/new' else int(prompt.split()[1]))
                        resume = not agent.get('paused', False)
                        continue
                    if prompt:
                        output.reset()
                        answer = await run_with_escape(client, agent['id'],
                                                       None if prompt == '/resume' else prompt)
                        if answer is None:
                            output.finish()
                            output('status', 'Stopped. Use /resume to continue.')
                            output.finish()
                            continue
                        if output.kind is None:
                            output('assistant', answer)
                        output.finish()
                except EOFError:
                    break
                except Exception as exc:
                    output('error', f'{type(exc).__name__}: {exc}')
                    output.finish()
        finally:
            output.finish()
    except (FileNotFoundError, ConnectionRefusedError):
        output.console.print(f'Worker is not running. Start: uv run main.py --worker --database {args.database}')


if __name__ == '__main__':
    asyncio.run(main())
