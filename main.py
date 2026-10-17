"""Example: LLM_API_KEY=... uv run main.py (or configure LLM_BASE_URL)."""
import argparse
import asyncio
from agents import Runtime
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
        title = {'assistant': 'Assistant', 'python': 'Python', 'stdout': 'Output · pending commit',
                 'result': 'Result', 'error': 'Error'}[self.kind]
        if self.kind == 'python':
            body = Syntax(self.text, 'python', theme='monokai', background_color=background, word_wrap=True)
        elif self.kind == 'assistant':
            body = Markdown(self.text)
        else:
            content = self.text or ('Completed' if self.kind == 'result' else '')
            body = Text(content, style='#b8c0cc' if dark else 'red')
        return Panel(body, title=title, title_align='left', padding=(0, 1),
                     style=f'on {background}' if dark else '',
                     border_style='#30363d' if dark else 'red' if self.kind == 'error' else 'cyan')

    def __call__(self, kind, text):
        if kind != self.kind or self.live is None:
            self.finish()
            self.kind, self.text = kind, ''
            self.live = Live(console=self.console, refresh_per_second=15, vertical_overflow='visible')
            self.live.start(refresh=False)
        self.text += text
        self.live.update(self.panel())

    def finish(self):
        if self.live is not None:
            self.live.stop()
            self.live = None

    def reset(self):
        self.finish()
        self.kind, self.text = None, ''


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, help='reopen a saved session ID')
    parser.add_argument('--database', default='agent.db')
    parser.add_argument('--workspaces')
    parser.add_argument('--template', help='starter files copied into each new agent home')
    parser.add_argument('--workers', type=int, default=4, help='shared Monty workers in this host process')
    parser.add_argument('--model-concurrency', type=int, default=128)
    args = parser.parse_args()
    output = ReplOutput()
    async with Runtime(args.database, workspaces=args.workspaces, workers=args.workers,
                       model_concurrency=args.model_concurrency, template=args.template) as runtime:
        agent = await runtime.agent(session_id=args.session, on_output=output)
        print('Monty agent — /new, /sessions, /use ID, /resume, /quit')
        resume = args.session is not None
        try:
            while True:
                try:
                    if resume:
                        resume = False
                        output.reset()
                        await agent.run()
                        output.finish()
                    prompt = (await asyncio.to_thread(input, f'{agent.id}:{agent.name}> ')).strip()
                    if prompt == '/quit':
                        break
                    if prompt == '/sessions':
                        for session in await agent.sessions():
                            print(f"{session['id']}: {session['name']} — {session['workspace']}")
                        continue
                    if prompt == '/new' or prompt.startswith('/use '):
                        replacement = await runtime.agent(on_output=output,
                            session_id=None if prompt == '/new' else int(prompt.split()[1]))
                        await agent.close()
                        agent = replacement
                        resume = True
                        continue
                    if prompt:
                        output.reset()
                        answer = await agent.run(None if prompt == '/resume' else prompt)
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
            await agent.close()


if __name__ == '__main__':
    asyncio.run(main())
