"""Example: LLM_API_KEY=... uv run main.py (or configure LLM_BASE_URL)."""
import argparse
from agents import CodeAgent
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, help='reopen a saved session ID')
    parser.add_argument('--database', default='agent.db')
    parser.add_argument('--workspaces')
    args = parser.parse_args()
    output = ReplOutput()
    options = {'workspaces': args.workspaces, 'on_output': output}
    agent = CodeAgent(args.database, session_id=args.session, **options)
    print('Monty agent — /new, /sessions, /use ID, /resume, /quit')
    try:
        while True:
            try:
                prompt = input(f'{agent.id}:{agent.name}> ').strip()
                if prompt == '/quit':
                    break
                if prompt == '/sessions':
                    for session in agent.sessions():
                        print(f"{session['id']}: {session['name']} — {session['workspace']}")
                    continue
                if prompt == '/new' or prompt.startswith('/use '):
                    replacement = CodeAgent(args.database, **options,
                                            session_id=None if prompt == '/new' else int(prompt.split()[1]))
                    agent.close()
                    agent = replacement
                    continue
                if prompt:
                    output.reset()
                    answer = agent.run(None if prompt == '/resume' else prompt)
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
        agent.close()


if __name__ == '__main__':
    main()
