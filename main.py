"""Example: LLM_API_KEY=... uv run main.py (or configure LLM_BASE_URL)."""
from agents import CodeAgent
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, help='reopen a saved session ID')
    parser.add_argument('--database', default='agent.db')
    parser.add_argument('--workspaces')
    args = parser.parse_args()
    agent = CodeAgent(args.database, session_id=args.session, workspaces=args.workspaces)
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
                    replacement = CodeAgent(args.database, workspaces=args.workspaces,
                                            session_id=None if prompt == '/new' else int(prompt.split()[1]))
                    agent.close()
                    agent = replacement
                    continue
                if prompt:
                    print(agent.run(None if prompt == '/resume' else prompt))
            except EOFError:
                break
            except Exception as exc:
                print(f'{type(exc).__name__}: {exc}')
    finally:
        agent.close()


if __name__ == '__main__':
    main()
