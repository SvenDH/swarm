"""Example: LLM_API_KEY=... uv run main.py (or configure LLM_BASE_URL)."""
from agents import CodeAgent


def main():
    agent = CodeAgent()
    print('Monty agent — /resume continues an interrupted turn; /quit exits.')
    try:
        while True:
            try:
                prompt = input('> ').strip()
                if prompt == '/quit':
                    break
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
