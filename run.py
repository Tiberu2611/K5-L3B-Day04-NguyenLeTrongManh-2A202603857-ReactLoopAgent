import argparse
import json
import os
import sys
from agent import run_agent
from providers import Provider
from status_tools import StatusTools
from bonus.tools import BonusTools


def run(question, mode, provider_name, model, max_steps, fail=False, bonus=False, base_url=None):
    provider = Provider(provider_name, model, base_url=base_url)
    tools = BonusTools(mode) if bonus else StatusTools(mode, fail=fail)
    return run_agent(question, provider, tools, max_steps=max_steps)


def main():
    p = argparse.ArgumentParser(description='Chạy status agent')
    p.add_argument('question', help='Câu hỏi cho agent')
    p.add_argument('--bonus', action='store_true', help='Dùng bộ tools bonus')
    p.add_argument('--mode', choices=['snapshot', 'live'], default='snapshot')
    p.add_argument('--provider', choices=['openai', 'gemini', 'anthropic', 'openrouter'], default=os.getenv('PROVIDER', 'openai'))
    p.add_argument('--model', default=os.getenv('MODEL'))
    p.add_argument('--base-url', default=None, help='Base URL của dịch vụ trung gian, ví dụ https://gateway.example/v1')
    p.add_argument('--max-steps', type=int, default=6)
    args = p.parse_args()
    if args.max_steps < 1:
        p.error('--max-steps phải >= 1')
    try:
        result = run(args.question, args.mode, args.provider, args.model, args.max_steps, bonus=args.bonus, base_url=args.base_url)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f'Lỗi: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'completed' else 1

if __name__ == '__main__':
    sys.exit(main())
