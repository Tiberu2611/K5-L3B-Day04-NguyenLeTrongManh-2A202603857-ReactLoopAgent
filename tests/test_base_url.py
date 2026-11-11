"""Base URL cho dịch vụ trung gian (OpenAI/Anthropic-compatible)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from providers import CONFIG, Provider, endpoint_url


class BaseUrlTests(unittest.TestCase):
    def test_default_url_unchanged(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'dummy'}, clear=True):
            self.assertEqual(Provider('openai', 'm').url, CONFIG['openai'][1])

    def test_openai_compatible_base_url(self):
        env = {'OPENAI_API_KEY': 'dummy', 'OPENAI_BASE_URL': 'https://gateway.example/v1/'}
        sample = {'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': {}}
        with patch.dict('os.environ', env, clear=True), patch('providers.post_json', return_value=sample) as post:
            Provider('openai', 'm').complete([{'role': 'user', 'content': 'hi'}], [])
        self.assertEqual(post.call_args.args[0], 'https://gateway.example/v1/chat/completions')
        self.assertEqual(post.call_args.args[1]['Authorization'], 'Bearer dummy')

    def test_base_url_argument_overrides_env(self):
        env = {'OPENAI_API_KEY': 'dummy', 'OPENAI_BASE_URL': 'https://env.example/v1'}
        with patch.dict('os.environ', env, clear=True):
            self.assertEqual(Provider('openai', 'm', base_url='https://arg.example/v1').url,
                             'https://arg.example/v1/chat/completions')
            self.assertEqual(Provider('openai', 'm').url, 'https://env.example/v1/chat/completions')

    def test_eval_runner_passes_base_url_to_agent_and_judge(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evals'))
        import run_evals
        from types import SimpleNamespace
        args = SimpleNamespace(provider='openai', model='m', judge_provider='openai', judge_model='m', max_steps=2,
                               base_url='https://agent.example/v1', judge_base_url='https://judge.example/v1')
        case = {'id': 'X', 'question': 'q', 'expected': 'e'}
        with patch.object(run_evals, 'Provider') as provider, \
             patch.object(run_evals, 'run_agent', return_value={'status': 'error', 'answer': '', 'trace': [], 'usage': []}):
            run_evals.run_case(case, args)
        self.assertEqual([c.kwargs['base_url'] for c in provider.call_args_list],
                         ['https://agent.example/v1', 'https://judge.example/v1'])

    def test_transient_502_is_retried_then_succeeds(self):
        import io, urllib.error
        import providers
        ok = io.BytesIO(b'{"ok": true}')
        ok.__enter__ = lambda: ok
        errors = [urllib.error.HTTPError('https://g.example', 502, 'bad gateway', {}, io.BytesIO(b''))
                  for _ in range(2)]
        with patch('providers.urllib.request.urlopen', side_effect=errors + [ok]) as urlopen, \
             patch('providers.time.sleep') as sleep:
            self.assertEqual(providers.post_json('https://g.example', {}, {}), {'ok': True})
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        for e in errors:
            e.close()

    def test_persistent_502_raises_after_max_attempts(self):
        import io, urllib.error
        import providers
        errors = [urllib.error.HTTPError('https://g.example', 502, 'bad gateway', {}, io.BytesIO(b''))
                  for _ in range(providers.MAX_ATTEMPTS)]
        with patch('providers.urllib.request.urlopen', side_effect=errors), patch('providers.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 502'):
                providers.post_json('https://g.example', {}, {})
        for e in errors:
            e.close()

    def test_endpoint_normalisation(self):
        self.assertEqual(endpoint_url('openai', 'https://g.example/v1/chat/completions'), 'https://g.example/v1/chat/completions')
        self.assertEqual(endpoint_url('anthropic', 'https://g.example'), 'https://g.example/v1/messages')
        self.assertEqual(endpoint_url('anthropic', 'https://g.example/v1'), 'https://g.example/v1/messages')
        with self.assertRaises(ValueError):
            endpoint_url('openai', 'g.example/v1')


if __name__ == '__main__':
    unittest.main()
