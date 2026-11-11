"""Bốn provider cùng một interface. Không chứa agent loop."""
import json
import os
import time
import urllib.error
import urllib.request

CONFIG = {
    'openai': ('OPENAI_API_KEY', 'https://api.openai.com/v1/chat/completions'),
    'gemini': ('GEMINI_API_KEY', 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'),
    'openrouter': ('OPENROUTER_API_KEY', 'https://openrouter.ai/api/v1/chat/completions'),
    'anthropic': ('ANTHROPIC_API_KEY', 'https://api.anthropic.com/v1/messages'),
}


def endpoint_url(name, base):
    """Đổi base URL của dịch vụ trung gian thành endpoint đầy đủ.

    OpenAI-compatible: base thường gồm /v1, ví dụ https://gateway.example/v1 -> .../v1/chat/completions.
    Anthropic-compatible: https://gateway.example hoặc .../v1 -> .../v1/messages.
    """
    base = base.strip().rstrip('/')
    if not base.startswith(('https://', 'http://')):
        raise ValueError(f'{name.upper()}_BASE_URL phải bắt đầu bằng https:// hoặc http://')
    suffix = '/messages' if name == 'anthropic' else '/chat/completions'
    if base.endswith(suffix):
        return base
    if name == 'anthropic' and not base.endswith('/v1'):
        base += '/v1'
    return base + suffix


RETRY_STATUS = {429, 500, 502, 503, 504}   # lỗi tạm thời của provider/gateway
MAX_ATTEMPTS = 3
MAX_RETRY_WAIT = 30
DEFAULT_TIMEOUT = 180   # giây; model có reasoning (nhất là judge) có thể trả lời > 60s


def request_timeout():
    """Thời gian chờ mỗi request, chỉnh bằng PROVIDER_TIMEOUT (giây)."""
    try:
        value = float(os.getenv('PROVIDER_TIMEOUT', DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def _retry_delay(exc, attempt):
    """Chờ theo Retry-After nếu hợp lệ, nếu không thì backoff 2s, 4s, ..."""
    retry_after = exc.headers.get('Retry-After') if getattr(exc, 'headers', None) else None
    try:
        return min(float(retry_after), MAX_RETRY_WAIT)
    except (TypeError, ValueError):
        return 2 ** attempt


def post_json(url, headers, payload):
    data = json.dumps(payload).encode()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json', **headers})
        try:
            with urllib.request.urlopen(req, timeout=request_timeout()) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                time.sleep(_retry_delay(exc, attempt))
                continue
            # Do not print response bodies: they may contain account or request data.
            raise RuntimeError(f'Provider trả HTTP {exc.code}; kiểm tra model, key và quota.') from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f'Không kết nối được provider: {getattr(exc, "reason", exc)}') from exc


class Provider:
    """complete(messages, schemas) -> {message, calls, finish, usage}.

    messages: system/user/assistant/tool theo format Chat Completions. message trả về
    cần được append nguyên vẹn; tool result có role, tool_call_id, content (JSON string).
    """
    def __init__(self, name, model, base_url=None):
        if name not in CONFIG:
            raise ValueError(f'Provider không hợp lệ: {name}')
        if not model:
            raise ValueError('Cần đặt MODEL là model có hỗ trợ tool calling.')
        self.name, self.model = name, model
        key_name, self.url = CONFIG[name]
        # Dịch vụ trung gian: truyền base_url (hoặc --base-url), nếu không thì đọc <PROVIDER>_BASE_URL,
        # ví dụ OPENAI_BASE_URL=https://gateway.example/v1. Không có cả hai thì dùng URL mặc định.
        base = base_url or os.getenv(f'{name.upper()}_BASE_URL')
        if base:
            self.url = endpoint_url(name, base)
        self.key = os.getenv(key_name)
        if not self.key:
            raise ValueError(f'Thiếu {key_name}; xem README.md.')

    def complete(self, messages, schemas):
        if self.name == 'anthropic':
            return self._anthropic(messages, schemas)
        payload = {'model': self.model, 'messages': messages}
        if schemas:
            payload['tools'] = schemas
        data = post_json(self.url, {'Authorization': f'Bearer {self.key}'}, payload)
        choice = data['choices'][0]
        message = choice['message']
        calls = [{'id': c['id'], 'name': c['function']['name'], 'arguments': c['function']['arguments']} for c in message.get('tool_calls', [])]
        return {'message': message, 'calls': calls, 'finish': choice.get('finish_reason'), 'usage': data.get('usage', {})}

    def _anthropic(self, messages, schemas):
        system = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
        native = []
        for m in messages:
            if m['role'] == 'system':
                continue
            if m['role'] == 'assistant':
                blocks = []
                if m.get('content'):
                    blocks.append({'type': 'text', 'text': m['content']})
                for call in m.get('tool_calls', []):
                    blocks.append({'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'], 'input': json.loads(call['function']['arguments'])})
                native.append({'role': 'assistant', 'content': blocks})
            elif m['role'] == 'tool':
                block = {'type': 'tool_result', 'tool_use_id': m['tool_call_id'], 'content': m['content']}
                if native and native[-1]['role'] == 'user' and isinstance(native[-1]['content'], list):
                    native[-1]['content'].append(block)
                else:
                    native.append({'role': 'user', 'content': [block]})
            else:
                native.append({'role': 'user', 'content': m['content']})
        tools = [{'name': x['function']['name'], 'description': x['function']['description'], 'input_schema': x['function']['parameters']} for x in schemas]
        payload = {'model': self.model, 'max_tokens': 1024, 'system': system, 'messages': native}
        if tools:
            payload['tools'] = tools
        data = post_json(self.url, {'x-api-key': self.key, 'anthropic-version': '2023-06-01'}, payload)
        blocks = data['content']
        calls = [{'id': b['id'], 'name': b['name'], 'arguments': json.dumps(b['input'], ensure_ascii=False)} for b in blocks if b['type'] == 'tool_use']
        text = '\n'.join(b['text'] for b in blocks if b['type'] == 'text')
        message = {'role': 'assistant', 'content': text or None}
        if calls:
            message['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}} for c in calls]
        reason = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length'}.get(data.get('stop_reason'), data.get('stop_reason'))
        return {'message': message, 'calls': calls, 'finish': reason, 'usage': data.get('usage', {})}
