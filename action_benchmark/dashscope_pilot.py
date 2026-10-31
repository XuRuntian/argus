"""Bounded, no-retry DashScope transport for a small benchmark pilot, not batch billing."""
from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path

import httpx

from label.atomic import write_atomic

MODEL = 'qwen3.6-plus'
ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
PRICE_SOURCE = 'https://help.aliyun.com/zh/model-studio/qwen3-6-plus'
IMAGE_SOURCE = 'https://help.aliyun.com/en/model-studio/vision'


def reserve_cny(messages, max_tokens):
    # UTF-8 bytes upper-bound text tokens. High-resolution images are explicitly off:
    # the documented default pixel ceiling is 2,621,440 (2,560 visual tokens).
    # Include framing slack and use the higher pricing tier even for short inputs.
    tokens = 2048
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            tokens += len(content.encode())
            continue
        for part in content:
            if part['type'] == 'text':
                tokens += len(part['text'].encode())
            elif part['type'] == 'image_url' and part['image_url']['url'].startswith('data:image/'):
                tokens += 2560 + 128
            else:
                raise ValueError('pilot accepts text and inline images only')
    if tokens > 991808:
        raise ValueError('request exceeds the conservative context bound')
    return (tokens * 8 + max_tokens * 48) / 1_000_000


class PilotClient:
    """One fresh receipt directory per run; failed dispatches retain their reservation."""

    model_name = MODEL

    def __init__(self, out, *, max_calls, cap_cny, max_tokens=8192):
        if type(max_calls) is not int or not 1 <= max_calls <= 8:
            raise ValueError('pilot permits 1 to 8 calls')
        if not math.isfinite(cap_cny) or not 0 < cap_cny <= 10:
            raise ValueError('pilot cap must be in (0, 10] CNY')
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError('pilot output limit must be in [1, 8192]')
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.max_calls, self.cap_cny, self.max_tokens = max_calls, cap_cny, max_tokens
        self.calls, self.reserved = 0, 0.0
        self.blocked = False

    def send(self, messages):
        if self.blocked:
            raise ValueError('pilot stopped after authentication failure or unknown dispatch outcome')
        key = os.environ.get('DASHSCOPE_API_KEY')
        if not key:
            raise ValueError('DASHSCOPE_API_KEY is not configured')
        amount = reserve_cny(messages, self.max_tokens)
        if self.calls >= self.max_calls or self.reserved + amount > self.cap_cny:
            raise ValueError('pilot call or conservative cost limit reached; no dispatch')
        self.calls += 1
        self.reserved += amount
        path = self.out / f'call_{self.calls:03d}.json'
        body = {'model': MODEL, 'messages': messages, 'max_tokens': self.max_tokens,
                'enable_thinking': False, 'vl_high_resolution_images': False,
                'temperature': 0.2, 'response_format': {'type': 'json_object'}}
        receipt = {'endpoint': ENDPOINT, 'model': MODEL, 'reserved_cny': amount,
                   'price_source': PRICE_SOURCE, 'image_token_source': IMAGE_SOURCE,
                   'status': 'dispatch_claimed', 'request': body}
        write_atomic(path, receipt)
        try:
            with httpx.Client(timeout=300, trust_env=False, follow_redirects=False) as client:
                response = client.post(ENDPOINT, headers={'Authorization': f'Bearer {key}'}, json=body)
            receipt['http_status'] = response.status_code
            # Never copy credentials from exception messages or response bodies into logs.
            receipt['response_text'] = response.text.replace(key, '<redacted>')
            if response.status_code != 200:
                if response.status_code in (401, 403):
                    self.blocked = True
                receipt['status'] = 'http_error'
                write_atomic(path, receipt)
                raise ValueError(f'DashScope returned HTTP {response.status_code}; no automatic retry')
            result = response.json()
            receipt['status'] = 'received'
            receipt['usage'] = result.get('usage')
            usage = result.get('usage') or {}
            if type(usage.get('prompt_tokens')) is int and type(usage.get('completion_tokens')) is int:
                price_in, price_out = (2, 12) if usage['prompt_tokens'] <= 256000 else (8, 48)
                receipt['estimated_cny'] = (usage['prompt_tokens'] * price_in + usage['completion_tokens'] * price_out) / 1e6
            write_atomic(path, receipt)
            return result
        except Exception:
            if receipt['status'] == 'dispatch_claimed':
                self.blocked = True
                receipt['status'] = 'outcome_unknown'
                write_atomic(path, receipt)
            raise

    def generate(self, prompt, images=None, system_prompt='Return strict JSON only.', model_override=None, **kwargs):
        if model_override not in (None, MODEL):
            raise ValueError('pilot model cannot be overridden')
        content = [{'type': 'text', 'text': prompt}]
        from PIL import Image
        for image in images or []:
            if isinstance(image, (str, Path)):
                with Image.open(image) as source:
                    picture = source.convert('RGB')
            else:
                picture = image.convert('RGB') if isinstance(image, Image.Image) else Image.fromarray(image).convert('RGB')
            buffer = io.BytesIO()
            picture.save(buffer, format='JPEG', quality=85)
            content.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()}})
        result = self.send([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}])
        if result['choices'][0].get('finish_reason') != 'stop':
            raise ValueError('pilot response did not finish normally; retained without retry')
        return result['choices'][0]['message']['content']

    def extract_json(self, text):
        return json.loads(text)


def label_one(job, episode_id, out):
    from action_benchmark.dataset import read
    from label import episode, harness

    job, out = Path(job).resolve(), Path(out).resolve()
    manifest = read(job / 'manifest.json')
    selected = next(row for row in manifest['episodes'] if row['episode_id'] == episode_id)
    ep = job / 'episodes' / selected['directory']
    client = PilotClient(out / 'receipts', max_calls=1, cap_cny=2.5)
    kwargs = dict(model=MODEL, reasoning='none', max_tokens=8192, timeout=300,
                  cell_w=448, sample_fps=2, inspection='off')
    record = harness.label_episode(ep, out / 'dry-run.json', api_key='dry-run', dry_run=True, **kwargs)
    request = episode.build_request(ep, cell_w=448, detail=harness.DETAIL, grid_cols=harness.GRID_COLS,
                                    inspect_evidence=True, model=MODEL, reasoning='none', sample_fps=2, inspection='off',
                                    derived_dir=out / '.derived' / 'jaws')
    problem = harness.saved_input_problem(record, ep)
    if problem:
        raise ValueError(problem)
    result = client.send([{'role': 'user', 'content': request['content']}])
    choice = result['choices'][0]
    record.pop('dry_run', None)
    record.update(provider_name='DashScope', transport='dashscope_pilot_v1', model_served=result.get('model'),
                  generation_id=result.get('id'), usage=result.get('usage'), finish_reason=choice.get('finish_reason'),
                  response_text=choice['message']['content'])
    record['config']['request_images']['dispatch_status'] = 'dispatched'
    record['labels'], record['parse_ok'] = harness.read_reply(record['response_text'], record)
    if choice.get('finish_reason') != 'stop':
        record['parse_ok'] = False
    write_atomic(out / 'labels' / (selected['directory'] + '.json'), record)
    return {'parse_ok': record['parse_ok'], 'usage': record['usage'],
            'problems': harness.annotation_problems(record)}
