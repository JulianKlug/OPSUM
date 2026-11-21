"""
Driver for the local Ollama server. The only module that speaks HTTP.

Callers ask for text or for a JSON object; transport, decoding options and
retries stay in here. Patient data never leaves the machine: the server
listens on localhost only.
"""
import json
import time
from enum import Enum

import requests

DEFAULT_BASE_URL = 'http://127.0.0.1:11434'
CHAT_ENDPOINT = '/api/chat'

HTTP_OK = 200
REQUEST_TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

# Keep the weights in GPU memory between calls: reloading takes ~30s
MODEL_KEEP_ALIVE = '30m'


class Thinking(Enum):
    """Native reasoning before the answer. Several times slower; only some models support it."""
    OFF = False
    ON = True


class LLMResponseError(RuntimeError):
    """The model could not be reached or did not return what was asked for."""


class OllamaClient:
    def __init__(self, model: str, context_tokens: int, seed: int = 42, base_url: str = DEFAULT_BASE_URL):
        self.model = model
        self._url = base_url + CHAT_ENDPOINT

        # Temperature 0 and a fixed seed: the same prompt gives the same answer
        self._options = {'temperature': 0, 'seed': seed, 'num_ctx': context_tokens}

    def chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int,
                  thinking: Thinking = Thinking.OFF) -> str:
        """Free-text answer."""
        return self._chat(system_prompt, user_prompt, max_tokens, thinking, response_format=None).strip()

    def chat_json(self, system_prompt: str, user_prompt: str, json_schema: dict, max_tokens: int,
                  thinking: Thinking = Thinking.OFF) -> dict:
        """Answer constrained to the given JSON schema, decoded."""
        last_error = None
        for _ in range(MAX_ATTEMPTS):
            content = self._chat(system_prompt, user_prompt, max_tokens, thinking, response_format=json_schema)

            try:
                return json.loads(content)
            except json.JSONDecodeError as error:
                # Typically a reply cut by the token limit
                last_error = error

        raise LLMResponseError(f'No valid JSON after {MAX_ATTEMPTS} attempts: {last_error}')

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens: int, thinking: Thinking, response_format) -> str:
        payload = {
            'model': self.model,
            'stream': False,
            # Always explicit: reasoning models think by default, which multiplies the runtime
            'think': thinking.value,
            'keep_alive': MODEL_KEEP_ALIVE,
            'options': dict(self._options, num_predict=max_tokens),
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        }
        if response_format is not None:
            payload['format'] = response_format

        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                time.sleep(RETRY_DELAY_SECONDS)

            try:
                response = requests.post(self._url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as error:
                last_error = error
                continue

            if response.status_code != HTTP_OK:
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
                continue

            return response.json()['message']['content']

        raise LLMResponseError(f'Ollama call failed after {MAX_ATTEMPTS} attempts: {last_error}')
