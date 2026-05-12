import asyncio
import os
import re
import time
from typing import List


# Family detection — drives which request kwargs are valid for the model.
_REASONING_PATTERNS = (
    re.compile(r'^o\d', re.I),       # o1, o3, o4-mini, ...
    re.compile(r'^gpt-5', re.I),     # GPT-5 family (also reasoning)
)
_NEW_CHAT_PATTERNS = (
    re.compile(r'^gpt-4\.1', re.I),  # gpt-4.1 family — same sampling as 4o, uses max_completion_tokens
)


def _is_reasoning_model(model: str) -> bool:
    """o-series and GPT-5 — reject sampling controls; require max_completion_tokens."""
    return any(p.match(model) for p in _REASONING_PATTERNS)


def _prefers_max_completion_tokens(model: str) -> bool:
    return _is_reasoning_model(model) or any(p.match(model) for p in _NEW_CHAT_PATTERNS)


class OpenAIAnnotationModel:
    """OpenAI / OpenAI-compatible annotation backend, multi-tier.

    Adapts per-model parameter conventions based on ``api_model``:

    - **gpt-4o\\***: standard chat completions — accepts ``temperature``,
      ``top_p``, ``seed``, ``frequency_penalty``, ``presence_penalty``,
      ``response_format``; uses ``max_tokens``.
    - **gpt-4.1\\***: same sampling support, but uses
      ``max_completion_tokens``.
    - **o-series** (o3, o4-mini, ...): reasoning. Uses
      ``max_completion_tokens``; **rejects** all sampling knobs;
      accepts ``reasoning_effort`` (low/medium/high).
    - **gpt-5\\***: reasoning. Same as o-series plus ``verbosity``
      (low/medium/high) and ``reasoning_effort='minimal'`` for near-zero
      thinking.

    Sampling fields not supported by the target model are silently dropped
    with a one-time warning, so a single YAML sampling block can be reused
    across families without per-model edits.
    """

    def __init__(self, config: dict):
        from openai import AsyncOpenAI, OpenAI

        api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        base_url = config.get('api_base_url', None)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = config.get('api_model', 'gpt-4o')
        self.image_detail = config.get('image_detail', 'low')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)

        self._request_kwargs = self._build_request_kwargs(config)

    def _build_request_kwargs(self, config: dict) -> dict:
        kwargs = {}
        is_reasoning = _is_reasoning_model(self.model)
        use_mct = _prefers_max_completion_tokens(self.model)

        # Token budget — newer / reasoning models renamed the field.
        max_out = config.get('max_tokens', 700)
        if use_mct:
            kwargs['max_completion_tokens'] = max_out
        else:
            kwargs['max_tokens'] = max_out

        sampling_keys = (
            'temperature', 'top_p', 'frequency_penalty',
            'presence_penalty', 'seed',
        )
        if is_reasoning:
            dropped = [k for k in sampling_keys if k in config]
            if dropped:
                print(
                    f"[openai-annotation] WARN: reasoning model {self.model!r} "
                    f"ignores sampling params {dropped}; dropping from request."
                )
        else:
            for k in sampling_keys:
                if k in config:
                    kwargs[k] = config[k]
            # No legacy setdefault here — absence of a sampling key means
            # "use the API default" (temperature=1, top_p=1 for chat models).
            # Configs that want greedy must set `temperature: 0` explicitly.

        if is_reasoning:
            if 'reasoning_effort' in config:
                kwargs['reasoning_effort'] = config['reasoning_effort']
            # `verbosity` is GPT-5 only.
            if 'verbosity' in config and self.model.lower().startswith('gpt-5'):
                kwargs['verbosity'] = config['verbosity']

        if 'response_format' in config:
            kwargs['response_format'] = config['response_format']

        return kwargs

    def vlm_inference(self, inputs: dict) -> List[str]:
        """Run inference using OpenAI API.

        Args:
            inputs: dict containing 'messages' (Qwen-format message list).

        Returns:
            List with a single response string.
        """
        messages = inputs['messages']
        openai_messages = self._convert_messages(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **self._request_kwargs,
                )
                return [response.choices[0].message.content]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"OpenAI API failed after {self.max_retries} attempts: {e}")
                    return [""]

    async def vlm_inference_async(self, inputs: dict) -> List[str]:
        """Async version of vlm_inference for use with asyncio.gather concurrency.

        OpenAI calls are I/O-bound, so in-process asyncio scales much better
        than subprocess-level DP: a single process handles N concurrent
        requests with one event loop and one dataset load.
        """
        messages = inputs['messages']
        openai_messages = self._convert_messages(messages)

        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **self._request_kwargs,
                )
                return [response.choices[0].message.content]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"OpenAI API failed after {self.max_retries} attempts: {e}")
                    return [""]

    def _convert_messages(self, qwen_messages: list) -> list:
        """Convert Qwen-format messages to OpenAI API format.

        Handles:
        - "type": "text" -> same
        - "type": "video" with base64 data URI frames -> multiple "type": "image_url" entries
        - "type": "image" -> "type": "image_url"
        """
        openai_messages = []
        for msg in qwen_messages:
            role = msg['role']
            content = msg['content']

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                openai_content = []
                for item in content:
                    if item['type'] == 'text':
                        openai_content.append({
                            "type": "text",
                            "text": item['text'],
                        })
                    elif item['type'] == 'video':
                        # Convert video frames to individual image_url entries
                        for frame_uri in item.get('video', []):
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": frame_uri,
                                    "detail": self.image_detail,
                                },
                            })
                    elif item['type'] == 'image':
                        image_uri = item.get('image', '')
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_uri,
                                "detail": self.image_detail,
                            },
                        })
                openai_messages.append({"role": role, "content": openai_content})

        return openai_messages
