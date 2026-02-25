import os
import time
from typing import List


class OpenAIAnnotationModel:
    """CoT annotation model backend using OpenAI API (or any OpenAI-compatible endpoint)."""

    def __init__(self, config: dict):
        from openai import OpenAI

        api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        base_url = config.get('api_base_url', None)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = config.get('api_model', 'gpt-4o')
        self.max_tokens = config.get('max_tokens', 700)
        self.temperature = config.get('temperature', 0)
        self.image_detail = config.get('image_detail', 'low')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)

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
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return [response.choices[0].message.content]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
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
