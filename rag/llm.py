"""
Chat providers.

    OpenAIChat      any OpenAI-compatible chat endpoint (OPENAI_BASE_URL optional)
    AnthropicChat   the Anthropic SDK (claude-opus-5 by default)
    NoLLM           no model at all: answers by quoting the retrieved passages

Each provider exposes:
    stream(system, messages)  -> iterator of text fragments
    complete(system, messages) -> str

`messages` is a list of {"role": "user" | "assistant", "content": str}.
"""
from __future__ import annotations

from typing import Iterator, Protocol

Message = dict[str, str]


class ChatProvider(Protocol):
    name: str
    generates: bool

    def stream(self, system: str, messages: list[Message], *, max_tokens: int) -> Iterator[str]: ...

    def complete(self, system: str, messages: list[Message], *, max_tokens: int) -> str: ...


class ProviderError(RuntimeError):
    """Raised when a provider cannot answer (auth, rate limit, network). The UI reports it in-world."""


class OpenAIChat:
    generates = True

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.name = f"openai:{model}"

    def _messages(self, system: str, messages: list[Message]) -> list[dict]:
        return [{"role": "system", "content": system}, *messages]

    def stream(self, system: str, messages: list[Message], *, max_tokens: int) -> Iterator[str]:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                max_tokens=max_tokens,
                temperature=0.3,
                stream=True,
            )
            for event in response:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except openai.RateLimitError as exc:
            raise ProviderError("The archive is busy right now. Try again in a moment.") from exc
        except openai.AuthenticationError as exc:
            raise ProviderError("The archive's key was refused. Check OPENAI_API_KEY.") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError("The archive could not reach its model.") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"The archive's model returned an error ({exc.status_code}).") from exc

    def complete(self, system: str, messages: list[Message], *, max_tokens: int) -> str:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except openai.APIError as exc:
            raise ProviderError("The archive's model returned an error.") from exc
        choice = response.choices[0] if response.choices else None
        return (choice.message.content or "").strip() if choice else ""


class AnthropicChat:
    generates = True

    def __init__(self, api_key: str, model: str = "claude-opus-5"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"

    def stream(self, system: str, messages: list[Message], *, max_tokens: int) -> Iterator[str]:
        import anthropic

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
                if final.stop_reason == "refusal":
                    yield "\n\nThe archive declined to answer that."
        except anthropic.RateLimitError as exc:
            raise ProviderError("The archive is busy right now. Try again in a moment.") from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderError("The archive's key was refused. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("The archive could not reach its model.") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"The archive's model returned an error ({exc.status_code}).") from exc

    def complete(self, system: str, messages: list[Message], *, max_tokens: int) -> str:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except anthropic.APIError as exc:
            raise ProviderError("The archive's model returned an error.") from exc
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text").strip()


class NoLLM:
    """
    No generation. The archive answers by returning the retrieved passages,
    clearly labelled, so retrieval can be inspected and the refusal path
    tested without a key. `generates` is False so the UI can say so.
    """

    name = "none"
    generates = False

    def stream(self, system: str, messages: list[Message], *, max_tokens: int) -> Iterator[str]:
        yield self.complete(system, messages, max_tokens=max_tokens)

    def complete(self, system: str, messages: list[Message], *, max_tokens: int) -> str:
        # The pipeline never calls this for a grounded answer; see Archive._extractive.
        return ""


def make_chat(provider: str, *, model: str, openai_key: str | None, openai_base_url: str | None, anthropic_key: str | None) -> ChatProvider:
    if provider == "anthropic":
        if not anthropic_key:
            raise RuntimeError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY.")
        return AnthropicChat(api_key=anthropic_key, model=model)
    if provider == "openai":
        if not openai_key:
            raise RuntimeError("LLM_PROVIDER=openai needs OPENAI_API_KEY.")
        return OpenAIChat(api_key=openai_key, model=model, base_url=openai_base_url)
    return NoLLM()
