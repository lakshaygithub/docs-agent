"""A thin wrapper so the agent doesn't care which model service it talks to.

chat(messages, tools, tool_choice) returns a plain dict:
    {"content": str | None,
     "tool_calls": [{"id", "name", "arguments"}],   # arguments is a JSON string
     "usage": {"prompt_tokens", "completion_tokens"}}
"""

import os


class OpenAIChatModel:
    """Azure OpenAI if AZURE_OPENAI_ENDPOINT is set, otherwise plain OpenAI."""

    def __init__(self):
        from openai import AzureOpenAI, OpenAI

        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                max_retries=3,  # the SDK retries rate limits and server errors with backoff
            )
            self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # the deployment name in Azure
        else:
            self.client = OpenAI(max_retries=3)
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,  # "required" forces a tool call
            parallel_tool_calls=False,  # one tool call per step, so the model sees each result before acting
            temperature=0,
            max_tokens=1024,  # stops a model that starts repeating itself
            timeout=120,
        )
        msg = resp.choices[0].message
        return {
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        }


class ScriptedModel:
    """Plays back prepared replies. Used in tests so they run without an API key."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self.choices: list[str] = []  # tool_choice on each call

    def chat(self, messages, tools, tool_choice="auto"):
        self.calls.append([dict(m) for m in messages])
        self.choices.append(tool_choice)
        if not self.replies:
            raise RuntimeError("ScriptedModel ran out of replies")
        reply = self.replies.pop(0)
        return {"content": reply.get("content"), "tool_calls": reply.get("tool_calls", []),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
