from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import random
import re
import shlex
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

import litellm
import litellm.types.utils
from openai import AzureOpenAI, OpenAI, NOT_GIVEN
import openai  # ← NEW: for exception classes
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, SecretStr
from swerex.exceptions import SwerexException
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    retry_if_exception_message,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv

from sweagent import REPO_ROOT
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FunctionCallingFormatError,
    InstanceCallLimitExceededError,
    InstanceCostLimitExceededError,
    ModelConfigurationError,
    TotalCostLimitExceededError,
)
from sweagent.tools.tools import ToolConfig
from sweagent.types import History, HistoryItem
from sweagent.utils.log import get_logger

try:
    import readline  # noqa: F401
except ImportError:
    readline = None

litellm.suppress_debug_info = True


_THREADS_THAT_USED_API_KEYS = []
"""Keeps track of thread orders so that we can choose the same API key for the same thread."""


class RetryConfig(PydanticBaseModel):
    """This configuration object specifies how many times to retry a failed LM API call."""

    retries: int = 20
    """Number of retries"""
    min_wait: float = 10
    """Minimum wait time between retries (random exponential wait)"""
    max_wait: float = 120
    """Maximum wait time between retries (random exponential wait)"""


class GenericAPIModelConfig(PydanticBaseModel):
    """This configuration object specifies a LM like GPT4 or similar.
    The model will be served with the help of the `litellm` library.
    """

    name: str = Field(description="Name of the model.")

    per_instance_cost_limit: float = Field(
        default=3.0,
        description="Cost limit for every instance (task).",
    )
    total_cost_limit: float = Field(default=0.0, description="Total cost limit.")
    per_instance_call_limit: int = Field(default=0, description="Per instance call limit.")
    temperature: float = 0.0
    """Sampling temperature"""
    top_p: float | None = 1.0
    """Sampling top-p"""
    api_base: str | None = None
    api_version: str | None = None
    api_key: SecretStr | None = None
    """API key to the model. We recommend using environment variables to set this instead
    or putting your environment variables in a `.env` file.
    You can concatenate more than one key by separating them with `:::`, e.g.,
    `key1:::key2`.
    If field starts with `$`, it will be interpreted as an environment variable.
    """
    stop: list[str] = []
    """Custom stop sequences"""

    completion_kwargs: dict[str, Any] = {}
    """Additional kwargs to pass to `litellm.completion`"""

    convert_system_to_user: bool = False
    """Whether to convert system messages to user messages. This is useful for
    models that do not support system messages like o1.
    """

    retry: RetryConfig = RetryConfig()
    """Retry configuration: How often to retry after a failure (e.g., from a rate limit)
    etc.
    """

    delay: float = 0.0
    """Minimum delay before querying (this can help to avoid overusing the API if sharing
    it with other people).
    """

    fallbacks: list[dict[str, Any]] = []
    """List of fallbacks to try if the main model fails
    See https://docs.litellm.ai/docs/completion/reliable_completions#fallbacks-sdk
    for more information.
    """

    choose_api_key_by_thread: bool = True
    """Whether to choose the API key based on the thread name (if multiple are configured).
    This ensures that with
    run-batch, we use the same API key within a single-thread so that prompt caching still works.
    """

    max_input_tokens: int | None = None
    """If set, this will override the max input tokens for the model that we usually look
    up from `litellm.model_cost`.
    Use this for local models or if you want to set a custom max input token limit.
    If this value is exceeded, a `ContextWindowExceededError` will be raised.
    Set this to 0 to disable this check.
    """

    max_output_tokens: int | None = None
    """If set, this will override the max output tokens for the model that we usually look
    up from `litellm.model_cost`.
    Use this for local models or if you want to set a custom max output token limit.
    If this value is exceeded, a `ContextWindowExceededError` will be raised.
    Set this to 0 to disable this check.
    """

    litellm_model_registry: str | None = None
    """If set, this will override the default model registry for litellm.
    Use this for local models or models not (yet) in the default litellm model registry for tracking costs.
    """

    # pydantic
    model_config = ConfigDict(extra="forbid")

    def get_api_keys(self) -> list[str]:
        """Returns a list of API keys that were explicitly set in this config.
        Does not return API keys that were set via environment variables/.env
        """
        if self.api_key is None:
            return []
        api_key = self.api_key.get_secret_value()
        if not api_key:
            return []
        if api_key.startswith("$"):
            env_var_name = api_key[1:]
            api_key = os.getenv(env_var_name, "")
            if not api_key:
                get_logger("swea-config", emoji="🔧").warning(f"Environment variable {env_var_name} not set")
                return []
        return api_key.split(":::")

    def choose_api_key(self) -> str | None:
        """Chooses an API key based on the API keys explicitly set in this config.
        If no API keys are set, returns None (which means that the API key will be
        taken from the environment variables/.env file).
        """
        api_keys = self.get_api_keys()
        if not api_keys:
            return None
        if not self.choose_api_key_by_thread:
            return random.choice(api_keys)
        thread_name = threading.current_thread().name
        if thread_name not in _THREADS_THAT_USED_API_KEYS:
            _THREADS_THAT_USED_API_KEYS.append(thread_name)
        thread_idx = _THREADS_THAT_USED_API_KEYS.index(thread_name)
        key_idx = thread_idx % len(api_keys)
        get_logger("config", emoji="🔧").debug(
            f"Choosing API key {key_idx} for thread {thread_name} (idx {thread_idx})"
        )
        return api_keys[key_idx]

    @property
    def id(self) -> str:
        name = self.name.replace("/", "--")
        if self.top_p is not None:
            top_p = f"{self.top_p:.2f}"
        else:
            top_p = "None"
        temperature = f"{self.temperature:.2f}"
        per_instance_cost_limit = f"{self.per_instance_cost_limit:.2f}"
        return f"{name}__t-{temperature}__p-{top_p}__c-{per_instance_cost_limit}"


class ReplayModelConfig(GenericAPIModelConfig):
    replay_path: Path = Field(description="Path to replay file when using the replay model.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )

    name: Literal["replay"] = Field(default="replay", description="Model name.")

    model_config = ConfigDict(extra="forbid")


class InstantEmptySubmitModelConfig(GenericAPIModelConfig):
    """Model that immediately submits an empty patch"""

    name: Literal["instant_empty_submit"] = Field(default="instant_empty_submit", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )
    delay: float = 0.0
    """Delay before answering"""

    model_config = ConfigDict(extra="forbid")


class HumanModelConfig(GenericAPIModelConfig):
    name: Literal["human"] = Field(default="human", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(default=0.0, description="Cost limit for all instances (tasks).")
    cost_per_call: float = 0.0
    catch_eof: bool = True
    """Whether to catch EOF and return 'exit' when ^D is pressed. Set to False when used in human_step_in mode."""
    model_config = ConfigDict(extra="forbid")


class HumanThoughtModelConfig(HumanModelConfig):
    name: Literal["human_thought"] = Field(default="human_thought", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )
    cost_per_call: float = 0.0

    model_config = ConfigDict(extra="forbid")


class CopilotClaudeModelConfig(GenericAPIModelConfig):
    """Configuration for GitHub Copilot Claude API model"""

    name: Literal["claude-sonnet-4"] = Field(default="claude-sonnet-4", description="Model name.")
    
    api_base: str | None = Field(
        default="https://api.enterprise.githubcopilot.com",
        description="GitHub Copilot API base URL"
    )
    api_version: str | None = Field(
        default="2025-05-01",
        description="GitHub Copilot API version"
    )
    
    vscode_copilot_dir: str | None = Field(
        default="~/repo/vscode-copilot",
        description="Path to vscode-copilot directory. If not provided, will use VSCODE_COPILOT_DIR env var or ~/vscode-copilot"
    )
    
    max_tokens: int = Field(
        default=8192,
        description="Maximum tokens for completion"
    )

    model_config = ConfigDict(extra="forbid")


ModelConfig = Annotated[
    GenericAPIModelConfig
    | ReplayModelConfig
    | InstantEmptySubmitModelConfig
    | HumanModelConfig
    | HumanThoughtModelConfig
    | CopilotClaudeModelConfig,
    Field(union_mode="left_to_right"),
]


class GlobalStats(PydanticBaseModel):
    """This class tracks usage numbers (costs etc.) across all instances."""

    total_cost: float = 0
    """Cumulative cost for all instances so far"""

    last_query_timestamp: float = 0
    """Timestamp of the last query. Currently only used with API models."""


GLOBAL_STATS = GlobalStats()
"""This object tracks usage numbers (costs etc.) across all instances.
Please use the `GLOBAL_STATS_LOCK` lock when accessing this object to avoid race conditions.
"""

GLOBAL_STATS_LOCK = Lock()
"""Lock for accessing `GLOBAL_STATS` without race conditions"""


class InstanceStats(PydanticBaseModel):
    """This object tracks usage numbers (costs etc.) for a single instance."""

    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) + getattr(other, field) for field in self.model_fields.keys()},
        )

    def __sub__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) - getattr(other, field) for field in self.model_fields.keys()},
        )


class AbstractModel(ABC):
    def __init__(self, config: ModelConfig, tools: ToolConfig):
        self.config: ModelConfig
        self.stats: InstanceStats

    def reset_stats(self):
        self.stats = InstanceStats()

    @abstractmethod
    def query(self, history: History, action_prompt: str = "> ") -> dict: ...

    @property
    def instance_cost_limit(self) -> float:
        """Cost limit for the model. Returns 0 if there is no limit."""
        return 0


def _handle_raise_commands(action: str) -> None:
    if action == "raise_runtime":
        raise SwerexException()
    elif action == "raise_cost":
        raise CostLimitExceededError()
    elif action == "raise_context":
        raise ContextWindowExceededError()
    elif action.startswith("raise_function_calling"):
        parts = shlex.split(action)
        error_code = parts[1]
        if len(parts) == 3:
            error_message = parts[2]
        assert len(parts) < 4
        raise FunctionCallingFormatError(error_message, error_code)  # type: ignore


class HumanModel(AbstractModel):
    def __init__(self, config: HumanModelConfig, tools: ToolConfig):
        """Model that allows for human-in-the-loop"""
        self.logger = get_logger("swea-lm", emoji="🤖")
        self.config: HumanModelConfig = config
        self.stats = InstanceStats()

        # Determine which commands require multi-line input
        self.multi_line_command_endings = {
            command.name: command.end_name for command in tools.commands if command.end_name is not None
        }
        self._readline_histfile = REPO_ROOT / ".swe-agent-human-history"
        self._load_readline_history()

    def _load_readline_history(self) -> None:
        """Load autocomplete history from file"""
        if readline is None:
            return
        if self._readline_histfile.is_file():
            self.logger.debug(f"Loading readline history from {self._readline_histfile}")
            readline.read_history_file(self._readline_histfile)

    def _save_readline_history(self) -> None:
        """Save autocomplete history to file"""
        if readline is None:
            return
        readline.write_history_file(self._readline_histfile)

    def _update_stats(
        self,
    ) -> None:
        self.stats.instance_cost += self.config.cost_per_call
        self.stats.api_calls += 1
        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            msg = f"Instance cost limit exceeded: {self.stats.instance_cost} > {self.config.per_instance_cost_limit}"
            raise InstanceCostLimitExceededError(msg)
        if 0 < self.config.total_cost_limit < self.stats.instance_cost:
            msg = f"Total cost limit exceeded: {self.stats.instance_cost} > {self.config.total_cost_limit}"
            raise TotalCostLimitExceededError(msg)

    def _query(
        self,
        history: History,
        action_prompt: str = "> ",
    ) -> dict:
        """Logic for handling user input to pass to SWEEnv"""
        action = input(action_prompt)
        self._save_readline_history()
        command_name = action.split()[0] if action.strip() else ""

        # Special handling for multi-line input actions (i.e. edit)
        if command_name in self.multi_line_command_endings:
            buffer = [action]
            end_keyword = self.multi_line_command_endings[command_name]
            while True:
                action = input("... ")
                buffer.append(action)
                if action.rstrip() == end_keyword:
                    # Continue reading input until terminating keyword inputted
                    break
            action = "\n".join(buffer)
        elif action.strip() == "start_multiline_command":  # do arbitrary multi-line input
            buffer = []
            while True:
                action = input("... ")
                if action.rstrip() == "end_multiline_command":
                    break
                buffer.append(action)
            action = "\n".join(buffer)
        else:
            # Input has escaped things like \n, so we need to unescape it
            action = action.encode("utf8").decode("unicode_escape")
        if action.strip() and action.strip().split()[0] == "spend_money":
            money = float(action.strip().split()[1])
            self.stats.instance_cost += money
            action = f"echo 'Spent {money} dollars'"
        _handle_raise_commands(action)
        self._update_stats()
        return {"message": action}

    def query(self, history: History, action_prompt: str = "> ", n: int | None = None, **kwargs) -> dict | list[dict]:
        """Wrapper to separate action prompt from formatting"""
        out = []
        n_samples = n or 1
        for _ in range(n_samples):
            try:
                out.append(self._query(history, action_prompt))
            except KeyboardInterrupt:
                print("^C (exit with ^D)")
                out.append(self.query(history, action_prompt))
            except EOFError:
                if self.config.catch_eof:
                    print("\nGoodbye!")
                    out.append({"message": "exit"})
                else:
                    # Re-raise EOFError when catch_eof is disabled
                    raise
        if n is None:
            return out[0]
        return out


class HumanThoughtModel(HumanModel):
    def query(self, history: History, **kwargs) -> dict:
        """Logic for handling user input (both thought + action) to pass to SWEEnv"""
        thought_all = ""
        thought = input("Thought (end w/ END_THOUGHT): ")
        while True:
            if "END_THOUGHT" in thought:
                thought = thought.split("END_THOUGHT")[0]
                thought_all += thought
                break
            thought_all += thought
            thought = input("... ")

        action = super()._query(history, action_prompt="Action: ")

        return {"message": f"{thought_all}\n```\n{action}\n```"}


class ReplayModel(AbstractModel):
    def __init__(self, config: ReplayModelConfig, tools: ToolConfig):
        """Model used for replaying a trajectory (i.e., taking all the actions for the `.traj` file
        and re-issuing them.
        """
        self.config = config
        self.stats = InstanceStats()

        if not self.config.replay_path.exists():
            msg = f"Replay file {self.config.replay_path} not found"
            raise FileNotFoundError(msg)

        self._replays = [
            list(json.loads(x).values())[0] for x in Path(self.config.replay_path).read_text().splitlines(keepends=True)
        ]
        self._replay_idx = 0
        self._action_idx = 0
        self.use_function_calling = tools.use_function_calling
        self.submit_command = tools.submit_command
        self.logger = get_logger("swea-lm", emoji="🤖")

    def _next_replay(self) -> None:
        """Called after last action"""
        self._replay_idx += 1
        self._action_idx = 0

    def query(self, history: History) -> dict:
        """Logic for tracking which replay action to pass to SWEEnv"""
        self.stats.api_calls += 1
        actions = self._replays[self._replay_idx]
        try:
            action = actions[self._action_idx]
        except IndexError:
            # log error
            self.logger.error("Reached end of replay trajectory without submitting. Submitting now.")
            if self.use_function_calling:
                action = {
                    "message": f"Calling `{self.submit_command}` to submit.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_submit",
                            "function": {
                                "name": self.submit_command,
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            else:
                action = f"```\n{self.submit_command}\n```"

        self._action_idx += 1

        # Assuming `submit` is always last action of replay trajectory
        if isinstance(action, str) and action == "submit":
            self._next_replay()
            return {"message": action}

        # Handle both dict and string actions
        if isinstance(action, dict):
            return action
        return {"message": action}


class PredeterminedTestModel(AbstractModel):
    def __init__(self, outputs: list[dict | str]):
        """Model that outputs a predetermined sequence of messages. Useful for testing."""
        self._outputs = outputs
        self._idx = -1
        self.stats = InstanceStats()

    def query(self, *args, **kwargs) -> dict:
        self._idx += 1
        output = self._outputs[self._idx]
        if isinstance(output, str):
            _handle_raise_commands(output)
            return {"message": output}
        if not isinstance(output, dict):
            msg = f"Output must be string or dict, got {type(output)}"
            raise ValueError(msg)
        result = {"message": output["message"]}
        if "tool_calls" in output:
            result["tool_calls"] = output["tool_calls"]
        return result


class InstantEmptySubmitTestModel(AbstractModel):
    def __init__(self, args: InstantEmptySubmitModelConfig, tools: ToolConfig):
        """This model immediately submits. Useful for testing purposes"""
        super().__init__(args, tools)
        self.config: InstantEmptySubmitModelConfig = args
        self.stats = InstanceStats()
        self._action_idx = 0

    def query(self, history: list[dict[str, str]]) -> dict:
        time.sleep(random.uniform(0, self.config.delay))
        # Need to at least do _something_ to submit
        if self._action_idx == 0:
            self._action_idx = 1
            action = (
                "DISCUSSION\n"
                "Let's reproduce the bug by creating a `reproduce.py` file.\n\n"
                "```\n"
                "touch reproduce.py\n"
                "```\n"
            )
        elif self._action_idx == 1:
            self._action_idx = 0
            action = "DISCUSSION\nThe task should be resolved, so let's submit the patch.\n\n```\nsubmit\n```\n"
        self.stats.api_calls += 1
        return {"message": action}


class LiteLLMModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        """Model served by the `litellm` library."""
        # Always copy config to avoid shared state between different instances
        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        if tools.use_function_calling:
            if not litellm.utils.supports_function_calling(model=self.config.name):
                msg = (
                    f"Model {self.config.name} does not support function calling. If your model"
                    " does not support function calling, you can use `parse_function='thought_action'` instead. "
                    "See https://swe-agent.com/latest/faq/ for more information."
                )
                self.logger.warning(msg)
        if self.config.litellm_model_registry is not None:
            with open(self.config.litellm_model_registry) as f:
                model_costs = json.load(f)
                litellm.register_model(model_costs)
        if self.config.max_input_tokens is not None:
            self.model_max_input_tokens = self.config.max_input_tokens
        else:
            self.model_max_input_tokens = litellm.model_cost.get(self.config.name, {}).get("max_input_tokens")

        if self.config.max_output_tokens is not None:
            self.model_max_output_tokens = self.config.max_output_tokens
        else:
            self.model_max_output_tokens = litellm.model_cost.get(self.config.name, {}).get("max_output_tokens")
            # Special handling for Claude 3.7 models to set 64k context by default when beta header not present
            # See https://github.com/SWE-agent/SWE-agent/pull/1016
            is_claude_3_7 = "claude-3-7-sonnet" in self.config.name or "claude-sonnet-4" in self.config.name
            has_128k_beta_header = (
                self.config.completion_kwargs.get("extra_headers", {}).get("anthropic-beta") == "output-128k-2025-02-19"
            )
            if is_claude_3_7 and not has_128k_beta_header:
                self.model_max_output_tokens = 64000
                self.logger.warning(
                    "Claude 3.7/4 models do not support 128k context by default. "
                    "Setting max output tokens to 64k. To enable 128k context, please set the "
                    "completion_kwargs to {'extra_headers': {'anthropic-beta': 'output-128k-2025-02-19'}}."
                )

        self.lm_provider = litellm.model_cost.get(self.config.name, {}).get("litellm_provider")

    @property
    def instance_cost_limit(self) -> float:
        """Cost limit for the model. Returns 0 if there is no limit."""
        return self.config.per_instance_cost_limit

    def _update_stats(self, *, input_tokens: int, output_tokens: int, cost: float) -> None:
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        # Log updated cost values to std. err
        self.logger.debug(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.instance_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        self.logger.debug(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={GLOBAL_STATS.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.config.total_cost_limit < GLOBAL_STATS.total_cost:
            self.logger.warning(f"Cost {GLOBAL_STATS.total_cost:.2f} exceeds limit {self.config.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise TotalCostLimitExceededError(msg)

        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            self.logger.warning(
                f"Cost {self.stats.instance_cost:.2f} exceeds limit {self.config.per_instance_cost_limit:.2f}"
            )
            msg = "Instance cost limit exceeded"
            raise InstanceCostLimitExceededError(msg)

        if 0 < self.config.per_instance_call_limit < self.stats.api_calls:
            self.logger.warning(f"API calls {self.stats.api_calls} exceeds limit {self.config.per_instance_call_limit}")
            msg = "Per instance call limit exceeded"
            raise InstanceCallLimitExceededError(msg)

    def _sleep(self) -> None:
        elapsed_time = time.time() - GLOBAL_STATS.last_query_timestamp
        if elapsed_time < self.config.delay:
            time.sleep(self.config.delay - elapsed_time)
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.last_query_timestamp = time.time()

    def _single_query(
        self, messages: list[dict[str, str]], n: int | None = None, temperature: float | None = None
    ) -> list[dict]:
        self._sleep()
        # Workaround for litellm bug https://github.com/SWE-agent/SWE-agent/issues/1109
        messages_no_cache_control = copy.deepcopy(messages)
        for message in messages_no_cache_control:
            if "cache_control" in message:
                del message["cache_control"]
        input_tokens: int = litellm.utils.token_counter(messages=messages_no_cache_control, model=self.config.name)
        if self.model_max_input_tokens is None:
            msg = (
                f"No max input tokens found for model {self.config.name!r}. "
                "If you are using a local model, you can set `max_input_token` in the model config to override this."
            )
            self.logger.warning(msg)
        elif input_tokens > self.model_max_input_tokens > 0:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)
        extra_args = {}
        if self.config.api_base:
            # Not assigned a default value in litellm, so only pass this if it's set
            extra_args["api_base"] = self.config.api_base
        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools
        # We need to always set max_tokens for anthropic models
        completion_kwargs = self.config.completion_kwargs
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens
        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
                model=self.config.name,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
                top_p=self.config.top_p,
                api_version=self.config.api_version,
                api_key=self.config.choose_api_key(),
                fallbacks=self.config.fallbacks,
                **completion_kwargs,
                **extra_args,
                n=n,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise
        self.logger.debug(f"Response: {response}")
        try:
            cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
            self.logger.debug(f"Error calculating cost: {e}, setting cost to 0.")
            if self.config.per_instance_cost_limit > 0 or self.config.total_cost_limit > 0:
                msg = (
                    f"Error calculating cost: {e} for your model {self.config.name}. If this is ok "
                    "(local models, etc.), please make sure you set `per_instance_cost_limit` and "
                    "`total_cost_limit` to 0 to disable this safety check."
                )
                self.logger.error(msg)
                raise ModelConfigurationError(msg)
            cost = 0
        choices: litellm.types.utils.Choices = response.choices  # type: ignore
        n_choices = n if n is not None else 1
        outputs = []
        output_tokens = 0
        for i in range(n_choices):
            output = choices[i].message.content or ""
            output_tokens += litellm.utils.token_counter(text=output, model=self.config.name)
            output_dict = {"message": output}
            if self.tools.use_function_calling:
                if response.choices[i].message.tool_calls:  # type: ignore
                    tool_calls = [call.to_dict() for call in response.choices[i].message.tool_calls]  # type: ignore
                else:
                    tool_calls = []
                output_dict["tool_calls"] = tool_calls
            outputs.append(output_dict)
        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return outputs

    def _query(
        self, messages: list[dict[str, str]], n: int | None = None, temperature: float | None = None
    ) -> list[dict]:
        if n is None:
            return self._single_query(messages, temperature=temperature)
        outputs = []
        # not needed for openai, but oh well.
        for _ in range(n):
            outputs.extend(self._single_query(messages))
        return outputs

    def query(self, history: History, n: int = 1, temperature: float | None = None) -> list[dict] | dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            exception_info = ""
            if attempt.retry_state.outcome is not None and attempt.retry_state.outcome.exception() is not None:
                exception = attempt.retry_state.outcome.exception()
                exception_info = f" due to {exception.__class__.__name__}: {str(exception)}"

            self.logger.warning(
                f"Retrying LM query: attempt {attempt.retry_state.attempt_number} "
                f"(slept for {attempt.retry_state.idle_for:.2f}s)"
                f"{exception_info}"
            )

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(min=self.config.retry.min_wait, max=self.config.retry.max_wait),
            reraise=True,
            retry=retry_if_not_exception_type(
                (
                    ContextWindowExceededError,
                    CostLimitExceededError,
                    RuntimeError,
                    litellm.exceptions.UnsupportedParamsError,
                    litellm.exceptions.NotFoundError,
                    litellm.exceptions.PermissionDeniedError,
                    litellm.exceptions.ContextWindowExceededError,
                    litellm.exceptions.APIError,
                    litellm.exceptions.ContentPolicyViolationError,
                    TypeError,
                    litellm.exceptions.AuthenticationError,
                    ContentPolicyViolationError,
                    ModelConfigurationError,
                    KeyboardInterrupt,
                )
            ),
            before_sleep=retry_warning,
        ):
            with attempt:
                result = self._query(messages, n=n, temperature=temperature)
        if n is None or n == 1:
            return result[0]
        return result

    def _history_to_messages(
        self,
        history: History,
    ) -> list[dict[str, str]]:
        history = copy.deepcopy(history)

        def get_role(history_item: HistoryItem) -> str:
            if history_item["role"] == "system":
                return "user" if self.config.convert_system_to_user else "system"
            return history_item["role"]

        messages = []
        for history_item in history:
            role = get_role(history_item)
            if role == "tool":
                message = {
                    "role": role,
                    "content": history_item["content"],
                    # Only one tool call per observations
                    "tool_call_id": history_item["tool_call_ids"][0],  # type: ignore
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                message = {"role": role, "content": history_item["content"], "tool_calls": tool_calls}
            else:
                message = {"role": role, "content": history_item["content"]}
            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]
            messages.append(message)
        n_cache_control = str(messages).count("cache_control")
        self.logger.debug(f"n_cache_control: {n_cache_control}")
        return messages


class AzureLLMModel(LiteLLMModel):
    """
    Azure implementation of LiteLLMModel.
    Falls back on the Azure OpenAI SDK instead of `litellm.completion`.
    """

    # All deployments that are available via the public TRAPI endpoint
    AZURE_SUPPORTED_MODELS = ["gpt-4o", "o3", "o3-mini", "o4-mini", "gpt-4.1", "gpt-4.5-preview", "o1", "gpt-4.1-mini"]

    _MODEL_META: dict[str, tuple[str, str, str]] = {
        #  name      -> (version,               instance,       api_version)
        "gpt-4o":  ("2024-05-13", "gcr/preview", "2024-10-21"),
        "o3":      ("2025-04-16", "msrne/shared", "2025-04-01-preview"),
        "o3-mini": ("2025-01-31", "msrne/shared", "2025-04-01-preview"),
        "o4-mini": ("2025-04-16", "msrne/shared", "2025-04-01-preview"),
        "gpt-4.1": ("2025-04-14", "gcr/shared", "2025-04-01-preview"),
        "gpt-4.5-preview": ("2025-02-27", "msrne/shared", "2025-04-01-preview"),
        "o1": ("2024-12-17", "msrne/shared", "2025-04-01-preview"),
        "gpt-4.1-mini": ("2025-04-14", "msrne/shared", "2025-04-01-preview"),
    }

    # Models that don't support custom temperature or top_p
    NOT_TEMPERATURE_MODELS = ["o1", "o3", "o3-mini", "o4-mini"]

    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        if args.name not in self.AZURE_SUPPORTED_MODELS:
            msg = f"{args.name} not in supported Azure models {self.AZURE_SUPPORTED_MODELS}"
            raise ValueError(msg)
        super().__init__(args, tools)

        version, instance, self._api_version = self._MODEL_META[self.config.name]
        self._deployment_name = re.sub(r"[^a-zA-Z0-9._-]", "", f"{self.config.name}_{version}")
        self._endpoint = f"https://trapi.research.microsoft.com/{instance}"

        self._credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(
                    exclude_cli_credential=True,
                    exclude_environment_credential=True,
                    exclude_shared_token_cache_credential=True,
                    exclude_developer_cli_credential=True,
                    exclude_powershell_credential=True,
                    exclude_interactive_browser_credential=True,
                    exclude_visual_studio_code_credentials=True,
                    managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
                ),
            ),
            "api://trapi/.default",
        )

        self._azure_client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            azure_ad_token_provider=self._credential,
            api_version=self._api_version,
        )

    def _single_query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        self._sleep()

        messages_no_cache_control = copy.deepcopy(messages)
        for m in messages_no_cache_control:
            if "cache_control" in m:
                del m["cache_control"]

        input_tokens = litellm.utils.token_counter(
            messages=messages_no_cache_control,
            model=self.config.name,
        )
        if self.model_max_input_tokens is None:
            msg = (
                f"No max input tokens found for model {self.config.name!r}. "
                "If you are using a local model, you can set `max_input_token` in the model config to override this."
            )
            self.logger.warning(msg)
        elif input_tokens > self.model_max_input_tokens > 0:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)

        # Build Azure request arguments                                   #
        azure_kwargs: dict[str, Any] = dict(
            model=self._deployment_name,
            messages=messages_no_cache_control,
            n=n or 1,
        )
        
        # Only set temperature and top_p for models that support them
        if self.config.name not in self.NOT_TEMPERATURE_MODELS:
            azure_kwargs["temperature"] = self.config.temperature if temperature is None else temperature
            azure_kwargs["top_p"] = self.config.top_p
        
        if self.tools.use_function_calling:
            azure_kwargs["tools"] = self.tools.tools

        # Call Azure OpenAI & basic error handling                        #
        try:
            response = self._azure_client.chat.completions.create(**azure_kwargs)  # type: ignore
        except openai.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise
        except openai.RateLimitError as e:
            # Let this bubble up for retry handling
            raise
        except openai.OpenAIError:
            raise

        # Convert response → SWE-agent format                             #
        outputs: list[dict] = []
        for choice in response.choices:  # type: ignore[attr-defined]
            out: dict[str, Any] = {"message": choice.message.content or ""}
            if self.tools.use_function_calling and getattr(choice.message, "tool_calls", None):
                out["tool_calls"] = [tc.model_dump() for tc in choice.message.tool_calls]  # type: ignore
            outputs.append(out)

        # Prefer server-reported token usage
        if getattr(response, "usage", None) is not None and getattr(response.usage, "completion_tokens", None) is not None:
            output_tokens = int(response.usage.completion_tokens or 0)
        else:
            # Fallback: approximate with litellm token counter
            output_tokens = sum(
                litellm.utils.token_counter(text=o["message"], model=self.config.name) for o in outputs
            )

        # NOTE: pricing for TRAPI models is unknown → record zero cost
        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=0.0)
        return outputs

    def query(self, history: History, n: int = 1, temperature: float | None = None) -> list[dict] | dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            if exception:
                self.logger.warning(
                    f"Retrying Azure query (attempt {retry_state.attempt_number}) due to {exception.__class__.__name__}: {exception}"
                )

        # Custom retry loop for Azure-specific errors
        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(
                min=self.config.retry.min_wait, max=self.config.retry.max_wait
            ),
            reraise=True,
            retry=retry_if_not_exception_type((
                ContextWindowExceededError,
                CostLimitExceededError,
                ModelConfigurationError,
                openai.AuthenticationError,
                openai.BadRequestError,  # retry on RateLimitError, but NOT on these
                KeyboardInterrupt,
            )),
            before_sleep=retry_warning,
        ):
            with attempt:
                outputs = self._single_query(messages, n=n, temperature=temperature)

        return outputs if n > 1 else outputs[0]


class CopilotClaudeModel(LiteLLMModel):
    """
    GitHub Copilot Claude API implementation.
    Uses OpenAI client format but connects to GitHub Copilot Claude endpoints.
    """

    COPILOT_CLAUDE_SUPPORTED_MODELS = ["claude-sonnet-4"]

    def __init__(self, args: CopilotClaudeModelConfig, tools: ToolConfig):
        if args.name not in self.COPILOT_CLAUDE_SUPPORTED_MODELS:
            msg = f"{args.name} not in supported Copilot Claude models {self.COPILOT_CLAUDE_SUPPORTED_MODELS}"
            raise ValueError(msg)
        super().__init__(args, tools)
        
        self.config: CopilotClaudeModelConfig = args
        self._client = None
        self._token_cache = None
        self._token_expires_at = 0

    def create_request_hmac(self, hmac_secret: str) -> str | None:
        """Create HMAC for request authentication"""
        if not hmac_secret:
            return None
        current = str(int(time.time()))
        signature = hmac.new(
            hmac_secret.encode("utf-8"), current.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{current}.{signature}"

    def fetch_token(self) -> str:
        """Fetch GitHub Copilot token using Node.js script"""
        # Cache token for 30 minutes to avoid frequent fetches
        if self._token_cache and time.time() < self._token_expires_at:
            return self._token_cache

        try:
            # Get the vscode-copilot directory path
            vscode_copilot_dir = (
                self.config.vscode_copilot_dir or 
                os.environ.get("VSCODE_COPILOT_DIR", os.path.expanduser("~/repo/vscode-copilot"))
            )
            vscode_copilot_dir = os.path.expanduser(vscode_copilot_dir)
            if not os.path.exists(vscode_copilot_dir):
                raise ValueError(f"vscode-copilot directory not found at: {vscode_copilot_dir}. "
                               "Set VSCODE_COPILOT_DIR environment variable or vscode_copilot_dir config to the correct path.")
            
            result = subprocess.run(
                ["npx", "tsx", "src/util/node/fetch-token-standalone.js"],
                capture_output=True,
                text=True,
                cwd=vscode_copilot_dir,  # Run from vscode-copilot directory
            )
            
            if result.returncode != 0:
                error_msg = f"Command failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nSTDOUT: {result.stdout}"
                raise ValueError(error_msg)
            
            token = result.stdout.strip()
            if not token:
                raise ValueError("fetch-token.js returned empty output")
            
            # Cache the token for 30 minutes
            self._token_cache = token
            self._token_expires_at = time.time() + 1800  # 30 minutes
            return token
        except Exception as e:
            raise ValueError(f"Failed to get Copilot token: {e}")

    @property
    def client(self):
        if self._client is None:
            # Get the vscode-copilot directory path
            vscode_copilot_dir = (
                self.config.vscode_copilot_dir or 
                os.environ.get("VSCODE_COPILOT_DIR", os.path.expanduser("~/repo/vscode-copilot"))
            )
            
            env_file_path = os.path.expanduser(os.path.join(vscode_copilot_dir, ".env"))

            # Try loading .env file if HMAC_SECRET not already set
            if not os.environ.get("HMAC_SECRET") and os.path.exists(env_file_path):
                try:
                    load_dotenv(dotenv_path=env_file_path)
                except Exception as e:
                    self.logger.warning("Failed to load .env file: %s", e)

            hmac_secret = os.environ.get("HMAC_SECRET")
            if not hmac_secret:
                raise ValueError(
                    "HMAC_SECRET not found. Please set it in environment variables or in a .env file in the vscode-copilot directory."
                )
            
            bearer_token = self.fetch_token()
            hmac_value = self.create_request_hmac(hmac_secret)
            
            if not hmac_value or not bearer_token:
                raise ValueError("Missing HMAC or Bearer token for GitHub Copilot Claude API")

            # Create OpenAI client with GitHub Copilot endpoint and custom headers
            self._client = OpenAI(
                api_key=bearer_token,
                base_url=self.config.api_base or "https://api.enterprise.githubcopilot.com",
                default_headers={
                    "X-Interaction-Type": "conversation-agent",
                    "OpenAI-Intent": "conversation-agent",
                    "X-GitHub-Api-Version": self.config.api_version or "2025-05-01",
                    "Copilot-Integration-Id": "vscode-chat-dev",
                    "VScode-SessionId": "sweagent-session",
                    "VScode-MachineId": "sweagent-machine",
                    "X-Interaction-Id": str(uuid.uuid4()),
                    "X-Initiator": "agent",
                    "Editor-Version": "sweagent/1.0",
                    "Editor-Plugin-Version": "sweagent/1.0",
                    "Request-Hmac": hmac_value,
                },
                timeout=None,
            )
        return self._client

    def _single_query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        self._sleep()

        messages_no_cache_control = copy.deepcopy(messages)
        for m in messages_no_cache_control:
            if "cache_control" in m:
                del m["cache_control"]

        input_tokens = litellm.utils.token_counter(
            messages=messages_no_cache_control,
            model=self.config.name,
        )
        if self.model_max_input_tokens is None:
            msg = (
                f"No max input tokens found for model {self.config.name!r}. "
                "If you are using a local model, you can set `max_input_token` in the model config to override this."
            )
            self.logger.warning(msg)
        elif input_tokens > self.model_max_input_tokens > 0:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)

        # Build request arguments for GitHub Copilot Claude API
        request_kwargs: dict[str, Any] = dict(
            model=self.config.name,
            messages=messages_no_cache_control,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
        )
        
        if self.config.top_p is not None:
            request_kwargs["top_p"] = self.config.top_p
        
        if self.tools.use_function_calling:
            request_kwargs["tools"] = self.tools.tools
            request_kwargs["tool_choice"] = "auto"

        # Call GitHub Copilot Claude API
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except openai.BadRequestError as e:
            if e.code == "context_length_exceeded" or "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise
        except openai.RateLimitError as e:
            # Let this bubble up for retry handling
            raise
        except openai.OpenAIError:
            raise

        # Convert response to SWE-agent format
        outputs: list[dict] = []
        combined_message = ""
        combined_tool_calls = []
        
        for choice in response.choices:
            if choice.message.content:
                combined_message += choice.message.content
            if self.tools.use_function_calling and getattr(choice.message, "tool_calls", None):
                combined_tool_calls.extend([tc.model_dump() for tc in choice.message.tool_calls])
        
        # Create single output with combined message and tool calls
        out: dict[str, Any] = {"message": combined_message}
        if combined_tool_calls:
            out["tool_calls"] = combined_tool_calls
        outputs.append(out)

        # Use server-reported token usage if available
        if getattr(response, "usage", None) is not None and getattr(response.usage, "completion_tokens", None) is not None:
            output_tokens = int(response.usage.completion_tokens or 0)
        else:
            # Fallback: approximate with litellm token counter
            output_tokens = sum(
                litellm.utils.token_counter(text=o["message"], model=self.config.name) for o in outputs
            )

        # NOTE: GitHub Copilot Claude API pricing may vary → record zero cost for now
        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=0.0)
        return outputs

    def query(self, history: History, n: int = 1, temperature: float | None = None) -> list[dict] | dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            if exception:
                self.logger.warning(
                    f"Retrying Copilot Claude query (attempt {retry_state.attempt_number}) due to {exception.__class__.__name__}: {exception}"
                )
            # Special handling for HMAC timestamp errors - clear client to force new token
            if isinstance(exception, openai.AuthenticationError) and "HMAC timestamp out of range" in str(exception):
                self.logger.info("Refreshing client due to HMAC timestamp error")
                # self._client = None  # Clear client to force recreation with new HMAC

        # Custom retry loop for Copilot Claude API-specific errors
        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(
                min=self.config.retry.min_wait, max=self.config.retry.max_wait
            ),
            reraise=True,
            retry=retry_if_not_exception_type((
                ContextWindowExceededError,
                CostLimitExceededError,
                ModelConfigurationError,
                # Remove openai.AuthenticationError from here to allow retry for HMAC timestamp errors
                openai.BadRequestError,
                KeyboardInterrupt,
            )) | retry_if_exception_message(match="HMAC timestamp out of range"),
            before_sleep=retry_warning,
        ):
            with attempt:
                outputs = self._single_query(messages, n=n, temperature=temperature)

        # To update to merge message and tool calls into a single dict
        return outputs if n > 1 else outputs[0]


def get_model(args: ModelConfig, tools: ToolConfig) -> AbstractModel:
    """Returns correct model object given arguments and commands"""
    # Convert GenericAPIModelConfig to specific model config if needed
    if isinstance(args, GenericAPIModelConfig) and not isinstance(
        args, HumanModelConfig | HumanThoughtModelConfig | ReplayModelConfig | InstantEmptySubmitModelConfig | CopilotClaudeModelConfig
    ):
        if args.name == "human":
            args = HumanModelConfig(**args.model_dump())
        elif args.name == "human_thought":
            args = HumanThoughtModelConfig(**args.model_dump())
        elif args.name == "replay":
            args = ReplayModelConfig(**args.model_dump())
        elif args.name == "instant_empty_submit":
            args = InstantEmptySubmitModelConfig(**args.model_dump())
        elif args.name in CopilotClaudeModel.COPILOT_CLAUDE_SUPPORTED_MODELS:
            args = CopilotClaudeModelConfig(**args.model_dump())

    if args.name == "human":
        assert isinstance(args, HumanModelConfig), f"Expected {HumanModelConfig}, got {args}"
        return HumanModel(args, tools)
    if args.name == "human_thought":
        assert isinstance(args, HumanThoughtModelConfig), f"Expected {HumanThoughtModelConfig}, got {args}"
        return HumanThoughtModel(args, tools)
    if args.name == "replay":
        assert isinstance(args, ReplayModelConfig), f"Expected {ReplayModelConfig}, got {args}"
        return ReplayModel(args, tools)
    elif args.name == "instant_empty_submit":
        assert isinstance(args, InstantEmptySubmitModelConfig), f"Expected {InstantEmptySubmitModelConfig}, got {args}"
        return InstantEmptySubmitTestModel(args, tools)
    if isinstance(args, CopilotClaudeModelConfig):
        return CopilotClaudeModel(args, tools)
    if isinstance(args, GenericAPIModelConfig) and args.name in AzureLLMModel.AZURE_SUPPORTED_MODELS:
        return AzureLLMModel(args, tools)
    assert isinstance(args, GenericAPIModelConfig), f"Expected {GenericAPIModelConfig}, got {args}"
    return LiteLLMModel(args, tools)
    assert isinstance(args, GenericAPIModelConfig), f"Expected {GenericAPIModelConfig}, got {args}"
    return LiteLLMModel(args, tools)
