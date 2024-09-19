import json
import uuid
from abc import ABC
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import SystemMessagePromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""  # noqa: E501

DEFAULT_RESPONSE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "__conversational_response",
        "description": (
            "Respond conversationally if no other tools should be called for a given query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "Conversational response to the user.",
                },
            },
            "required": ["response"],
        },
    },
}

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydantic = Union[Dict, _BM]


def _is_pydantic_class(obj: Any) -> bool:
    """
    Checks if the tool provided is a Pydantic class.
    """
    return isinstance(obj, type) and (
            issubclass(obj, BaseModel) or BaseModel in obj.__bases__
    )


def _is_pydantic_object(obj: Any) -> bool:
    """
    Checks if the tool provided is a Pydantic object.
    """
    return isinstance(obj, BaseModel)


class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


def parse_json_garbage(s: str) -> Any:
    """
    Parse a JSON-like string and return it as a Python object.
    Parsing begins at the first occurrence of "{" or "[".
    """
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        response = json.loads(s)
        return response
    except (json.JSONDecodeError, ValueError) as e:
        if isinstance(e, json.JSONDecodeError):
            response = json.loads(s[: e.pos])
            return response
        raise e


def parse_response(message: BaseMessage) -> str:
    """Extract `function_call` from `AIMessage`."""
    if isinstance(message, AIMessage):
        kwargs = message.additional_kwargs
        tool_calls = message.tool_calls
        if len(tool_calls) > 0:
            tool_call = tool_calls[-1]
            args = tool_call.get("args")
            return json.dumps(args)
        elif "function_call" in kwargs:
            if "arguments" in kwargs["function_call"]:
                return kwargs["function_call"]["arguments"]
            raise ValueError(
                f"`arguments` missing from `function_call` within AIMessage: {message}"
            )
        else:
            raise ValueError("`tool_calls` missing from AIMessage: {message}")
    raise ValueError(f"`message` is not an instance of `AIMessage`: {message}")


class ToolCallingLLM(BaseChatModel, ABC):
    """ToolCallingLLM mixin to enable tool calling features on non tool calling models.

    Note: This is an incomplete mixin and should not be used directly. It must be used to extent an existing Chat Model.

    Setup:
      Install dependencies for your Chat Model.
      Any API Keys or setup needed for your Chat Model is still applicable.

    Key init args — completion params:
      Refer to the documentation of the Chat Model you wish to extend with Tool Calling.

    Key init args — client params:
      Refer to the documentation of the Chat Model you wish to extend with Tool Calling.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
      ```
      # Example implementation using LiteLLM
      from langchain_community.chat_models import ChatLiteLLM

      class LiteLLMFunctions(ToolCallingLLM, ChatLiteLLM):

          def __init__(self, **kwargs: Any) -> None:
              super().__init__(**kwargs)

          @property
          def _llm_type(self) -> str:
              return "litellm_functions"

      llm = LiteLLMFunctions(model="ollama/phi3")
      ```

    Invoke:
      ```
      messages = [
        ("human", "What is the capital of France?")
      ]
      llm.invoke(messages)
      ```
      ```
      AIMessage(content='The capital of France is Paris.', id='run-497d0e1a-d63b-45e8-9c8b-5e76d99b9468-0')
      ```

    Tool calling:
      ```
      from langchain_core.pydantic_v1 import BaseModel, Field

      class GetWeather(BaseModel):
          '''Get the current weather in a given location'''

          location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

      class GetPopulation(BaseModel):
          '''Get the current population in a given location'''

          location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

      llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
      ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
      ai_msg.tool_calls
      ```
      ```
      [{'name': 'GetWeather', 'args': {'location': 'Austin, TX'}, 'id': 'call_25ed526917b94d8fa5db3fe30a8cf3c0'}]
      ```

    Structured output:
      ```
      from typing import Optional

      from langchain_core.pydantic_v1 import BaseModel, Field

      class Joke(BaseModel):
          '''Joke to tell user.'''

          setup: str = Field(description="The setup of the joke")
          punchline: str = Field(description="The punchline to the joke")
          rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

      structured_llm = llm.with_structured_output(Joke)
      structured_llm.invoke("Tell me a joke about cats")
      ```
      ```
      Joke(setup='Why was the cat sitting on the computer?', punchline='Because it wanted to be online!', rating=7)
      ```
      See `ToolCallingLLM.with_structured_output()` for more.

    Response metadata
      Refer to the documentation of the Chat Model you wish to extend with Tool Calling.

    """  # noqa: E501

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE

    override_bind_tools: bool = True

    def __init__(self, **kwargs: Any) -> None:
        override_bind_tools = True
        if "override_bind_tools" in kwargs:
            override_bind_tools = kwargs["override_bind_tools"]
            del kwargs["override_bind_tools"]
        super().__init__(**kwargs)
        self.override_bind_tools = override_bind_tools

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if self.override_bind_tools:
            return self.bind(functions=tools, **kwargs)
        else:
            return super().bind_tools(tools, **kwargs)

    def _generate_system_message_and_functions(
            self,
            kwargs: Dict[str, Any],
    ) -> Tuple[BaseMessage, List]:
        functions = kwargs.get("tools", kwargs.get("functions", []))
        functions = [
            fn["function"]
            if (not _is_pydantic_class(fn) and
                not _is_pydantic_object(fn) and
                "name" not in fn.keys() and
                "function" in fn.keys() and
                "name" in fn["function"].keys())
            else fn
            for fn in functions]
        if "functions" in kwargs:
            del kwargs["functions"]
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    "If `function_call` is specified, you must also pass a "
                    "matching function in `functions`."
                )
            del kwargs["function_call"]
        functions = [convert_to_openai_tool(fn) for fn in functions]
        functions.append(DEFAULT_RESPONSE_FUNCTION)
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        return system_message, functions

    def _process_response(
            self, response_message: BaseMessage, functions: List[Dict]
    ) -> AIMessage:
        chat_generation_content = response_message.content
        if not isinstance(chat_generation_content, str):
            raise ValueError("ToolCallingLLM does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
        except json.JSONDecodeError:
            try:
                parsed_chat_result = parse_json_garbage(chat_generation_content)
            except json.JSONDecodeError:
                raise ValueError(
                    f"'{self.model}' did not respond with valid JSON.\n"  # type: ignore[attr-defined]
                    "Please try again.\n"
                    f"Response: {chat_generation_content}"
                )
        called_tool_name = (
            parsed_chat_result["tool"] if "tool" in parsed_chat_result else None
        )
        called_tool = next(
            (fn for fn in functions if fn["function"]["name"] == called_tool_name), None
        )
        if (
                called_tool is None
                or called_tool["function"]["name"] == DEFAULT_RESPONSE_FUNCTION["function"]["name"]
        ):
            if (
                    "tool_input" in parsed_chat_result
                    and "response" in parsed_chat_result["tool_input"]
            ):
                response = parsed_chat_result["tool_input"]["response"]
            elif "response" in parsed_chat_result:
                response = parsed_chat_result["response"]
            else:
                raise ValueError(
                    f"Failed to parse a response from {self.model} output: "  # type: ignore[attr-defined]
                    f"{chat_generation_content}"
                )
            return AIMessage(content=response)

        called_tool_arguments = (
            parsed_chat_result["tool_input"]
            if "tool_input" in parsed_chat_result
            else {}
        )

        response_message_with_functions = AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name=called_tool_name,
                    args=called_tool_arguments if called_tool_arguments else {},
                    id=f"call_{str(uuid.uuid4()).replace('-', '')}",
                )
            ],
        )

        return response_message_with_functions

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        response_message = super()._generate(  # type: ignore[safe-super]
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        response = self._process_response(
            response_message.generations[0].message, functions
        )
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        response_message = await super()._agenerate(
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        response = self._process_response(
            response_message.generations[0].message, functions
        )
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def astream(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        generation: Optional[BaseMessageChunk] = None
        async for chunk in super().astream(
                [system_message] + super()._convert_input(input).to_messages(),
                stop=stop,
                **kwargs,
        ):
            if generation is None:
                generation = chunk
            else:
                generation += chunk
        assert generation is not None
        response = self._process_response(generation, functions)
        yield cast(BaseMessageChunk, response)
