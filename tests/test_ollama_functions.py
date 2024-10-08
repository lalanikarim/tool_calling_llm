"""Test OllamaFunctions"""

import unittest
from typing import Any

import pytest
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.utils.function_calling import convert_to_openai_tool

from tool_calling_llm import ToolCallingLLM


class OllamaFunctions(ToolCallingLLM, ChatOllama):  # type: ignore[misc]
    """Function chat model that uses ChatLiteLLM."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama_functions"


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


@pytest.fixture(scope="function")
def base_model(request):
    request.cls.base_model = OllamaFunctions(model="phi3")


@pytest.mark.usefixtures("base_model")
class TestOllamaFunctions(unittest.TestCase):
    """
    Test OllamaFunctions
    """

    def test_default_ollama_functions(self) -> None:
        # bind functions
        model = self.base_model.bind_tools(
            tools=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                               "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call={"name": "get_current_weather"},
        )

        res = model.invoke("What's the weather in San Francisco?")

        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call
        self.assertEqual("get_current_weather", tool_call.get("name"))

    def test_ollama_functions_tools(self) -> None:
        model = self.base_model.bind_tools(
            tools=[PubmedQueryRun(), DuckDuckGoSearchResults(max_results=2)]  # type: ignore[call-arg]
        )
        res = model.invoke("What causes lung cancer?")
        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call
        self.assertEqual("pub_med", tool_call.get("name"))

    def test_default_ollama_functions_default_response(self) -> None:
        # bind functions
        model = self.base_model.bind_tools(
            tools=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                               "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ]
        )

        res = model.invoke("What is the capital of France?")

        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            assert tool_call
            self.assertEqual("__conversational_response", tool_call.get("name"))

    def test_ollama_structured_output(self) -> None:
        structured_llm = self.base_model.with_structured_output(Joke, include_raw=False)

        res = structured_llm.invoke("Tell me a joke about cats")
        assert isinstance(res, Joke)

    def test_ollama_structured_output_with_json(self) -> None:
        joke_schema = convert_to_openai_tool(Joke)
        structured_llm = self.base_model.with_structured_output(joke_schema, include_raw=False)

        res = structured_llm.invoke("Tell me a joke about cats")
        assert "setup" in res
        assert "punchline" in res

    def test_ollama_structured_output_raw(self) -> None:
        structured_llm = self.base_model.with_structured_output(Joke, include_raw=True)

        res = structured_llm.invoke("Tell me a joke about cars")
        assert isinstance(res, dict)
        assert "raw" in res
        assert "parsed" in res
        assert isinstance(res["raw"], AIMessage)
        assert isinstance(res["parsed"], Joke)


@pytest.mark.usefixtures("base_model")
class TestOllamaFunctionsAsync(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.asyncio
    async def test_ollama_structured_output_async(self) -> None:
        structured_llm = self.base_model.with_structured_output(Joke, include_raw=False)

        res = await structured_llm.ainvoke("Tell me a joke about cats")
        assert isinstance(res, Joke)

    @pytest.mark.asyncio
    async def test_ollama_structured_output_astream(self) -> None:
        structured_llm = self.base_model.with_structured_output(Joke, include_raw=False)

        async for output in structured_llm.astream("Tell me a joke about cats"):
            assert isinstance(output, Joke)
