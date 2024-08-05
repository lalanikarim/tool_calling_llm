Tool Calling LLM
================

Tool Calling LLM is a python mixin that lets you add tool calling capabilities effortlessly to [LangChain](https://langchain.com)'s Chat Models that don't yet support tool/function calling natively. Simply create a new chat model class with ToolCallingLLM and your favorite chat model to get started.

With ToolCallingLLM you also get access to the following functions:
1. `.bind_tools()` allows you to bind tool definitions with a llm.
2. `.with_structured_output()` allows you to return structured data from your model.

At this time, ToolCallingLLM has been tested to work with ChatOllama, ChatNVIDIA, and ChatLiteLLM with Ollama provider.

The [OllamaFunctions](https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/) was the original inspiration for this effort. The code for ToolCallingLLM was abstracted out of `OllamaFunctions` to allow it to be reused with other non tool calling Chat Models.

Installation
------------

```bash
pip install --upgrade tool_calling_llm
```

Usage
-----

Creating a Tool Calling LLM is as simple as creating a new sub class of the original ChatModel you wish to add tool calling features to.  

Below sample code demonstrates how you might enhance `ChatOllama` chat model from `langchain-ollama` package with tool calling capabilities.

```python
from tool_calling_llm import ToolCallingLLM
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun


class OllamaWithTools(ToolCallingLLM, ChatOllama):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self):
        return "ollama_with_tools"


llm = OllamaWithTools(model="llama3.1",format="json")
tools = [DuckDuckGoSearchRun()]
llm_tools = llm.bind_tools(tools=tools)

llm_tools.invoke("Who won the silver medal in shooting in the Paris Olympics in 2024?")
```

This yields output as follows:
```
AIMessage(content='', id='run-9c3c7a78-97af-4d06-835e-aa81174fd7e8-0', tool_calls=[{'name': 'duckduckgo_search', 'args': {'query': 'Paris Olympics 2024 shooting silver medal winner'}, 'id': 'call_67b06088e208482497f6f8314e0f1a0e', 'type': 'tool_call'}])
```
For more comprehensive examples, refer to [ToolCallingLLM-Tutorial.ipynb](ToolCallingLLM-Tutorial.ipynb) jupyter notebook.
