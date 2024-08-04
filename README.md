Tool Calling LLM
================

Tool Calling LLM is a python mixin that lets you add tool calling capabilities effortlessly to LangChain Chat Models that don't yet support tool/function calling natively. Simply create a new chat model class with ToolCallingLLM and your favorite chat model to get started.

With ToolCallingLLM you also get access to .with_structured_output() which will allow you to return structured data from your model.

At this time, ToolCallingLLM has been tested to work with ChatNVIDIA and ChatLiteLLM with Ollama provider.

The OllamaFunctions was the original inspiration for this effort. The code for ToolCallingLLM was abstracted out of OllamaFunctions to allow it to be reused with other non tool calling Chat Models.
