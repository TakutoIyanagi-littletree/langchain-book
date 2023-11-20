from langchain.agents import load_tools

tools = load_tools(["ddg-search", "wikipedia"])

print(tools)
