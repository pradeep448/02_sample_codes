# agent_app.py
import os
from typing import Optional, Dict, Any, List

from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain.memory import ConversationBufferMemory

# Optional: web search tool
from duckduckgo_search import DDGS

# -------- Tools --------
@tool("duck_search", return_direct=False)
def duck_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return top results as text."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    lines = []
    for h in hits:
        title = h.get("title", "")
        body = h.get("body", "")
        href = h.get("href", "")
        lines.append(f"- {title} :: {href}\n{body}")
    return "\n".join(lines)

@tool("python_repl", return_direct=False)
def python_repl(code: str) -> str:
    """Execute Python code in a restricted REPL. Return stdout or error."""
    import contextlib, io, sys
    buf = io.StringIO()
    local_vars = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "min": min, "max": max, "sum": sum}}, local_vars)
    except Exception as e:
        return f"ERROR: {e}"
    return buf.getvalue() or "OK"

TOOLS = [duck_search, python_repl]

# -------- LLM --------
def get_llm(model: str = "llama3.2", temperature: float = 0.2):
    return Ollama(model=model, temperature=temperature)

# -------- Prompt (ReAct) --------
SYSTEM = """You are a helpful, cautious agent. Think step-by-step.
You can use tools. If a tool is not needed, answer directly.
If you are unsure, say you don't know.
Return concise, correct answers with sources when using search."""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{input}"),
        # Tool-usage scratchpad is automatically appended by the agent runtime.
    ]
)

# -------- Memory --------
def get_memory():
    return ConversationBufferMemory(memory_key="history", return_messages=True)

# -------- Build Agent --------
def build_agent():
    llm = get_llm()
    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )
    memory = get_memory()
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,          # prints ReAct traces
        handle_parsing_errors=True,
        max_iterations=6,      # safety
        return_intermediate_steps=True,
    )
    return executor

# -------- Run --------
if __name__ == "__main__":
    agent = build_agent()

    # Example 1: Direct reasoning (no tools)
    res1 = agent.invoke({"input": "Summarize the Pythagorean theorem in one sentence."})
    print("\nAnswer 1:\n", res1["output"])

    # Example 2: Use search tool
    res2 = agent.invoke({"input": "What is the latest stable Python version? Cite a source."})
    print("\nAnswer 2:\n", res2["output"])

    # Example 3: Use Python REPL tool
    res3 = agent.invoke({"input": "Compute the mean and std of [2,5,9,4]. Use the python_repl tool."})
    print("\nAnswer 3:\n", res3["output"])
