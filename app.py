"""
Gresham House Agentic AI Demo
FastAPI + LangGraph + Groq + FastMCP
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

from fastmcp import FastMCP

# App initialization
app = FastAPI(title="Gresham House Agentic AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FastMCP
mcp = FastMCP("Gresham House Agent")

# ============== LLM CONFIGURATION ==============

def get_llm():
    """Get Groq LLM"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    return ChatGroq(
        model="llama-3.2-90b-text-preview",
        temperature=0,
        api_key=api_key
    )

# ============== AGENT STATE ==============

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    query: str
    use_case: str
    final_response: str

# ============== TOOLS ==============

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query[:500])[:2000]
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """Read content from a file in the data directory."""
    try:
        safe_path = Path("data") / file_path.replace("..", "")
        if safe_path.exists():
            return safe_path.read_text()[:3000]
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Read error: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the data directory."""
    try:
        safe_path = Path("data") / file_path.replace("..", "")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Write error: {str(e)}"

@tool
def query_warehouse(sql_query: str) -> str:
    """Query the SQLite data warehouse. SELECT statements only."""
    try:
        if not sql_query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."
        conn = sqlite3.connect("data/gresham_demo.db")
        cursor = conn.cursor()
        cursor.execute(sql_query[:1000])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        results = [dict(zip(columns, row)) for row in rows[:50]]
        return f"Found {len(results)} rows:\n" + str(results)
    except Exception as e:
        return f"Query error: {str(e)}"

@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    try:
        safe_dir = Path("data") / directory.replace("..", "")
        if safe_dir.exists():
            files = [str(f) for f in safe_dir.iterdir()]
            return f"Files found:\n" + "\n".join(files[:20])
        return f"Directory not found: {directory}"
    except Exception as e:
        return f"List error: {str(e)}"

@tool
def get_schema_info() -> str:
    """Get information about available database tables."""
    try:
        conn = sqlite3.connect("data/gresham_demo.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        schema_info = []
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            schema_info.append(f"Table: {table_name}")
            schema_info.append(f"Columns: {[col[1] for col in columns]}")
            schema_info.append("---")
        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        return f"Schema error: {str(e)}"

# ============== LANGGRAPH AGENT ==============

def create_agent():
    """Create the LangGraph ReAct agent."""
    llm = get_llm()
    tools = [search_web, read_file, write_file, query_warehouse, list_files, get_schema_info]
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            messages = [{"role": "user", "content": state.get("query", "")}]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": messages + [{"role": "assistant", "content": response.content if hasattr(response, 'content') else str(response)}],
            "final_response": response.content if hasattr(response, 'content') else str(response)
        }
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ============== MCP SERVER TOOLS ==============

@mcp.tool()
async def mcp_search_web(query: str) -> str:
    return search_web.invoke(query)

@mcp.tool()
async def mcp_read_file(file_path: str) -> str:
    return read_file.invoke(file_path)

@mcp.tool()
async def mcp_write_file(file_path: str, content: str) -> str:
    return write_file.invoke(file_path, content)

@mcp.tool()
async def mcp_query_warehouse(sql_query: str) -> str:
    return query_warehouse.invoke(sql_query)

@mcp.tool()
async def mcp_list_files(directory: str = ".") -> str:
    return list_files.invoke(directory)

@mcp.tool()
async def mcp_get_schema_info() -> str:
    return get_schema_info.invoke()

# ============== FASTAPI ENDPOINTS ==============

class QueryRequest(BaseModel):
    query: str
    use_case: Optional[str] = "Hybrid"

class QueryResponse(BaseModel):
    response: str
    use_case: str
    tools_available: List[str]

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gresham House Agent</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
            .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 20px 0; }
            input, button { padding: 10px; margin: 5px 0; }
            button { background: #007bff; color: white; border: none; cursor: pointer; }
            #response { background: #f5f5f5; padding: 15px; border-radius: 4px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>🏠 Gresham House Agentic AI Demo</h1>
        <div class="card">
            <h3>Agent Interface</h3>
            <input type="text" id="query" placeholder="Ask the agent..." style="width: 100%;">
            <button onclick="submit()">Send</button>
            <div id="response" style="margin-top: 20px;">Response will appear here...</div>
        </div>
        <div class="card">
            <h3>MCP Endpoint</h3>
            <p>Connect Claude Desktop to: <code id="mcp-url"></code></p>
        </div>
        <script>
            document.getElementById('mcp-url').textContent = window.location.origin + '/mcp/';
            async function submit() {
                const query = document.getElementById('query').value;
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                });
                const data = await response.json();
                document.getElementById('response').textContent = data.response;
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/query")
async def api_query(request: QueryRequest):
    try:
        agent = create_agent()
        initial_state = {
            "messages": [],
            "query": request.query,
            "use_case": request.use_case,
            "final_response": ""
        }
        config = {"configurable": {"thread_id": "default"}}
        result = agent.invoke(initial_state, config)
        return QueryResponse(
            response=result.get("final_response", "No response"),
            use_case=request.use_case,
            tools_available=["search_web", "read_file", "write_file", "query_warehouse", "list_files", "get_schema_info"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Mount MCP server
mcp.mount(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
