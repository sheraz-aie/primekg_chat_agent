# api_server.py
import asyncio
import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent
from dotenv import load_dotenv
from pydantic import SecretStr
import os

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp/"

# --- FastAPI App ---
app = FastAPI(
    title="Neo4j Query API",
    description="API for querying Neo4j database using natural language",
    version="1.0.0"
)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    question: str
    cypher_query: str
    result: Dict[str, Any]  # Always a dictionary, lists are wrapped in {"data": [...]}
    success: bool
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Which diseases does Quetiapine treat?",
                "cypher_query": "MATCH (d:Drug {name: 'Quetiapine'})-[:TREATS]->(disease) RETURN disease.name AS DiseaseName",
                "result": {
                    "data": [
                        {"DiseaseName": "bipolar disorder"},
                        {"DiseaseName": "schizophrenia"}
                    ],
                    "count": 2,
                    "type": "list"
                },
                "success": True,
                "message": "Query executed successfully"
            }
        }

# --- Neo4j Query Service ---
class Neo4jQueryService:
    def __init__(self):
        self.mcp_server_url = MCP_SERVER_URL
        self.openai_api_key = OPENAI_API_KEY
        
    async def query_neo4j(self, question: str) -> QueryResponse:
        """
        Process a natural language question and return Neo4j query results.
        """
        try:
            # Create the MCP client using FastMCP with StreamableHttpTransport.
            transport = StreamableHttpTransport(url=self.mcp_server_url)
            client = Client(transport=transport)
            
            # The async context manager ensures the client is properly connected and closed.
            async with client:
                print(f"Client connected: {client.is_connected()}")
                
                # Test the connection and list available tools first
                try:
                    available_tools = await client.list_tools()
                    tool_names = [tool.name for tool in available_tools]
                    print(f"Successfully connected to MCP server. Available tools: {tool_names}")
                except Exception as conn_error:
                    error_msg = f"Failed to connect to MCP server at {self.mcp_server_url}: {str(conn_error)}"
                    print(error_msg)
                    return QueryResponse(
                        question=question,
                        cypher_query="",
                        result={"error": error_msg},
                        success=False,
                        message=error_msg
                    )
                
                # --- Define Tool-Calling Functions within the client context ---
                async def get_schema(_input) -> str:
                    """
                    Invokes the 'get_neo4j_schema' tool and returns the schema as a string.
                    """
                    print("Chain step: Fetching schema...")
                    try:
                        # First, let's check what tools are available
                        available_tools = await client.list_tools()
                        tool_names = [tool.name for tool in available_tools]
                        print(f"Available tools: {tool_names}")
                        
                        # Try different possible tool names
                        schema_tool_name = None
                        possible_names = ["get-neo4j-schema", "get_neo4j_schema", "get-schema", "schema"]
                        for name in possible_names:
                            if name in tool_names:
                                schema_tool_name = name
                                break
                        
                        if not schema_tool_name:
                            print(f"Warning: No schema tool found. Available tools: {tool_names}")
                            return f"Schema tool not available. Available tools: {', '.join(tool_names)}"
                        
                        print(f"Using schema tool: {schema_tool_name}")
                        schema_result = await client.call_tool(schema_tool_name, {})
                        
                        if schema_result and len(schema_result) > 0:
                            if isinstance(schema_result[0], TextContent):
                                return schema_result[0].text
                            else:
                                print(f"Schema result type: {type(schema_result[0])}")
                                return str(schema_result[0])
                        
                        return "No schema available - empty result"
                        
                    except Exception as e:
                        print(f"Error fetching schema: {str(e)}")
                        return f"Error fetching schema: {str(e)}"

                async def run_cypher(query: str) -> str:
                    """
                    Takes a Cypher query string and executes it using the 'read_neo4j_cypher' tool.
                    """
                    print(f"Chain step: Executing Cypher -> {query.strip()}")
                    try:
                        # Get available tools again to check cypher tool name
                        available_tools = await client.list_tools()
                        tool_names = [tool.name for tool in available_tools]
                        
                        # Try different possible tool names for cypher execution
                        cypher_tool_name = None
                        possible_names = ["read-neo4j-cypher", "read_neo4j_cypher", "execute-cypher", "cypher"]
                        for name in possible_names:
                            if name in tool_names:
                                cypher_tool_name = name
                                break
                        
                        if not cypher_tool_name:
                            print(f"Warning: No cypher tool found. Available tools: {tool_names}")
                            return json.dumps({
                                "error": "Cypher execution tool not available", 
                                "available_tools": tool_names,
                                "success": False
                            })
                        
                        print(f"Using cypher tool: {cypher_tool_name}")
                        result = await client.call_tool(cypher_tool_name, {"query": query})
                        
                        if result and len(result) > 0:
                            if isinstance(result[0], TextContent):
                                result_text = result[0].text
                                # Try to parse as JSON to validate format
                                try:
                                    json.loads(result_text)
                                    return result_text
                                except json.JSONDecodeError:
                                    # If not valid JSON, wrap in a JSON structure
                                    return json.dumps({"raw_result": result_text, "type": "text"})
                            else:
                                print(f"Cypher result type: {type(result[0])}")
                                return json.dumps({"result": str(result[0]), "type": str(type(result[0]).__name__)})
                        
                        return json.dumps([]) # Return empty JSON array on failure
                        
                    except Exception as e:
                        print(f"Error executing cypher: {str(e)}")
                        return json.dumps({"error": f"Error executing cypher: {str(e)}", "success": False})

                # --- Define the LangChain Components ---
                chat_llm = ChatOpenAI(
                    model='gpt-4.1-mini', 
                    api_key=SecretStr(self.openai_api_key) if self.openai_api_key else None, 
                    temperature=0.0
                )

                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """You are an expert Neo4j developer. Your task is to write a Cypher query based on a provided schema and a user's question.
You must only return the Cypher query itself, with no explanation, preamble, or markdown.

Here is the database schema:
{schema}""",
                        ),
                        ("human", "{question}"),
                    ]
                )
                
                # Store the generated query for response
                generated_query = ""
                
                async def capture_and_run_cypher(query: str) -> str:
                    """
                    Captures the generated query and executes it.
                    """
                    nonlocal generated_query
                    generated_query = query.strip()
                    return await run_cypher(query)
                
                # --- Construct the Unified LCEL Chain ---
                full_chain = (
                    {
                        "schema": RunnableLambda(get_schema),
                        "question": RunnablePassthrough(),
                    }
                    | prompt_template
                    | chat_llm
                    | StrOutputParser()
                    | RunnableLambda(capture_and_run_cypher)
                )

                # --- Invoke the Chain ---
                print(f"Executing chain for question: '{question}'")
                final_result = await full_chain.ainvoke(question)

                # Parse the result as JSON and ensure it's a dictionary
                try:
                    parsed_result = json.loads(final_result) if final_result else []
                    
                    # If the result is a list, wrap it in a dictionary
                    if isinstance(parsed_result, list):
                        result_data = {
                            "data": parsed_result,
                            "count": len(parsed_result),
                            "type": "list"
                        }
                    elif isinstance(parsed_result, dict):
                        result_data = parsed_result
                    else:
                        result_data = {"value": parsed_result, "type": str(type(parsed_result).__name__)}
                        
                except json.JSONDecodeError as json_error:
                    # If JSON parsing fails, treat as raw text
                    result_data = {
                        "raw_result": final_result,
                        "parse_error": str(json_error),
                        "type": "raw_text"
                    }

                # Determine if the query was successful based on the result
                success = True
                message = "Query executed successfully"
                
                # Check if there were any errors in the result
                if isinstance(result_data, dict):
                    if "error" in result_data:
                        success = False
                        message = f"Query execution failed: {result_data.get('error', 'Unknown error')}"
                    elif result_data.get("success") == False:
                        success = False
                        message = f"Query execution failed: {result_data.get('error', 'Unknown error')}"

                return QueryResponse(
                    question=question,
                    cypher_query=generated_query,
                    result=result_data,
                    success=success,
                    message=message
                )
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return QueryResponse(
                question=question,
                cypher_query="",
                result={},
                success=False,
                message=f"Error: {str(e)}"
            )

# --- Initialize Service ---
query_service = Neo4jQueryService()

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Neo4j Query API",
        "description": "Use /query endpoint to ask questions about the Neo4j database",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed diagnostics."""
    try:
        # Test connection to MCP server
        transport = StreamableHttpTransport(url=MCP_SERVER_URL)
        client = Client(transport=transport)
        
        async with client:
            connected = client.is_connected()
            
            if connected:
                try:
                    # Test tool listing
                    tools = await client.list_tools()
                    tool_names = [tool.name for tool in tools]
                    
                    # Test schema tool specifically
                    schema_available = any(name in ["get-neo4j-schema", "get_neo4j_schema", "get-schema", "schema"] 
                                         for name in tool_names)
                    cypher_available = any(name in ["read-neo4j-cypher", "read_neo4j_cypher", "execute-cypher", "cypher"] 
                                         for name in tool_names)
                    
                    return {
                        "status": "healthy",
                        "mcp_server": "connected",
                        "mcp_server_url": MCP_SERVER_URL,
                        "available_tools": tool_names,
                        "schema_tool_available": schema_available,
                        "cypher_tool_available": cypher_available,
                        "tools_count": len(tools)
                    }
                except Exception as tool_error:
                    return {
                        "status": "partial",
                        "mcp_server": "connected_but_tools_unavailable",
                        "mcp_server_url": MCP_SERVER_URL,
                        "error": f"Connected but failed to list tools: {str(tool_error)}"
                    }
            else:
                return {
                    "status": "unhealthy",
                    "mcp_server": "disconnected",
                    "mcp_server_url": MCP_SERVER_URL
                }
                
    except Exception as e:
        return {
            "status": "unhealthy",
            "mcp_server": "error",
            "mcp_server_url": MCP_SERVER_URL,
            "error": str(e)
        }

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """
    Query the Neo4j database using natural language.
    
    Args:
        request: QueryRequest containing the natural language question
        
    Returns:
        QueryResponse with the generated Cypher query and results
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = await query_service.query_neo4j(request.question)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example queries that can be used with the API."""
    return {
        "examples": [
            {
                "question": "Which diseases does Quetiapine treat?",
                "description": "Find diseases treated by a specific drug"
            },
            {
                "question": "What are the side effects of Aspirin?",
                "description": "Find side effects of a specific medication"
            },
            {
                "question": "Show me drugs that treat depression",
                "description": "Find drugs for a specific condition"
            },
            {
                "question": "What proteins does the drug Metformin interact with?",
                "description": "Find protein interactions for a drug"
            }
        ]
    }

@app.get("/debug/mcp")
async def debug_mcp_connection():
    """Debug endpoint to test MCP server connection and tools."""
    debug_info = {
        "mcp_server_url": MCP_SERVER_URL,
        "connection_test": None,
        "tools_test": None,
        "schema_test": None,
        "error_details": None
    }
    
    try:
        transport = StreamableHttpTransport(url=MCP_SERVER_URL)
        client = Client(transport=transport)
        
        async with client:
            # Test 1: Connection
            connected = client.is_connected()
            debug_info["connection_test"] = {
                "connected": connected,
                "status": "✅ Connected" if connected else "❌ Not connected"
            }
            
            if connected:
                # Test 2: List tools
                try:
                    tools = await client.list_tools()
                    tool_names = [tool.name for tool in tools]
                    debug_info["tools_test"] = {
                        "success": True,
                        "status": "✅ Tools listed successfully",
                        "available_tools": tool_names,
                        "count": len(tools)
                    }
                    
                    # Test 3: Try to get schema
                    schema_tool_names = ["get-neo4j-schema", "get_neo4j_schema", "get-schema", "schema"]
                    schema_tool = None
                    for name in schema_tool_names:
                        if name in tool_names:
                            schema_tool = name
                            break
                    
                    if schema_tool:
                        try:
                            schema_result = await client.call_tool(schema_tool, {})
                            debug_info["schema_test"] = {
                                "success": True,
                                "status": f"✅ Schema retrieved using tool '{schema_tool}'",
                                "tool_used": schema_tool,
                                "result_type": str(type(schema_result[0])) if schema_result else "None",
                                "result_length": len(schema_result) if schema_result else 0
                            }
                        except Exception as schema_error:
                            debug_info["schema_test"] = {
                                "success": False,
                                "status": f"❌ Schema tool '{schema_tool}' failed",
                                "tool_used": schema_tool,
                                "error": str(schema_error)
                            }
                    else:
                        debug_info["schema_test"] = {
                            "success": False,
                            "status": "❌ No schema tool found",
                            "available_tools": tool_names,
                            "expected_tools": schema_tool_names
                        }
                        
                except Exception as tools_error:
                    debug_info["tools_test"] = {
                        "success": False,
                        "status": "❌ Failed to list tools",
                        "error": str(tools_error)
                    }
            
    except Exception as e:
        debug_info["error_details"] = {
            "error_type": str(type(e).__name__),
            "error_message": str(e),
            "status": "❌ Connection failed"
        }
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    print("Starting Neo4j Query API server...")
    print("Make sure the MCP server is running at http://127.0.0.1:8000/mcp/")
    uvicorn.run(app, host="0.0.0.0", port=8080)
