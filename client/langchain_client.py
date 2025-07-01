# langchain_client.py
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

model_name = 'gpt-4.1-mini'  # Default to gpt-4o if not set
# 1. Define the URL of our running MCP server.
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp/"

chat_llm = ChatOpenAI(model=model_name, api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None, temperature=0.0)
#chat_llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=0.0)

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

# 3. Define the natural language question.
USER_QUESTION = "Which diseases does Quetiapine treat?"


async def main():
    # 4. Create the MCP client using FastMCP with StreamableHttpTransport.
    transport = StreamableHttpTransport(url=MCP_SERVER_URL)
    client = Client(transport=transport)
    
    schema_tool_str = "get_neo4j_schema"
    cypher_tool_str = "read_neo4j_cypher"
    
    async with client:
        print(f"Client connected: {client.is_connected()}")
        
        # 5. Load tools from the remote server.
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Create a dictionary from the tools list for easier access
        tools_dict = {tool.name: tool for tool in tools}
        
        if schema_tool_str not in tools_dict:
            print("Error: get-neo4j-schema tool not found!")
            return
        
        if cypher_tool_str not in tools_dict:
            print("Error: read-neo4j-cypher tool not found!")
            return
            
        print("Tools loaded successfully from remote server!")

        # 6. Get the Neo4j schema first
        schema_result = await client.call_tool(schema_tool_str, {})
        # Handle different content types safely
        schema = "No schema available"
        if schema_result:
            content = schema_result[0]
            # Check if it's text content using isinstance
            if isinstance(content, TextContent):
                schema = content.text
            else:
                schema = str(content)
        print(f"Schema retrieved: {len(schema)} characters")

        # 7. Create a runnable that uses the chat LLM to generate a Cypher query.
        cypher_gen_chain = (
            prompt_template
            | chat_llm
            | StrOutputParser()
        )

        # 8. Run the query generation with the schema and user question.
        cypher_query = await cypher_gen_chain.ainvoke({
            "schema": schema,
            "question": USER_QUESTION
        })

        print(f"Generated Cypher Query: {cypher_query}")
        
        # 9. Execute the Cypher query using the cypher tool
        result = await client.call_tool(cypher_tool_str, {"query": cypher_query})
        print(f"Query Result: {result}")
    
    print(f"Client connection status: {client.is_connected()}")


if __name__ == "__main__":
    asyncio.run(main())