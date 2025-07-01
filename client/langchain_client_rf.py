# langchain_client_refactored.py
import asyncio
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp/"
USER_QUESTION = "Which diseases does Quetiapine treat?"
#USER_QUESTION = "What proteins does the drug Metformin interact with?"

# --- Main Application Logic ---

async def main():
    """
    Connects to the MCP server and runs a refactored LCEL chain to query Neo4j.
    """
    # Create the MCP client using FastMCP with StreamableHttpTransport.
    transport = StreamableHttpTransport(url=MCP_SERVER_URL)
    client = Client(transport=transport)
    
    # The async context manager ensures the client is properly connected and closed.
    async with client:
        print(f"Client connected: {client.is_connected()}")
        
        # --- Define Tool-Calling Functions within the client context ---
        # These async functions wrap the manual client.call_tool invocations.

        async def get_schema(_input) -> str:
            """
            Invokes the 'get_neo4j_schema' tool and returns the schema as a string.
            """
            print("Chain step: Fetching schema...")
            schema_result = await client.call_tool("get_neo4j_schema", {})
            if schema_result and isinstance(schema_result[0], TextContent):
                return schema_result[0].text
            return "No schema available"

        async def run_cypher(query: str) -> str:
            """
            Takes a Cypher query string and executes it using the 'read_neo4j_cypher' tool.
            """
            print(f"Chain step: Executing Cypher -> {query.strip()}")
            result = await client.call_tool("read_neo4j_cypher", {"query": query})
            if result and isinstance(result[0], TextContent):
                return result[0].text
            return "[]" # Return empty JSON array on failure

        # --- Define the LangChain Components ---
        
        chat_llm = ChatOpenAI(
            model='gpt-4.1-mini', 
            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None, 
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
        
        # --- Construct the Unified LCEL Chain ---
        # This chain now handles the entire workflow automatically.
        print("Constructing the full LangChain Expression Language (LCEL) chain...")
        
        full_chain = (
            {
                # The first step assigns two keys to a dictionary:
                # 1. 'schema' is populated by calling our get_schema function.
                # 2. 'question' is the original user input, passed through.
                "schema": RunnableLambda(get_schema),
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | chat_llm
            | StrOutputParser()
            | RunnableLambda(run_cypher) # The generated query string is passed here
        )

        # --- Invoke the Chain ---
        print(f"\nExecuting chain for question: '{USER_QUESTION}'")
        final_result = await full_chain.ainvoke(USER_QUESTION)

        print("\n--- Final JSON Result ---")
        print(final_result)
    
    print(f"\nClient connection status: {client.is_connected()}")

if __name__ == "__main__":
    asyncio.run(main())