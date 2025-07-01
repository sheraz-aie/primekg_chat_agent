# neo4j_mcp_server.py
import os
from fastmcp import FastMCP, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Define the connection details for your local Neo4j 'primekg' database.
# Ensure your Docker container is running.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Default URI for Neo4j
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")  # IMPORTANT: Replace or set env var
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD") # IMPORTANT: Replace or set env var
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "primekg")  # Default database name

# Verify that required environment variables are not null
if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME environment variable is required")
if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD environment variable is required")

print("Using Neo4j configuration:")

def create_server():
    """
    Creates a FastMCP proxy server for the mcp-neo4j-cypher package.
    """
    backend_config = {
        "mcpServers": {
            "neo4j-backend": {
                "command": "uvx",
                "args": ["mcp-neo4j-cypher==0.2.3", "--transport", "stdio"],
                "env": {
                    "NEO4J_URI": NEO4J_URI,
                    "NEO4J_USERNAME": NEO4J_USERNAME,
                    "NEO4J_PASSWORD": NEO4J_PASSWORD,
                    "NEO4J_DATABASE": NEO4J_DATABASE,
                },
            }
        }
    }
    
    
     # The Client constructor now receives a valid MCPConfig dictionary.
    backend_client = Client(backend_config)


    # FastMCP.as_proxy() creates a new server that forwards all requests
    # to the backend client[cite: 1556, 1690].
    # This allows us to bridge the STDIO backend to an HTTP frontend.
    proxy_server = FastMCP.as_proxy(
        backend_client,
        name="Neo4jCypherProxy"
    )
    
    return proxy_server

if __name__ == "__main__":
    # Ensure the Neo4j password is set before running.
    if "NEO4J_PASSWORD" not in os.environ or os.environ["NEO4J_PASSWORD"] == "your-neo4j-password":
        print("ERROR: Please set the NEO4J_PASSWORD environment variable before running.")
    else:
        server = create_server()
        print("Starting Neo4j MCP Proxy Server on http://127.0.0.1:8000/mcp/")
        
        # Run the server using the recommended Streamable HTTP transport[cite: 510].
        server.run(
            transport="http",
            host="127.0.0.1",
            port=8000
        )