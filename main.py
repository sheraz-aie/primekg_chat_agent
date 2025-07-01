import sys
import asyncio

def main():
    """
    Main entry point with options to run different components.
    """
    print("Neo4j MCP Experiment Options:")
    print("1. Run original langchain client (python langchain_client.py)")
    print("2. Run refactored langchain client (python langchain_client_rf.py)")  
    print("3. Start FastAPI server (python api_server.py)")
    print("4. Start Neo4j MCP server (python neo4j_mcp_server.py)")
    print("\nFor FastAPI server, you can also run: uvicorn api_server:app --host 0.0.0.0 --port 8080 --reload")
    print("Make sure to start the Neo4j MCP server first before running clients or API server.")

if __name__ == "__main__":
    main()
