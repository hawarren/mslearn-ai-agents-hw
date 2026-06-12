import os
import asyncio
import json
import sys
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam

# Add references
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Clear the console
os.system('cls' if os.name=='nt' else 'clear')

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

async def connect_to_server(exit_stack: AsyncExitStack):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=None
    )

    # Start the MCP server
    '''In a standard production setup, the server would run separately from the client. But
    for the sake of this demo, the client is responsible for starting the server 
    using standard input/output transport. 
    This creates a lightweight communication channel 
    between the two components and simplifies the local development setup.
    '''
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio,write = stdio_transport


    # Create an MCP client session
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    # List available tools
    response = await session.list_tools()
    tools = response.tools
    print("\nConnected to server with tools:", [tool.name for tool in tools])

    return session

async def chat_loop(session):

    # Connect to the agents client
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):

        # Get the mcp tools available from the server
        response = await session.list_tools()
        tools = response.tools

        # Build a function for each tool
        def make_tool_func(tool_name):
            async def tool_func(**kwargs):
                result = await session.call_tool(tool_name, kwargs)
                return result
            tool_func.__name__ = tool_name
            return tool_func
        #Store the functions in a dictionary for easy access when creating the agent
        functions_dict = {tool.name: make_tool_func(tool.name) for tool in tools}
        # Create FunctionTool definitions for the agent
        mcp_function_tools: FunctionTool = []
        for tool in tools:
            function_tool = FunctionTool(
                name=tool.name,
                description=tool.description,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=True
            )
            mcp_function_tools.append(function_tool)

        # Create the agent
        agent = project_client.agents.create_version(
            agent_name="inventory-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions="""
                You are an inventory assistant. Here are some general guidelines:
                - Recommend restock if item inventory < 10 and weekly sales > 5.
                - Recommend clearance if item inventory > 20 and weekly sales < 5.
                """,
                tools=mcp_function_tools,
            ),
        )


        # Track the last completed response so each turn can preserve context
        previous_response_id = None

        while True:
            user_input = input("Enter a prompt for the inventory agent. Use 'quit' to exit.\nUSER: ").strip()
            if user_input.lower() == "quit":
                print("Exiting chat.")
                break

            # Send a prompt to the agent
            input_list: ResponseInputParam = [
                {"type": "message", "role": "user", "content": user_input}
            ]

            try:
                # Retrieve the agent's response, which may include function calls to the MCP server tools
                response_kwargs = {
                    "input": input_list,
                    "extra_body": {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                }
                if previous_response_id is not None:
                    response_kwargs["previous_response_id"] = previous_response_id

                response = openai_client.responses.create(**response_kwargs)
                while True:
                    # Check the run status for failures
                    if response.status == "failed":
                        print(f"Response failed: {response.error}")
                        break
                    input_list = []

                    # Process function calls
                    for item in response.output:
                        if item.type == "function_call":
                            # Retrieve the matching function tool
                            function_name = item.name
                            kwargs = json.loads(item.arguments)
                            required_function=functions_dict.get(function_name)
                            if required_function is None:
                                input_list.append(
                                    FunctionCallOutput(
                                        type="function_call_output",
                                        call_id=item.call_id,
                                        output=f"Error: unknown tool '{function_name}'",
                                    )                        
                                )
                                continue
                            
                            #Invoke the function
                            output = await required_function(**kwargs)
                            
                            # Append the output text
                            input_list.append(
                                FunctionCallOutput(
                                    type="function_call_output",
                                    call_id=item.call_id,
                                    output=output.content[0].text,
                                )                        
                            )
                    if not input_list:
                        break

                    # Send function call outputs back to the model and retrieve a response
                    
                    response = openai_client.responses.create(
                        input=input_list,
                        previous_response_id=response.id,
                        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                    )               
                previous_response_id = response.id
                print(f"Agent response: {response.output_text}")
            except Exception as exc:
                print(f"Request failed: {exc}")
                continue
           
        # Delete the agent when done
        print("Cleaning up agents:")
        project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        print("Deleted inventory agent.")


async def main():
    import sys
    exit_stack = AsyncExitStack()
    try:
        session = await connect_to_server(exit_stack)
        await chat_loop(session)
    finally:
        await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())