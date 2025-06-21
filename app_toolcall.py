import streamlit as st
import boto3
import json
import functions
from inference import ModelInference
from prompter import PromptManager
from validator import validate_function_call_schema
from utils import validate_and_extract_tool_calls, get_assistant_message
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from botocore.exceptions import ClientError
import os
import re
import base64
import pandas as pd
import io
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(
    page_title="Tool Calling Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = None

# Define model IDs at the top for global use (same as app_docling.py)
CHAT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Define available model IDs
AVAILABLE_MODELS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-instant-v1",
    "anthropic.claude-v2"
]

def enable_bedrock(region='ap-south-1'):
    """
    Enable AWS Bedrock using credentials and region from environment variables.
    Returns a Bedrock runtime client.
    """
    # Set region if not already set
    if not os.environ.get('AWS_REGION'):
        os.environ['AWS_REGION'] = region

    # Create Bedrock runtime client using environment variables
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=os.environ['AWS_REGION'],
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.environ.get('AWS_SESSION_TOKEN')  # This will be None if not set, which is fine
    )

    return bedrock_runtime

def _display_tool_trace(tool_traces):
    """Helper function to display the detailed tool trace inside a collapsible window."""
    if tool_traces:
        with st.expander("Tool Trace", expanded=False):
            for i, trace in enumerate(tool_traces):
                st.markdown(f"---")
                st.markdown(f"#### Step {i+1}: Tool Call")
                st.json(trace['call'])
                
                st.markdown(f"#### Step {i+1}: Tool Result")
                result = trace.get('result', {})
                error = result.get('error')

                if error:
                    st.error(error)
                else:
                    content = result.get("content")
                    plot_image = result.get("plot_image")

                    if plot_image:
                        try:
                            st.image(base64.b64decode(plot_image), caption="Generated Chart")
                        except Exception as e:
                            st.error(f"Failed to display chart from tool: {e}")
                    
                    if content is not None:
                        is_empty_string = isinstance(content, str) and not content.strip()
                        is_empty_collection = isinstance(content, (list, dict)) and not content

                        # Display a clear message for empty results
                        if is_empty_string or is_empty_collection:
                            st.info("Tool returned no results.")
                        # Display other results based on their type
                        elif isinstance(content, pd.DataFrame):
                            st.dataframe(content)
                        elif isinstance(content, (dict, list)):
                            st.json(content)
                        else:
                            st.code(str(content), language=None)
                    # Handle cases where there's no content but also no error (e.g., a tool that just returns a plot)
                    elif not plot_image:
                        st.info("Tool executed but returned no displayable content.")

# Bedrock Chat Completion (same as app_docling.py)
def bedrock_chat(prompt, model_id=None):
    if model_id is None:
        model_id = CHAT_MODEL_ID
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    response = st.session_state.bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

# Title and description
st.title("🤖 Tool Calling Agent")
st.markdown("""
This agent can execute code, search the web, and retrieve stock information using various tools.
Ask me anything and I'll use the appropriate tools to help you!
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Choose Model",
        ["Bedrock Claude", "Local Model"],
        help="Select whether to use AWS Bedrock or a local model"
    )
    
    if model_option == "Local Model":
        model_path = st.text_input(
            "Model Path",
            value="NousResearch/Hermes-2-Pro-Llama-3-8B",
            help="Path to the local model"
        )
        load_in_4bit = st.checkbox("Load in 4-bit", value=False)
        chat_template = st.selectbox("Chat Template", ["chatml", "zephyr", "vicuna"])
        max_depth = st.slider("Max Recursion Depth", 1, 10, 5)
    else:
        # Bedrock configuration
        st.subheader("AWS Bedrock Settings")
        aws_region = st.text_input("AWS Region", value="ap-south-1", help="Use the same region as your working setup")
        model_id = st.selectbox(
            "Bedrock Model",
            AVAILABLE_MODELS,
            index=0,
            help="Select a model that's enabled in your AWS Bedrock console"
        )
        max_depth = st.slider("Max Recursion Depth", 1, 10, 5)

# Initialize Bedrock client if using Bedrock
if model_option == "Bedrock Claude" and "bedrock_client" not in st.session_state:
    try:
        st.session_state.bedrock_client = enable_bedrock(region=aws_region)
        st.success("✅ Bedrock client initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize Bedrock client: {e}")
        st.session_state.bedrock_client = None

# Initialize local model if selected
if model_option == "Local Model" and st.session_state.inference_engine is None:
    if st.button("Initialize Local Model"):
        with st.spinner("Loading model..."):
            try:
                st.session_state.inference_engine = ModelInference(
                    model_path, 
                    chat_template, 
                    str(load_in_4bit)
                )
                st.success("✅ Local model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load local model: {e}")

# Function to extract tool calls from text using regex
def extract_tool_calls_from_text(text):
    """Extract tool calls from <tool_call> and <code_interpreter> tags with robust parsing."""
    tool_calls = []
    pattern = r'<(tool_call|code_interpreter)>(.*?)</\1>'
    
    for tag_name, content in re.findall(pattern, text, re.DOTALL):
        content = content.strip()
        
        # 1. Handle explicit code_interpreter calls
        if tag_name == "code_interpreter":
            tool_calls.append({
                "name": "code_interpreter",
                "arguments": { "code_markdown": content }
            })
            continue

        # 2. Handle generic tool_call tag, which could be JSON or raw code
        try:
            # Clean up potential markdown ```json ... ``` wrapper
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            
            parsed_json = json.loads(content)
            
            tool_name = None
            args_dict = {}

            # Find tool name (flexible)
            if 'name' in parsed_json:
                tool_name = parsed_json.get('name')
            elif 'tool' in parsed_json:
                tool_name = parsed_json.get('tool')

            # Find arguments (flexible)
            if 'arguments' in parsed_json:
                args_dict = parsed_json.get('arguments', {})
            elif 'args' in parsed_json:
                args_dict = parsed_json.get('args', {})
            else:
                # Assume the rest of the dict are the arguments
                temp_args = parsed_json.copy()
                temp_args.pop('name', None)
                temp_args.pop('tool', None)
                args_dict = temp_args

            # If we found a tool name, build the standardized tool call
            if tool_name:
                # Ensure args_dict is a dictionary before passing it
                if not isinstance(args_dict, dict):
                    args_dict = {}
                tool_calls.append({
                    "name": tool_name,
                    "arguments": args_dict
                })
            else:
                 # If no tool name found, it's likely code
                raise json.JSONDecodeError("No tool name found", content, 0)

        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails or it's not a dict, it's probably raw code.
            # Treat it as a call to the code interpreter.
            tool_calls.append({
                "name": "code_interpreter",
                "arguments": { "code_markdown": content }
            })
                
    return tool_calls

# Function to execute tool calls with Bedrock (using simple chat)
def execute_tool_calls_bedrock(query, max_depth=5):
    """Execute tool calls using AWS Bedrock with simple chat, displaying messages in real-time."""
    
    if not st.session_state.bedrock_client: 
        err_msg = {"role": "error", "content": "Bedrock client not initialized"}
        display_message(err_msg)
        st.session_state.messages.append(err_msg)
        return
    
    # Get available tools
    tools = functions.get_openai_tools()
    
    # Create tool descriptions for the prompt
    tools_description = ""
    for tool in tools:
        func_info = tool.get("function", {})
        tools_description += f"- {func_info.get('name', 'Unknown')}: {func_info.get('description', 'No description')}\n"
    
    # Initialize conversation history FOR THE MODEL
    conversation_history = [
        {"role": "user", "content": query}
    ]
    
    tool_traces = [] # To store the detailed trace of each tool call
    depth = 0
    while depth < max_depth:
        # Build the prompt with tools and conversation history
        system_prompt = f"""You are a helpful AI assistant. Your goal is to answer the user's request by planning and executing the necessary tools.

**Your Task:**
1.  Analyze the user's request and plan the steps.
2.  Use one or more tools sequentially to gather information or perform actions.
3.  To create a chart, first get the data, then use the `code_interpreter` with Matplotlib to generate the plot.
4.  Provide a final, direct answer based **only** on the tool results. Do not include apologies or conversational filler.

**Tool Format:** Call tools using JSON inside a `<tool_call>` tag.

**CRITICAL RULES:**
- Your final answer **MUST** use the exact data returned by the tools. Do not use your own knowledge or invent information.
- If a tool returns an error or no results, try a different tool or approach to answer the user's request. Only report failure to the user if you have no other options.
- The `code_interpreter` tool CANNOT call other tools (like `google_search_and_scrape`). You MUST use a separate `<tool_call>` for each tool.

**Available tools:**
```
{tools_description}
```
Begin by planning your steps.
"""

        # Build the full prompt for the model
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        full_prompt = f"{system_prompt}\n\n{history_text}"

        try:
            # Get response from Bedrock
            response_text = bedrock_chat(full_prompt, model_id)
            
            # Extract tool calls from the response
            tool_calls = extract_tool_calls_from_text(response_text)
            
            # If there are tool calls, execute them and continue the loop
            if tool_calls:
                # All the intermediate "thinking" steps will be hidden.
                # The expander has been removed to create a cleaner UI.
                
                # Add the AI's thinking process to the conversation history for the next turn
                conversation_history.append({"role": "assistant", "content": response_text})

                for tool_call in tool_calls:
                    function_name = tool_call.get("name", "unknown")
                    
                    try:
                        # Validate and execute the function
                        validation, message = validate_function_call_schema(tool_call, tools)
                        if validation:
                            function_args = tool_call["arguments"]

                            # Safeguard for code_interpreter: ensure code is in a markdown block
                            if function_name == 'code_interpreter':
                                code = function_args.get('code_markdown', '')
                                if not code.strip().startswith('```python'):
                                    function_args['code_markdown'] = f"```python\n{code}\n```"

                            function_to_call = getattr(functions, function_name)
                            
                            # Reverting to keyword-based argument passing for robustness.
                            # This is safer than positional arguments and should prevent errors.
                            if hasattr(function_to_call, 'func'):
                                result = function_to_call.func(**function_args)
                            else:
                                result = function_to_call(**function_args) # Fallback for non-decorated functions
                            
                            # Prepare result for UI and model
                            result_payload = {"name": function_name, "content": ""}
                            if isinstance(result, dict):
                                # Handle dicts from code_interpreter (plots, stdout) vs other tools (data)
                                if function_name == 'code_interpreter':
                                    result_payload['content'] = result.get('stdout', '')
                                    if 'plot_image' in result and result['plot_image']:
                                        result_payload['plot_image'] = result['plot_image']
                                else:
                                    # For other tools that return a dict, check for an error key
                                    if 'error' in result:
                                        result_payload['error'] = result['error']
                                    else:
                                        result_payload['content'] = result
                            elif isinstance(result, pd.DataFrame):
                                result_payload['content'] = result
                            else:
                                result_payload['content'] = str(result)
                            
                            tool_traces.append({'call': tool_call, 'result': result_payload})
                            
                            # The tool message is added to the session state
                            tool_msg = {"role": "tool", **result_payload}
                            st.session_state.messages.append(tool_msg)
                            
                            # Special case: If the result contains a plot, display it immediately
                            # This ensures charts are visible without cluttering the UI with other tool outputs.
                            if tool_msg.get("plot_image"):
                                display_message(tool_msg)
                            
                            # Add a concise summary of the tool's result to the model's history
                            # to avoid overwhelming it with raw data.
                            tool_response_for_model = ""
                            if isinstance(result, pd.DataFrame):
                                if not result.empty:
                                    tool_response_for_model = f"Tool {function_name} returned a DataFrame with {result.shape[0]} rows. Here are the first 3 rows:\n{result.head(3).to_string()}"
                                else:
                                    tool_response_for_model = f"Tool {function_name} returned an empty DataFrame."
                            else:
                                tool_response_for_model = f"Tool {function_name} returned: {str(result)}"
                            
                            conversation_history.append({"role": "tool", "content": tool_response_for_model})

                        else: # Validation failed
                            error_content = f"Tool validation error for {function_name}: {message}"
                            tool_traces.append({'call': tool_call, 'result': {'error': error_content}})
                            err_msg = {"role": "error", "content": error_content}
                            st.session_state.messages.append(err_msg)
                            conversation_history.append({"role": "tool", "content": error_content})

                    except Exception as e: # Execution failed
                        error_content = f"Execution error for {function_name}: {str(e)}"
                        tool_traces.append({'call': tool_call, 'result': {'error': error_content}})
                        err_msg = {"role": "error", "content": error_content}
                        st.session_state.messages.append(err_msg)
                        conversation_history.append({"role": "tool", "content": error_content})

                # After executing tools, the loop will continue, allowing for more tool calls if needed.
                depth += 1
                if depth >= max_depth:
                    break  # Exit if max depth is reached

            # If no tool calls, this is the final answer
            else:
                _display_tool_trace(tool_traces)
                final_msg = {"role": "assistant", "content": response_text}
                display_message(final_msg)
                st.session_state.messages.append(final_msg)
                return # We're done, so we exit the function entirely

        except Exception as e:
            err_msg = {"role": "error", "content": f"An unexpected error occurred: {str(e)}"}
            display_message(err_msg)
            st.session_state.messages.append(err_msg)
            break
    
    # This part is reached if the loop breaks from max_depth or an error
    _display_tool_trace(tool_traces)

    if depth >= max_depth:
         err_msg = {"role": "error", "content": "Maximum tool call depth reached. The AI may be in a loop."}
         display_message(err_msg)
         st.session_state.messages.append(err_msg)

# Function to execute tool calls with local model
def execute_tool_calls_local(query, max_depth=5):
    """Execute tool calls using local model with recursive function calling"""
    
    if not st.session_state.inference_engine:
        return "Local model not initialized"
    
    try:
        # Use the inference engine's recursive function calling
        st.session_state.inference_engine.generate_function_call(
            query, 
            "chatml", 
            None, 
            max_depth
        )
        return "Local model execution completed"
    except Exception as e:
        return f"Error: {str(e)}"

def display_message(msg):
    """Helper function to display a message in the chat UI."""
    with st.chat_message(msg["role"]):
        # Assistant messages
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            
            # Use a regex to find markdown tables
            table_pattern = r'(\n\s*\|.*\|.*\n(?:\|.*\|.*\n)+)'
            tables = re.findall(table_pattern, content)

            if tables:
                # If tables are found, split the content and render them with st.table
                # This provides better formatting than standard markdown.
                
                # Split content by the found tables
                remaining_content_parts = re.split(table_pattern, content)
                
                for part in remaining_content_parts:
                    if part in tables:
                        # This part is a table
                        try:
                            # Convert markdown table to a list of lists
                            lines = part.strip().split('\n')
                            header = [h.strip() for h in lines[0].strip('|').split('|')]
                            data = [
                                [cell.strip() for cell in row.strip('|').split('|')]
                                for row in lines[2:] # Skip header and separator line
                            ]
                            df = pd.DataFrame(data, columns=header)
                            st.table(df)
                        except Exception:
                            # If parsing fails, fall back to markdown
                            st.markdown(part)
                    else:
                        # This part is regular text
                        st.markdown(part)
            else:
                # If no tables, just render the markdown as is
                st.markdown(content)
            
            # The logic to handle base64 images is removed from here
            # as it's now handled in the 'tool' message display.

        # Tool messages
        elif msg["role"] == "tool":
            tool_name = msg.get('name', 'unknown_tool')
            content = msg.get("content")
            
            # For code interpreter, show plot if it exists, otherwise show text output
            if tool_name == 'code_interpreter':
                st.markdown("🔧 **Code Interpreter Output**")
                if msg.get("plot_image"):
                    try:
                        st.image(base64.b64decode(msg["plot_image"]), caption="Generated Chart")
                    except Exception as e:
                        st.error(f"Failed to display chart from tool: {e}")

                if content:
                    st.code(content, language="text")
            # For all other tools, format the output based on its type
            else:
                st.markdown(f"🔧 **Tool Output ({tool_name})**")
                if content is not None:
                    if isinstance(content, pd.DataFrame):
                        st.dataframe(content)
                    elif isinstance(content, dict) or (isinstance(content, list) and content and isinstance(content[0], dict)):
                        st.json(content)
                    else:
                        st.code(str(content), language=None)
        
        # Error messages
        elif msg["role"] == "error":
            st.error(msg["content"])
            
        # Any other message type
        else:
            st.markdown(msg.get("content", ""))

# Main chat interface
st.header("💬 Chat with the Agent")

# Display chat messages from history
for message in st.session_state.messages:
    display_message(message)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message({"role": "user", "content": prompt})
    
    # Generate and display the response
    with st.spinner("Thinking..."):
        if model_option == "Bedrock Claude":
            execute_tool_calls_bedrock(prompt, max_depth)
        else:
            response_messages = execute_tool_calls_local(prompt, max_depth)
            
            # Handle local model response (which may be a string or list)
            if isinstance(response_messages, list):
                for msg in response_messages:
                    display_message(msg)
                    st.session_state.messages.append(msg)
            elif isinstance(response_messages, str):
                # Display the string message if that's what's returned
                info_msg = {"role": "assistant", "content": response_messages}
                display_message(info_msg)
                st.session_state.messages.append(info_msg)

# Display available tools
with st.expander("🔧 Available Tools"):
    tools = functions.get_openai_tools()
    for tool in tools:
        func_info = tool.get("function", {})
        st.markdown(f"**{func_info.get('name', 'Unknown')}**")
        st.markdown(f"*{func_info.get('description', 'No description')}*")
        if func_info.get('parameters'):
            st.markdown("Parameters:")
            for param, details in func_info['parameters'].get('properties', {}).items():
                st.markdown(f"  - `{param}`: {details.get('type', 'unknown')}")
        st.divider()

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **This app uses the same Bedrock setup as your working apps:**
    
    - Uses the same `bedrock_chat()` function
    - Uses the same model ID and region
    - Uses the same AWS credentials
    
    **If you still get errors:**
    - Check that your AWS credentials are properly set
    - Ensure you're using the correct region
    - Try using the Local Model option instead
    """)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


