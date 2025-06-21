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
    """Extract tool calls from <tool_call> and <code_interpreter> tags."""
    tool_calls = []
    # Pattern to find <tool_call> or <code_interpreter> blocks
    pattern = r'<(tool_call|code_interpreter)>(.*?)</\1>'
    
    for tag_name, content in re.findall(pattern, text, re.DOTALL):
        content = content.strip()
        
        # If the AI explicitly uses the code_interpreter tag, the content is always treated as code.
        if tag_name == "code_interpreter":
            tool_calls.append({
                "name": "code_interpreter",
                "arguments": { "code_markdown": content }
            })
            continue # Move to next match

        # If the AI uses the generic tool_call tag, we expect a JSON object.
        if tag_name == "tool_call":
            try:
                # Clean up potential markdown ```json ... ``` wrapper
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                
                parsed_json = json.loads(content)
                
                # Standard, expected format
                if 'name' in parsed_json and 'arguments' in parsed_json:
                    tool_calls.append(parsed_json)
                
                # Check for the specific malformed AI output and transform it
                elif parsed_json.get("tool") == "code_interpreter" and "code" in parsed_json:
                    tool_calls.append({
                        "name": "code_interpreter",
                        "arguments": { "code_markdown": parsed_json["code"] }
                    })
                
                else:
                    # The JSON is malformed in an unexpected way. Fallback to treating it as code.
                    print(f"Warning: Malformed tool call JSON. Treating as code: {content}")
                    tool_calls.append({
                        "name": "code_interpreter",
                        "arguments": { "code_markdown": content }
                    })
            except json.JSONDecodeError:
                # The most likely error is that the AI wrote raw code instead of JSON.
                # We will treat this as a call to the code interpreter.
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
    
    depth = 0
    while depth < max_depth:
        # Build the prompt with tools and conversation history
        system_prompt = f"""You are a helpful AI assistant with access to various tools. You can execute code, search the web, and get stock information.
Plan your steps and then execute the tools required to answer the user's request. After a tool is executed, first provide a helpful summary or answer based on the tool's output. Only make another tool call if it is essential to fulfilling the user's original request.
When you need to use a tool, wrap your function call in XML tags like this: <tool_call> (for JSON structured calls) or <code_interpreter> (for raw code). Always provide helpful responses to the user's queries."""

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
                # All the intermediate "thinking" steps will be hidden in this expander
                with st.expander("View Chain of Thought", expanded=False):
                    # Display the AI's raw response, but don't save it to the main chat history
                    display_message({"role": "assistant", "content": response_text})
                    
                    # Display a detailed trace of the tool calls
                    st.markdown("---")
                    st.markdown("### Parsed Tool Calls")
                    st.json(tool_calls)

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
                            
                            # Bypass the LangChain tool wrapper and call the original function directly
                            # This makes it a standard keyword-based python function call
                            if hasattr(function_to_call, 'func'):
                                result = function_to_call.func(**function_args)
                            else:
                                result = function_to_call(**function_args) # Fallback for non-decorated functions
                            
                            # Prepare result for UI and model
                            result_payload = {"name": function_name, "content": ""}
                            if isinstance(result, dict):
                                result_payload['content'] = result.get('stdout', '')
                                if 'plot_image' in result and result['plot_image']:
                                    result_payload['plot_image'] = result['plot_image']
                            elif isinstance(result, pd.DataFrame):
                                result_payload['content'] = result # Pass the DataFrame directly
                            else:
                                result_payload['content'] = str(result)
                            
                            # Display rich result immediately
                            tool_msg = {"role": "tool", **result_payload}
                            display_message(tool_msg)
                            st.session_state.messages.append(tool_msg)
                            
                            # Add simpler text result for the model's history
                            conversation_history.append({"role": "tool", "content": f"Tool {function_name} returned: {str(result)}"})

                        else: # Validation failed
                            error_content = f"Tool validation error for {function_name}: {message}"
                            err_msg = {"role": "error", "content": error_content}
                            display_message(err_msg)
                            st.session_state.messages.append(err_msg)
                            conversation_history.append({"role": "tool", "content": error_content})

                    except Exception as e: # Execution failed
                        error_content = f"Execution error for {function_name}: {str(e)}"
                        err_msg = {"role": "error", "content": error_content}
                        display_message(err_msg)
                        st.session_state.messages.append(err_msg)
                        conversation_history.append({"role": "tool", "content": error_content})

                # After executing tools, force the model to provide a response before calling another tool
                # This check ensures we don't get stuck in a loop
                if tool_calls:
                    depth += 1
                    if depth >= max_depth:
                        break # Exit if max depth is reached

                    # Now, create a new prompt that asks the model to respond to the user based on the tool output
                    final_prompt_text = "Based on the tool results, please provide a direct answer to the user's request."
                    conversation_history.append({"role": "user", "content": final_prompt_text})
                    
                    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
                    full_prompt = f"{system_prompt}\n\n{history_text}"
                    
                    response_text = bedrock_chat(full_prompt, model_id)
                    # This should be the final answer, so we display it and break the loop
                    final_msg = {"role": "assistant", "content": response_text}
                    display_message(final_msg)
                    st.session_state.messages.append(final_msg)
                    break

            # If no tool calls, this is the final answer
            else:
                final_msg = {"role": "assistant", "content": response_text}
                display_message(final_msg)
                st.session_state.messages.append(final_msg)
                break # Exit the loop

        except Exception as e:
            err_msg = {"role": "error", "content": f"An unexpected error occurred: {str(e)}"}
            display_message(err_msg)
            st.session_state.messages.append(err_msg)
            break
    
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
            
            # Define patterns for both markdown and HTML images
            base64_pattern_md = r'!\[.*?\]\(data:image/png;base64,(.*?)\)'
            base64_pattern_html = r'<img src="data:image/png;base64,(.*?)">'
            
            # Search for both patterns
            match_md = re.search(base64_pattern_md, content, re.DOTALL)
            match_html = re.search(base64_pattern_html, content, re.DOTALL)
            
            if match_md:
                # Handle Markdown image
                text_content = re.sub(base64_pattern_md, '', content, flags=re.DOTALL).strip()
                base64_data = match_md.group(1)
                pattern_to_remove = base64_pattern_md
            elif match_html:
                # Handle HTML image
                text_content = re.sub(base64_pattern_html, '', content, flags=re.DOTALL).strip()
                base64_data = match_html.group(1)
                pattern_to_remove = base64_pattern_html
            else:
                # No image found, just display content
                st.markdown(content)
                return # Exit function

            # If an image was found and parsed
            if text_content:
                st.markdown(text_content)
            try:
                # Decode and display the image
                st.image(base64.b64decode(base64_data), caption="Generated Chart from AI")
            except Exception as e:
                st.error(f"Failed to display chart from AI response. Raw data might be incomplete. Error: {e}")

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


