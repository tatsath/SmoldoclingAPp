import streamlit as st
import pandas as pd
import os
import boto3
import json
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from pathlib import Path
from CSVAlchemy.core import encode_workbook
from CSVAlchemy.utils import load_workbook_from_file
from CSVAlchemy.config import get_config
import tempfile

st.set_page_config(page_title="LangChain: Chat with Excel Files", page_icon="🦜")
st.title("🦜 LangChain: Chat with Excel Files")

# File format support
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """Clear the Submit Button State"""
    st.session_state["submit"] = False

def enable_bedrock(region='ap-south-1', embedding_model_id="amazon.titan-embed-text-v2:0"):
    """
    Enable AWS Bedrock using credentials and region from environment variables.
    Returns a Bedrock runtime client and BedrockEmbeddings object.
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
        aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
    )

    return bedrock_runtime

# Bedrock Chat Completion function
def bedrock_chat(prompt, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
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
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

# Custom LLM class that wraps the Bedrock chat function
class BedrockLLMWrapper:
    def __init__(self, bedrock_runtime, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
        self.bedrock_runtime = bedrock_runtime
        self.model_id = model_id
        self.temperature = 0
    
    def __call__(self, prompt):
        # Handle StringPromptValue objects
        if hasattr(prompt, 'to_string'):
            prompt = prompt.to_string()
        elif hasattr(prompt, 'text'):
            prompt = prompt.text
        return bedrock_chat(prompt, self.model_id)
    
    def invoke(self, prompt):
        return self.__call__(prompt)
    
    def bind(self, **kwargs):
        """Bind additional parameters to the LLM"""
        return self
    
    def with_config(self, **kwargs):
        """Return a new instance with updated configuration"""
        return self
    
    def with_fallbacks(self, fallbacks):
        """Return a new instance with fallbacks"""
        return self

@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    """Load data from uploaded file, handling multiple sheets for Excel files"""
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    
    if ext not in file_formats:
        st.error(f"Unsupported file format: {ext}")
        return None
    
    try:
        if ext in ['xlsx', 'xls', 'xlsm', 'xlsb']:
            # For Excel files, read all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheets = {}
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                # Clean column names to be strings
                df.columns = [str(c) for c in df.columns]
                sheets[sheet_name] = df
            return sheets
        else:
            # For CSV files, return as single dataframe
            df = file_formats[ext](uploaded_file)
            # Clean column names to be strings
            df.columns = [str(c) for c in df.columns]
            return {"Sheet1": df}
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Initialize Bedrock
try:
    bedrock_runtime = enable_bedrock()
    llm = BedrockLLMWrapper(bedrock_runtime)
except Exception as e:
    st.error(f"Failed to initialize Bedrock: {str(e)}")
    st.stop()

# File upload
uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file",
    type=list(file_formats.keys()),
    help="Various file formats are supported including Excel files with multiple sheets",
    on_change=clear_submit,
)

# Add analysis method selection
analysis_method = st.sidebar.radio(
    "Choose Analysis Method",
    ("Pandas Agent", "CSVAlchemy"),
    help="Pandas Agent is good for direct queries. CSVAlchemy provides a deeper, more structured analysis."
)

if not uploaded_file:
    if analysis_method == "Pandas Agent":
        st.warning(
            "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
        )

# Load data and handle multiple sheets
sheets_data = None
selected_sheet = None
if uploaded_file:
    sheets_data = load_data(uploaded_file)
    
    if sheets_data:
        if len(sheets_data) > 1:
            # Multiple sheets - let user choose
            sheet_names = list(sheets_data.keys())
            selected_sheet = st.selectbox(
                "Select a sheet to analyze:",
                sheet_names,
                help="Choose which sheet from your Excel file to work with"
            )
            df = sheets_data[selected_sheet]
            
            # Show sheet info
            st.subheader(f"📊 Sheet: {selected_sheet}")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {', '.join([str(col) for col in df.columns.tolist()])}")
            
            # Show data preview
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df.head(10).fillna('[NaN]'), use_container_width=True)
        else:
            # Single sheet
            sheet_name = list(sheets_data.keys())[0]
            df = sheets_data[sheet_name]
            selected_sheet = sheet_name
            
            st.subheader(f"📊 File: {uploaded_file.name}")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {', '.join([str(col) for col in df.columns.tolist()])}")
            
            # Show data preview
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df.head(10).fillna('[NaN]'), use_container_width=True)

# Chat interface
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you analyze this data?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What would you like to know about this data?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not uploaded_file:
        st.info("Please upload a file first.")
        st.stop()

    if analysis_method == "Pandas Agent":
        if sheets_data is None:
            st.info("Please wait for the data to be loaded.")
            st.stop()
        # Create pandas agent
        try:
            # Try the pandas dataframe agent first
            try:
                pandas_df_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True,
                    max_iterations=3,
                )
                
                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = pandas_df_agent.run(prompt, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                    
            except Exception as agent_error:
                # Fallback: Use direct Python REPL approach
                st.warning("Using fallback approach due to agent compatibility issues.")
                
                # Create a simple prompt for the dataframe
                enhanced_prompt = f"""You are a data analyst. You have access to a pandas DataFrame called 'df' with the following columns: {list(df.columns)}.
                
                The user asked: {prompt}
                
                Please write Python code to answer this question using the DataFrame 'df'. 
                IMPORTANT: Store your final answer in a variable called 'result'.
                Return only the Python code, no explanations.
                
                Example:
                result = df[df['First Name'] == 'John']['Age'].values[0]"""
                
                # Get Python code from LLM
                python_code = bedrock_chat(enhanced_prompt)
                
                # Execute the code safely
                try:
                    # Create a safe execution environment
                    local_vars = {'df': df, 'pd': pd, 'np': None, 'plt': None, 'sns': None}
                    
                    # Try to import packages, but don't fail if they're missing
                    try:
                        import numpy as np
                        local_vars['np'] = np
                    except ImportError:
                        pass
                    
                    try:
                        import matplotlib.pyplot as plt
                        local_vars['plt'] = plt
                    except ImportError:
                        pass
                    
                    try:
                        import seaborn as sns
                        local_vars['sns'] = sns
                    except ImportError:
                        pass
                    
                    # Check if seaborn is available and modify the prompt if needed
                    if local_vars['sns'] is None and ('seaborn' in python_code.lower() or 'sns' in python_code.lower()):
                        # Seaborn not available, regenerate code with matplotlib only
                        fallback_prompt = f"""You are a data analyst. You have access to a pandas DataFrame called 'df' with the following columns: {list(df.columns)}.
                        
                        The user asked: {prompt}
                        
                        IMPORTANT: Use only matplotlib.pyplot (plt) for charts, NOT seaborn. Seaborn is not available.
                        Store your final answer in a variable called 'result'.
                        Return only the Python code, no explanations.
                        
                        Example for charts:
                        import matplotlib.pyplot as plt
                        result = df.groupby('Country')['Age'].mean().plot(kind='bar')
                        plt.title('Chart Title')"""
                        
                        python_code = bedrock_chat(fallback_prompt)
                    
                    # Execute the code
                    exec(python_code, {}, local_vars)
                    
                    # Get the result
                    if 'result' in local_vars:
                        result_value = local_vars['result']
                        
                        # Check if result is a matplotlib figure/axes
                        if hasattr(result_value, 'figure') or str(type(result_value)).find('matplotlib') != -1:
                            # It's a chart - display it
                            st.pyplot(result_value.figure if hasattr(result_value, 'figure') else result_value)
                            response = f"**Chart generated successfully!**\n\n**Code executed:**\n```python\n{python_code}\n```"
                        else:
                            # Regular result
                            response = f"**Answer:** {result_value}\n\n**Code executed:**\n```python\n{python_code}\n```"
                    else:
                        # Check if any matplotlib figures were created
                        if 'plt' in local_vars and local_vars['plt']:
                            # Try to get the current figure
                            try:
                                fig = local_vars['plt'].gcf()
                                if fig.get_axes():  # If figure has axes, it's a chart
                                    st.pyplot(fig)
                                    response = f"**Chart generated successfully!**\n\n**Code executed:**\n```python\n{python_code}\n```"
                                else:
                                    response = f"Code executed but no chart or result found. Here's the code:\n```python\n{python_code}\n```"
                            except:
                                response = f"Code executed but no 'result' variable found. Here's the code:\n```python\n{python_code}\n```\n\nPlease try rephrasing your question."
                        else:
                            response = f"Code executed but no 'result' variable found. Here's the code:\n```python\n{python_code}\n```\n\nPlease try rephrasing your question."
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                    
                except Exception as exec_error:
                    error_response = f"Error executing code: {str(exec_error)}\n\nCode attempted:\n```python\n{python_code}\n```\n\n**Note:** If you need seaborn for advanced charts, please install it with: `pip install seaborn`"
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                    st.write(error_response)
                
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})

    elif analysis_method == "CSVAlchemy":
        try:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Process with CSVAlchemy
            config = get_config()
            workbook = load_workbook_from_file(tmp_path)
            encoded_data = encode_workbook(workbook, config)

            # Create a prompt with the structured data
            alchemy_prompt = f"""You are a data analyst. You have been provided with a structured representation of an Excel workbook.

Here is the data:
{json.dumps(encoded_data, indent=2)}

The user's question is: {prompt}

Please answer the user's question based on the provided data. Provide a clear, concise answer. If the data contains tables, you can refer to them by their sheet and name."""

            with st.spinner("Analyzing with CSVAlchemy..."):
                response = bedrock_chat(alchemy_prompt)

                # Extract python code from the response
                code_to_execute = None
                if "```python" in response:
                    code_to_execute = response.split("```python")[1].split("```")[0].strip()

                if code_to_execute:
                    st.write("Here is the code I am going to run:")
                    st.code(code_to_execute, language="python")
                    
                    # Execute the code safely
                    try:
                        # Create a safe execution environment
                        local_vars = {'df': None, 'pd': pd, 'np': None, 'plt': None, 'sns': None, 'data': encoded_data}
                        if 'numpy' in code_to_execute.lower():
                            import numpy as np
                            local_vars['np'] = np
                        if 'matplotlib' in code_to_execute.lower() or 'plt' in code_to_execute.lower():
                            import matplotlib.pyplot as plt
                            local_vars['plt'] = plt
                        if 'seaborn' in code_to_execute.lower() or 'sns' in code_to_execute.lower():
                            import seaborn as sns
                            local_vars['sns'] = sns
                        
                        # Execute the code
                        exec(code_to_execute, {}, local_vars)
                        
                        # Get the result
                        if 'result' in local_vars:
                            result_value = local_vars['result']
                            
                            # Check if result is a matplotlib figure/axes
                            if hasattr(result_value, 'figure') or str(type(result_value)).find('matplotlib') != -1:
                                st.pyplot(result_value.figure if hasattr(result_value, 'figure') else result_value)
                                response_text = "**Chart generated successfully!**"
                            else:
                                response_text = f"**Answer:** {result_value}"
                        else:
                            # Check if any matplotlib figures were created
                            if 'plt' in local_vars and local_vars['plt']:
                                try:
                                    fig = local_vars['plt'].gcf()
                                    if fig.get_axes():
                                        st.pyplot(fig)
                                        response_text = "**Chart generated successfully!**"
                                    else:
                                        response_text = "Code executed, but no chart or result was found."
                                except:
                                    response_text = "Code executed, but no result variable was found."
                            else:
                                response_text = "Code executed, but no result variable was found."

                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.write(response_text)

                    except Exception as exec_error:
                        st.error(f"Error executing generated code: {exec_error}")
                else:
                    # If no code block is found, just show the text response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)

        except Exception as e:
            st.error(f"An error occurred with CSVAlchemy: {e}")
        finally:
            # Clean up the temporary file
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

# Additional features
if uploaded_file and sheets_data:
    st.markdown("---")
    st.subheader("📈 Data Analysis Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show Data Info"):
            st.write("**Data Types:**")
            st.write(df.dtypes)
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
    
    with col2:
        if st.button("Show Statistics"):
            st.write("**Numeric Statistics:**")
            st.write(df.describe())
    
    with col3:
        if st.button("Download Processed Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{uploaded_file.name}_processed.csv",
                mime="text/csv"
            )