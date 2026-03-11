"""E2B Code Sandbox with Mercury Coder"""
# Imports
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from e2b_code_interpreter import Sandbox
from langchain.agents.middleware import FilesystemFileSearchMiddleware
import sys
import os
import base64
import pandas as pd
from dotenv import load_dotenv
import time
import warnings


warnings.filterwarnings('ignore')

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-5.1-codex-mini"
)

checkpointer = InMemorySaver()

# Creating the Sandbox
sbx = Sandbox.create(timeout=10*60)



def get_dataset_info(file_path):
    if file_path.endswith == '.csv':
        df = pd.read_csv(file_path, nrows=3)
    
    elif file_path.endswith == '.xlsx' or file_path.endswith == '.xls':
        df = pd.read_excel(file_path, nrows=3)
    
    elif file_path.endswith == '.json':
        df = pd.read_json(file_path, nrows=3)
    
    elif file_path.endswith == '.sql':
        df = pd.read_sql(file_path, nrows=3)

    elif file_path.endswith == '.feather':
        df = pd.read_feather(file_path, nrows=3)
    
    elif file_path.endswith == '.parquet':
        df = pd.read_parquet(file_path, nrows=3)
    
    elif file_path.endswith == '.sas7bdat':
        df = pd.read_sas(file_path, nrows=3)
    
    return f"Columns: {list(df.columns)}\nSample data:\n{df.to_string()}"


### Tools

# Uploads a file
@tool
def upload_file(local_file_name: str):
    """Upload a data file to the E2B sandbox for analysis.
    
    Args:
        local_file_path: Local path to the file (e.g., "IMDB-Movie-Data.csv")
        
    Returns:
        Success message with sandbox_path and dataset_info
    """

    local_file_name = local_file_name.lstrip('/').lstrip('\\')
    local_file_path = f"./data/{local_file_name}"
    
    if not os.path.exists(local_file_path):
        return f"Error: file not found at {local_file_path}"
    
    with open(local_file_path, "rb") as f:
        sandbox_file = sbx.files.write(f"data/{local_file_name}", f)

    return f"File uploaded successfully!\nSandbox path: {sandbox_file.path}"

# Runs python code
@tool
def run_python_code(code: str):
    """Execute Python code in E2B sandbox.
    
    Args:
        code: Valid executable Python code. Do not pass anything else other than python code.
        
    Returns:
        Execution result
    """
    print("Running code in E2B Sandbox...")

    execution = sbx.run_code(code)
    print("Code execution has completed!!!")

    if execution.error:
        return f"Error: {execution.error.name}\nValue: {execution.error.value}"
    
    os.makedirs('images', exist_ok=True)

    output = []
    timestamp = int(time.time())

    output.append(str(execution))

    for idx, result in enumerate(execution.results):
        if result.png:
            filename = f"images/{timestamp}_chart_{idx}.png"
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(result.png))
            
            output.append(f"Chart saved to {filename}")

    return "\n".join(output) if output else "Code executed but no output was returned"

# Prompt
CODE_EXECUTION_PROMPT = """You are a data analysis assistant. You MUST use the available tools to complete tasks.

AVAILABLE TOOLS:
1. glob_search - Search for files in LOCAL filesystem only (searches ./data directory on your machine)
2. upload_file - Upload files from local to sandbox
3. run_python_code - Execute Python code in sandbox environment

FILE LOCATIONS:
- Local files: Use glob_search to find files in ./data directory
- Sandbox files: After upload, files are stored in '/home/user/data/' directory in code environment
- To check sandbox files: Use run_python_code with 'import os; print(os.listdir("/home/user/data/"))'

WORKFLOW - Follow these steps in order:
1. Search for data files using glob_search (for LOCAL file discovery only)
2. Upload file using upload_file (transfers from local to sandbox)
3. ANALYZE THE DATASET FIRST - Use run_python_code to:
   - Check file format (CSV, Excel, JSON, etc.)
   - For CSV/text files: Get shape, columns, data types, first few rows, null values
   - For Excel files: List all sheet names, then analyze each sheet separately
   - Get basic statistics using df.describe()
   - Identify data quality issues
4. PERFORM ANALYSIS - Use run_python_code multiple times to:
   - Clean data if needed
   - Calculate aggregations, groupings, or statistics
   - Answer specific questions from the user
5. CREATE VISUALIZATIONS (if requested) - Use run_python_code to:
   - Generate matplotlib plots with proper titles and labels
   - Use plt.show() to display charts (NOT plt.gcf())

DATASET EXPLORATION TEMPLATE:
For CSV files:
```python
import pandas as pd
df = pd.read_csv('/home/user/data/filename.csv')
print(f"Shape: {df.shape}")
print(f"\\nColumns: {df.columns.tolist()}")
print(f"\\nData Types:\\n{df.dtypes}")
print(f"\\nFirst 5 rows:\\n{df.head()}")
print(f"\\nNull Values:\\n{df.isnull().sum()}")
print(f"\\nBasic Statistics:\\n{df.describe()}")
```

For Excel files:
```python
import pandas as pd
excel_file = pd.ExcelFile('/home/user/data/filename.xlsx')
print(f"Sheet Names: {excel_file.sheet_names}")
for sheet in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    print(f"\\n--- Sheet: {sheet} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
```

VISUALIZATION RULES:
- Only create plots if user explicitly asks for: "plot", "chart", "graph", "visualize", "show", "draw"
- ALWAYS use matplotlib for visualizations
- ALWAYS add meaningful titles, axis labels, and legends
- Use plt.show() to display the plot (NEVER use plt.gcf() or display())
- Common plot types: bar chart, line chart, pie chart, scatter plot, histogram

MULTI-STEP ANALYSIS:
- Use run_python_code tool MULTIPLE times for complex analysis
- Step 1: Always start with dataset info extraction
- Step 2: Perform specific analysis based on user query
- Step 3: Create visualization if requested
- Each step should be a separate tool call with focused code

CRITICAL RULES:
- You MUST call the appropriate tool for each step - do not just think, ACT by calling tools
- NEVER skip the dataset exploration step
- Use run_python_code multiple times rather than one large code block
- All file paths in code must use '/home/user/data/' prefix"""




config = {"configurable": {"thread_id": "test-0001"}}

def ask():
    print("\nChat mode started. Type 'q' or 'quit'")
    # Agent
    code_agent = create_agent(
        model=model,
        tools=[upload_file, run_python_code],
        system_prompt=CODE_EXECUTION_PROMPT,
        checkpointer=checkpointer,
        middleware=[
            FilesystemFileSearchMiddleware(
                root_path="./",
                use_ripgrep=True,
                max_file_size_mb=1000
            )
        ]
    )
    while True:
        query = input("You: ").strip()
        print()
        if query.lower() == 'q' or query.lower() == 'quit':
            print("Exiting chat mode...")
            break
            
        response = code_agent.invoke({'messages': [HumanMessage(query)]}, config=config)
        
        print(response['messages'][-1].text)

if __name__ == "__main__":

    ask()

