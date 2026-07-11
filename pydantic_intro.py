import os
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel  # 🧠 Corrected import name!

class ProcessingSummary(BaseModel):
    files_processed: list[str] = Field(description="List of files successfully cleaned.")
    actions_taken: list[str] = Field(description="Detailed cleanup actions applied per file.")
    status: str = Field(description="Final execution status")

# 🔄 Update the string inside the object instance to version 2.5
agent = Agent(
    model=GoogleModel('gemini-2.5-flash'),  
    output_type=ProcessingSummary,  
    system_prompt=(
        "You are an expert Data Engineering Agent. Your job is to inspect the source directory, "
        "identify data files, analyze their flaws (like missing values or bad formatting), "
        "and use your pandas tool to clean them. Once cleaned, summarize your work."
    )
)

# --- NATIVE FILE SYSTEM TOOLS ---

@agent.tool_plain
def list_source_files() -> list[str]:
    """Lists all files available in the target source directory."""
    source_dir = "./source_data"
    if not os.path.exists(source_dir):
        return []
    return os.listdir(source_dir)

@agent.tool_plain
def read_file_head(filename: str) -> str:
    """Reads the first 5 lines of a file to understand its schema and flaws."""
    path = os.path.join("./source_data", filename)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(path, nrows=5)
            return df.to_string()
        elif filename.endswith('.json'):
            df = pd.read_json(path, lines=True if filename.endswith('jsonl') else False).head(5)
            return df.to_string()
        return "Unsupported file type preview."
    except Exception as e:
        return f"Error reading file: {str(e)}"

@agent.tool_plain
def clean_and_save_csv(filename: str, fill_na_columns: list[str], drop_na_columns: list[str]) -> str:
    """Cleans a CSV file by filling NaNs with defaults or dropping rows, then saves to output."""
    src_path = os.path.join("./source_data", filename)
    out_dir = "./cleaned_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    try:
        df = pd.read_csv(src_path)
        
        for col in fill_na_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(0)
                    
        if drop_na_columns:
            valid_cols = [c for c in drop_na_columns if c in df.columns]
            df = df.dropna(subset=valid_cols)

        df.to_csv(out_path, index=False)
        return f"Successfully cleaned {filename} and saved to {out_path}."
    except Exception as e:
        return f"Failed to clean file due to error: {str(e)}"

# --- ISOLATED EXECUTION ENVIRONMENT ---

if __name__ == "__main__":
    # Safety Check: Prevent messy traceback logs if the API key is missing
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("\n🛑 STOP: Missing API Key Configuration!")
        print("Please export your API key in your terminal before running the script:")
        print('👉 export OPENAI_API_KEY="your_key_here"\n')
        exit(1)

    # Set up simulated environment directories
    os.makedirs("./source_data", exist_ok=True)
    dirty_data = {
        "transaction_id": [101, 102, 103, 104],
        "customer_name": ["Austin", None, "Charlie", "Delta"],
        "amount": [250.0, 45.5, None, 12.0]
    }
    pd.DataFrame(dirty_data).to_csv("./source_data/user_transactions.csv", index=False)

    print("🚀 Triggering Data Engineering Agent...")
    
    # Execute the agent workflow
    result = agent.run_sync(
        "Look into the source folder, check what files are there, inspect their structural data, "
        "and clean any columns that have missing or empty values. Save the output."
    )
    
    print("\n--- Agent Execution Summary ---")
    # 🔄 Accessing data via .output to align with the modern spec
    print(f"Status: {result.output.status}")
    print(f"Processed: {result.output.files_processed}")
    print("Actions taken:")
    for action in result.output.actions_taken:
        print(f" - {action}")