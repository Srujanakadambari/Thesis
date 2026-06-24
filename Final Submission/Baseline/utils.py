def execute_code(code: str):
    """Check if the code is runnable"""
    code = code.strip()
    code = code.replace("```python", "").replace("```", "")
    try:
        exec(code)
    except Exception as e:
        print(f"Execution of code failed: {e}")
        
