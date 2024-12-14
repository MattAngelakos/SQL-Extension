import subprocess
import difflib

def run_script(script_name):
    try:
        result = subprocess.run(
            ["python3", script_name], capture_output=True, text=True, check=True
        )
        return "\n".join(
            line for line in result.stdout.splitlines() if line.startswith("+") or line.startswith("|") or line.startswith("-")
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e.stderr}")
        return None

def compare_outputs(output1, output2):
    diff = difflib.unified_diff(
        output1.splitlines(), output2.splitlines(),
        lineterm='',
        fromfile='sql.py Output',
        tofile='_generated.py Output'
    )
    return "\n".join(diff)

def main():
    sql_output = run_script("sql.py")
    generated_output = run_script("_generated.py")
    if sql_output is None or generated_output is None:
        print("Error: Could not retrieve outputs from one or both scripts.")
        return
    diff = compare_outputs(sql_output, generated_output)
    if diff:
        print("Differences found between sql.py and _generated.py:")
        print(diff)
    else:
        print("No differences found. Both scripts produce the same output.")

if __name__ == "__main__":
    main()
