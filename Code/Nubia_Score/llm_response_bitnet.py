import os
import subprocess
import json
import sys

# BitNet inference script path
bitnet_script_path = "/work/mahsam/IJCAI/BitNet/run_inference.py"
if not os.path.exists(bitnet_script_path):
    raise FileNotFoundError(f"Error: run_inference.py not found at {bitnet_script_path}")

# Model directories (Updated as requested)
model_directories = {
    "bitnet_b1_58_large": '/work/mahsam/IJCAI/BitNet/models/bitnet_b1_58-large/ggml-model-i2_s.gguf',
    "llama3_8B": '/work/mahsam/IJCAI/BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf'
}

# Output directory for results
output_base_directory = "/work/mahsam/IJCAI/llm_response_without_whisper_Bitnet_results_CVPR_contextbased"
os.makedirs(output_base_directory, exist_ok=True)

# JSON file containing questions and answers
json_file_path = '/work/mahsam/IJCAI/selected_questions.json'

def load_json_data(json_path):
    """Load JSON data from the given path."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return []

qa_data = load_json_data(json_file_path)

def run_bitnet_model(model_path, context, question, max_tokens=50, ctx_size=8192):
    """Runs BitNet model using run_inference.py script with given parameters."""
    prompt = f"""Based on the following context, provide a detailed and specific answer. Make sure your response fully addresses the information given:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""

    llm_command = [
        sys.executable,
        bitnet_script_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0",
        "-c", str(ctx_size)
    ]

    print(f"Running BitNet model with command: {' '.join(llm_command)}")
    result = subprocess.run(llm_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during BitNet inference: {result.stderr}")
        return None

    response = result.stdout.strip()
    response_parts = response.split("Answer:", 1)
    return response_parts[1].strip() if len(response_parts) > 1 else response

# Main processing loop for all questions and models
for i, qa_pair in enumerate(qa_data):
    context = qa_pair.get("context", "No context found")
    question = qa_pair.get("question", "No question found")
    correct_answer = qa_pair.get("answers", "No correct answer provided")

    for model_name, model_path in model_directories.items():
        llm_response = run_bitnet_model(model_path, context, question)

        if not llm_response:
            print(f"No response from model {model_name} for context index: {i}")
            continue

        # Logging results
        print(f"Model: {model_name}")
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Response: {llm_response}")
        print(f"Correct Answer: {correct_answer}\n")

        # Create output directory for each model
        model_output_directory = os.path.join(output_base_directory, model_name)
        os.makedirs(model_output_directory, exist_ok=True)

        # Output file for each model
        model_output_file = os.path.join(model_output_directory, f"{model_name}.txt")

        with open(model_output_file, 'a') as file:
            file.write(f"Model: {model_name}\n")
            file.write(f"Context: {context}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Response: {llm_response}\n")
            file.write(f"Correct Answer: {correct_answer}\n\n")

        print(f"Results saved to {model_output_file}")
