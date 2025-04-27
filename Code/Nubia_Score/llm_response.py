import os
import subprocess
import json

# Directory containing the LLM models
model_directories = {
    "gemma2_Q2_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/gemma2_Q2_K.gguf",
    "gemma2_Q4_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/gemma2_Q4_0.gguf",
    "gemma2_Q6_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/gemma2_Q6_K.gguf",
    "gemma2_Q8_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/gemma2_Q8_0.gguf",
    "gemma2": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/gemma2.gguf",

    "Llama-3.2-1B_Q2_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q2_K.gguf",
    "Llama-3.2-1B_Q4_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q4_0.gguf",
    "Llama-3.2-1B_Q6_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q6_K.gguf",
    "Llama-3.2-1B_Q8_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q8_0.gguf",
    "Llama-3.2-1B": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-1B.gguf",

    "Llama-3.2-3B_Q2_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q2_K.gguf",
    "Llama-3.2-3B_Q4_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q4_0.gguf",
    "Llama-3.2-3B_Q6_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q6_K.gguf",
    "Llama-3.2-3B_Q8_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q8_0.gguf",
    "Llama-3.2-3B": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/Llama-3.2-3B.gguf",

    "Llama-3.2-8B_Q2_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q2_K.gguf",
    "Llama-3.2-8B_Q4_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q4_0.gguf",
    "Llama-3.2-8B_Q6_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q6_K.gguf",
    "Llama-3.2-8B_Q8_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q8_0.gguf",
    "Llama-3.2-8B": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/llama-3.2-8B.gguf",

    "phi-3_Q2_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/phi-3_Q2_K.gguf",
    "phi-3_Q4_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/phi-3_Q4_0.gguf",
    "phi-3_Q6_K": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/phi-3_Q6_K.gguf",
    "phi-3_Q8_0": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/phi-3_Q8_0.gguf",
    "phi-3": "/work/mahsam/IJCAI/llama.cpp/models/uniform_precision_models/phi-3.gguf"
}

# llama-cli path
llama_cli_path = "/work/mahsam/IJCAI/llama.cpp/llama-cli"
if not os.path.exists(llama_cli_path):
    raise FileNotFoundError(f"llama-cli not found at {llama_cli_path}")

# Output directory for results
output_base_directory = '/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased'
os.makedirs(output_base_directory, exist_ok=True)

# JSON file for questions and answers
json_file_path = '/work/mahsam/IJCAI/selected_questions.json'

def load_json_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return []

qa_data = load_json_data(json_file_path)

def run_llm(model_path, context, question, max_tokens=50, ctx_size=8192):
    print ("question", question)
    """Runs the LLM model using llama-cli with the provided context."""
    prompt = f"""Based on the following context, provide a detailed and specific answer. Make sure your response fully addresses the information given:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""

    llm_command = [
        llama_cli_path,
        "--model", model_path,
        "--prompt", prompt,
        "--n_predict", str(max_tokens),
        "--ctx_size", str(ctx_size)
    ]

    print(f"Running LLM command: {' '.join(llm_command)}")
    result = subprocess.run(llm_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during LLM inference: {result.stderr}")
        return None

    response = result.stdout.strip()
    response_parts = response.split("Answer:", 1)
    return response_parts[1].strip() if len(response_parts) > 1 else response

# Main processing loop
for i, qa_pair in enumerate(qa_data):
    
    context = qa_pair.get("context", "No context found")
    question = qa_pair.get("question", "No question found")
    correct_answer = qa_pair.get("answers", "No correct answer provided")

    for model_name, model_path in model_directories.items():
        llm_response = run_llm(model_path, context, question)

        if not llm_response:
            print(f"No response from model {model_name} for context index: {i}")
            continue

        cleaned_response = llm_response
        print(f"Model: {model_name}")
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Response: {cleaned_response}")
        print(f"Correct Answer: {correct_answer}")

        # Create directory for each model
        model_output_directory = os.path.join(output_base_directory, model_name)
        os.makedirs(model_output_directory, exist_ok=True)

        # Output file named after the model inside its directory
        model_output_file = os.path.join(model_output_directory, f"{model_name}.txt")

        # ✅ Updated output format to match the provided sample
        with open(model_output_file, 'a') as file:
            file.write(f"Model: {model_name}\n")
            file.write(f"Context: {context}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Response: {cleaned_response}\n")
            file.write(f"Correct Answer: {correct_answer}\n\n")

        print(f"Results saved to {model_output_file}")