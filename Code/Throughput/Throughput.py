import subprocess
import csv
import time

# List of model paths
models = [
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/gemma2.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/gemma2_Q2_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/gemma2_Q4_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/gemma2_Q6_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/gemma2_Q8_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-1B.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q2_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q4_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q6_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-1B_Q8_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-3B.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q2_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q4_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q6_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/Llama-3.2-3B_Q8_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/llama-3.2-8B.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q2_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q4_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q6_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/llama-3.2-8B_Q8_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/phi-3.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/phi-3_Q2_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/phi-3_Q4_0.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/phi-3_Q6_K.gguf",
     "/home/icas/Desktop/llama.cpp/models/uniform_precision_models/phi-3_Q8_0.gguf"
]

# Output CSV file
output_file = "model_eval_times_avg_uniform_precision_CVPR.csv"
# Open the CSV file and write the header
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Average Tokens Per Second"])

# Number of iterations per model
iterations = 10

# Loop through each model
for model in models:
    print(f"Running model: {model}...")

    tokens_per_second_list = []

    for i in range(iterations):
        print(f"  Iteration {i + 1}/{iterations}...")

        command = [
            "/home/icas/Documents/llama.cpp/llama-cli",
            "-m", model,
            "-p", "What are the benefits of deploying AI models on edge devices? Provide a concise response.",
            "-n", "50", "--ctx-size", "4096"
        ]

        process = subprocess.run(command, capture_output=True, text=True)

        # Extract the output and get the tokens per second
        output = process.stdout + process.stderr  # Combine stdout and stderr
       #  print("Full output:", output)
        try:
            tokens_per_second = float(output.split("\n")[-3].split(",")[1].split("tokens")[0].replace(" ",""))
            if tokens_per_second != float('inf'):
                tokens_per_second_list.append(tokens_per_second)
            print(f"    Tokens per second: {tokens_per_second}")
        except (IndexError, ValueError):
            print("    Tokens per second not found.")
    
        # Wait to avoid system overload
        time.sleep(5)

    # Compute average if we have valid data
    if tokens_per_second_list:
        avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
    else:
        avg_tokens_per_second = "N/A"

    print(f"  -> Average Tokens Per Second for {model}: {avg_tokens_per_second}\n")

    # Append the result to the CSV file
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model, avg_tokens_per_second])

print(f"Evaluation complete. Results saved to {output_file}")
