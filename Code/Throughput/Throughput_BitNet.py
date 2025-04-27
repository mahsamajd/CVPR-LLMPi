import subprocess
import csv

# List of model paths
models = [
    "/home/icas/Desktop/BitNet/models/bitnet_b1_58-large/ggml-model-i2_s.gguf",
    "/home/icas/Desktop/BitNet/models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf"
]

# Output CSV file
output_file = "model_eval_times_2.csv"
num_iterations = 10  # Number of iterations per model

# Open the CSV file and write the header
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Avg Tokens Per Second"])

# Loop through each model
for model in models:
    print(f"Running model: {model} for {num_iterations} iterations...")

    tokens_per_second_list = []  # Store results for averaging

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")

        command = [
            "python",
            "/home/icas/Desktop/BitNet/run_inference.py",
            "-m", model,
            "-p", "What are the benefits of deploying AI models on edge devices? Provide a concise response.",
        ]

        process = subprocess.run(command, capture_output=True, text=True)

        # Combine stdout and stderr
        output = process.stdout + process.stderr  
        print(output.split("\n")[-3])  # Debugging: Show the relevant output line

        try:
            # Extract tokens per second
            tokens_per_second = float(output.split("\n")[-3].split(",")[1].split("tokens")[0].strip())
            tokens_per_second_list.append(tokens_per_second)
            print(f"Tokens per second: {tokens_per_second}")
        except (IndexError, ValueError) as e:
            print(f"Error extracting tokens per second: {e}")
    
    # Compute average tokens per second
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else "N/A"
    
    # Save the final result in CSV
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model, avg_tokens_per_second])

print(f"Evaluation complete. Results saved to {output_file}")
