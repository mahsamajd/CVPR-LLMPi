import os
import json
from nubia_score import Nubia
import re
# Initialize NUBIA
nubia = Nubia()

def get_nubia_score(text1, text2):
    try:
        if not text1.strip() or not text2.strip() or text2.strip() in ['[end of text]', '_' * len(text2.strip())]:
            print("Skipped scoring due to empty or placeholder response.")
            return None
        return nubia.score(text1, text2)
    except Exception as e:
        print(f"Error calculating NUBIA score: {e}")
        return None

def parse_entry(entry):
    data = {}
    lines = entry.strip().split('\n')

    question, response, correct_answer = None, None, None

    for line in lines:
        line = line.strip()
        if line.startswith('Question:'):
            question = line.split('Question:', 1)[1].strip()
        elif line.startswith('Response:'):
            response = line.split('Response:', 1)[1].strip()
        elif line.startswith('Correct Answer:'):
            correct_answer = line.split('Correct Answer:', 1)[1].strip()

    # Handle correct_answer if it's a dict-like string
    try:
        correct_answer_json = json.loads(correct_answer.replace("'", '"'))
        if isinstance(correct_answer_json, dict):
            correct_answer = ' '.join(correct_answer_json.get('text', []))
    except json.JSONDecodeError:
        pass  # keep original if JSON decoding fails

    if question and response and correct_answer:
        data['Question'] = question
        data['Response'] = response
        data['Correct Answer'] = correct_answer

    return data
# Updated model directories
model_directories = {
    "gemma2": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/gemma2/gemma2.txt",
    "gemma2_Q2_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/gemma2_Q2_K/gemma2_Q2_K.txt",
    "gemma2_Q4_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/gemma2_Q4_0/gemma2_Q4_0.txt",
    "gemma2_Q6_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/gemma2_Q6_K/gemma2_Q6_K.txt",
    "gemma2_Q8_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/gemma2_Q8_0/gemma2_Q8_0.txt",
     # Llama-3.2-1B models
    "Llama-3.2-1B": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-1B/Llama-3.2-1B.txt",
    "Llama-3.2-1B_Q2_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-1B_Q2_K/Llama-3.2-1B_Q2_K.txt",
    "Llama-3.2-1B_Q4_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-1B_Q4_0/Llama-3.2-1B_Q4_0.txt",
    "Llama-3.2-1B_Q6_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-1B_Q6_K/Llama-3.2-1B_Q6_K.txt",
    "Llama-3.2-1B_Q8_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-1B_Q8_0/Llama-3.2-1B_Q8_0.txt",

    # Llama-3.2-3B models
    "Llama-3.2-3B": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-3B/Llama-3.2-3B.txt",
    "Llama-3.2-3B_Q2_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-3B_Q2_K/Llama-3.2-3B_Q2_K.txt",
    "Llama-3.2-3B_Q4_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-3B_Q4_0/Llama-3.2-3B_Q4_0.txt",
    "Llama-3.2-3B_Q6_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-3B_Q6_K/Llama-3.2-3B_Q6_K.txt",
    "Llama-3.2-3B_Q8_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-3B_Q8_0/Llama-3.2-3B_Q8_0.txt",

    # Llama-3.2-8B models
    "Llama-3.2-8B": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-8B/Llama-3.2-8B.txt",
    "Llama-3.2-8B_Q2_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-8B_Q2_K/Llama-3.2-8B_Q2_K.txt",
    "Llama-3.2-8B_Q4_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-8B_Q4_0/Llama-3.2-8B_Q4_0.txt",
    "Llama-3.2-8B_Q6_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-8B_Q6_K/Llama-3.2-8B_Q6_K.txt",
    "Llama-3.2-8B_Q8_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/Llama-3.2-8B_Q8_0/Llama-3.2-8B_Q8_0.txt",

    # Phi-3 models
    "phi-3": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/phi-3/phi-3.txt",
    "phi-3_Q2_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/phi-3_Q2_K/phi-3_Q2_K.txt",
    "phi-3_Q4_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/phi-3_Q4_0/phi-3_Q4_0.txt",
    "phi-3_Q6_K": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/phi-3_Q6_K/phi-3_Q6_K.txt",
    "phi-3_Q8_0": "/work/mahsam/IJCAI/llm_response_without_whisper_results_CVPR_contextbased/phi-3_Q8_0/phi-3_Q8_0.txt",
    "bitnet_b1_58_large": "/work/mahsam/IJCAI/llm_response_without_whisper_Bitnet_results_CVPR_contextbased/bitnet_b1_58_large/bitnet_b1_58_large.txt",
    "llama3_8B": "/work/mahsam/IJCAI/llm_response_without_whisper_Bitnet_results_CVPR_contextbased/llama3_8B/llama3_8B.txt"
    
}

output_base_folder = '/work/mahsam/IJCAI/nubia_scores_results_Combinedtext_CVPR'
os.makedirs(output_base_folder, exist_ok=True)

# Iterate over each model file
for model_name, txt_file_path in model_directories.items():
    output_folder = os.path.join(output_base_folder, model_name)
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f'nubia_scores_{os.path.basename(txt_file_path)}')

    print(f"\nProcessing TXT file: {txt_file_path} for model: {model_name}\n")

    results = []
    processed_count = 0

    with open(txt_file_path, 'r', encoding='utf-8') as txtfile:
        content = txtfile.read()
        #entries = content.split('\n\n')  # Assuming entries are separated by blank lines
        entries = re.split(r'\n(?=Model:)', content)

        for entry in entries:
            try:
                data = parse_entry(entry)
                question = data.get('Question')
                response = data.get('Response')
                correct_answer = data.get('Correct Answer')

                if question and response and correct_answer:
                    combined_text = f"Question: {question}\nCorrect Answer: {correct_answer}"

                    print(f"\nScoring Entry:\nCombined Text: {combined_text}\nResponse: {response}\n")
                    score = get_nubia_score(combined_text, response)

                    if score is not None:
                        result_str = (
                            f"Question: {question}\n"
                            f"Response: {response}\n"
                            f"Correct Answer: {correct_answer}\n"
                            f"NUBIA Score: {score}\n\n"
                        )
                        results.append(result_str)
                        processed_count += 1
                        print(f"Processed question: {question[:30]}... Score: {score:.4f} for model: {model_name}")
                    else:
                        print(f"Skipped scoring for question: {question[:30]}... due to empty response or placeholder.")
                else:
                    print(f"Skipping incomplete entry: {data}")
            except Exception as e:
                print(f"Error processing entry in {txt_file_path}: {e}")

    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(results)
        print(f"\n✅ NUBIA scores saved to {output_file}\n")
    else:
        print(f"⚠️ No valid results to write for {txt_file_path}. Check your input TXT file and NUBIA installation.")

    print(f"✅ Total processed entries for {txt_file_path} in model {model_name}: {processed_count}\n")

print("🎉 Processing complete for all model files.")
