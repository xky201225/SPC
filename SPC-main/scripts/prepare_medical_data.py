import json
import gzip
import os
import glob
from tqdm import tqdm

CRITIC_PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful critic. Given a Problem, a Partial Solution, the current Last Step of the solution, You need to provide a critique for the correctness of the Last Step.

You need to response a step-by-step analysis:
1. Analyzing the general thought of the Partial Solution. 
2. Critique. You should write a brief critique here. This part should also maintain logical coherence with the summary of the general thought of the Partial Solution.
3. Conclusion. At the end of the response, output <Answer>Correct</Answer> or <Answer>Incorrect</Answer> to represent the correctness of the Last Step.<|im_end|>
<|im_start|>user
## Problem
{problem}

## Partial Solution
{partial_solution}

## Last Step
{last_step}<|im_end|>
<|im_start|>assistant
"""

def generate_critique_answer(is_correct: bool, critique_reason: str = None) -> str:
    label = "Correct" if is_correct else "Incorrect"
    if critique_reason:
        return f"## Analysis\n{critique_reason}\n\n## Conclusion\nThe correctness of the last step is <Answer>{label}</Answer>.<|im_end|>\n"
    
    if is_correct:
        return f"## Analysis\nThe last step follows logically from the partial solution and correctly addresses the problem.\n\n## Conclusion\nThe correctness of the last step is <Answer>{label}</Answer>.<|im_end|>\n"
    else:
        return f"## Analysis\nThe last step contains an error and does not correctly follow from the partial solution.\n\n## Conclusion\nThe correctness of the last step is <Answer>{label}</Answer>.<|im_end|>\n"

def convert_prm_record(record: dict) -> list:
    results = []
    problem = record.get('problem', '')
    partial_solution = record.get('partial_solution', '')
    correct_step = record.get('correct_last_step', '')
    incorrect_step = record.get('incorrect_last_step', '')
    
    correct_prompt = CRITIC_PROMPT_TEMPLATE.format(
        problem=problem,
        partial_solution=partial_solution,
        last_step=correct_step
    )
    correct_answer = generate_critique_answer(True)
    results.append({
        'prompt': correct_prompt,
        'answer': correct_answer,
        'reward': 1.0,
        'weight': 1.0,
        'source': 'prm_correct',
        'unique_problem_id': record.get('unique_problem_id', '')
    })
    
    incorrect_prompt = CRITIC_PROMPT_TEMPLATE.format(
        problem=problem,
        partial_solution=partial_solution,
        last_step=incorrect_step
    )
    incorrect_answer = generate_critique_answer(False, record.get('incorrect_step_critique', ''))
    results.append({
        'prompt': incorrect_prompt,
        'answer': incorrect_answer,
        'reward': 1.0,
        'weight': 1.0,
        'source': 'prm_incorrect',
        'unique_problem_id': record.get('unique_problem_id', '')
    })
    
    return results

def convert_process_bench_record(record: dict) -> dict:
    problem = record.get('problem', '')
    partial_solution = record.get('partial_solution', '')
    next_step = record.get('next_step', '')
    label = record.get('process_bench_human_label', 1)
    is_correct = (label == 1)
    
    prompt = CRITIC_PROMPT_TEMPLATE.format(
        problem=problem,
        partial_solution=partial_solution,
        last_step=next_step
    )
    answer = generate_critique_answer(is_correct)
    
    return {
        'prompt': prompt,
        'answer': answer,
        'reward': 1.0,
        'weight': 1.0,
        'source': 'process_bench',
        'unique_problem_id': record.get('file_name', record.get('source_id', ''))
    }

def read_jsonl_gz(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def read_jsonl(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def process_and_write_ddxplus_data(base_dir: str, output_file, stats: dict):
    merged_dir = os.path.join(base_dir, 'merged_ddxplus')
    
    prm_files = sorted(glob.glob(os.path.join(merged_dir, 'all_ddxplus_prm_train.semantic.part*.jsonl.gz')))
    process_files = sorted(glob.glob(os.path.join(merged_dir, 'all_ddxplus_process_bench_train.semantic.part*.jsonl.gz')))
    
    count = 0
    
    print("Processing DDXPlus PRM files...")
    for filepath in tqdm(prm_files):
        for record in read_jsonl_gz(filepath):
            for converted in convert_prm_record(record):
                output_file.write(json.dumps(converted, ensure_ascii=False) + '\n')
                count += 1
    
    print("Processing DDXPlus Process Bench files...")
    for filepath in tqdm(process_files):
        for record in read_jsonl_gz(filepath):
            converted = convert_process_bench_record(record)
            output_file.write(json.dumps(converted, ensure_ascii=False) + '\n')
            count += 1
    
    stats['ddxplus'] = count
    print(f"Total DDXPlus records: {count}")
    return count

def process_and_write_other_three_data(base_dir: str, output_file, stats: dict):
    merged_dir = os.path.join(base_dir, 'merged')
    
    prm_file = os.path.join(merged_dir, 'all_3_prm_train.jsonl')
    process_file = os.path.join(merged_dir, 'all_3_process_bench_train.jsonl')
    
    count = 0
    
    print("Processing MedQA/MedMCQA/PubMedQA PRM file...")
    if os.path.exists(prm_file):
        for record in tqdm(read_jsonl(prm_file)):
            for converted in convert_prm_record(record):
                output_file.write(json.dumps(converted, ensure_ascii=False) + '\n')
                count += 1
    
    print("Processing MedQA/MedMCQA/PubMedQA Process Bench file...")
    if os.path.exists(process_file):
        for record in tqdm(read_jsonl(process_file)):
            converted = convert_process_bench_record(record)
            output_file.write(json.dumps(converted, ensure_ascii=False) + '\n')
            count += 1
    
    stats['other_three'] = count
    print(f"Total Other Three records: {count}")
    return count

def main():
    data_dir = r'd:\SPC\data\train'
    output_dir = r'd:\SPC\data\train\converted'
    os.makedirs(output_dir, exist_ok=True)
    
    ddxplus_dir = os.path.join(data_dir, 'medspc_train_ddxplus_v1_semantic')
    other_dir = os.path.join(data_dir, 'merged_otherThree')
    
    stats = {}
    total_count = 0
    
    new_data_file = os.path.join(output_dir, 'medical_train_converted.jsonl.gz')
    print(f"Writing to {new_data_file}...")
    
    with gzip.open(new_data_file, 'wt', encoding='utf-8') as f:
        total_count += process_and_write_ddxplus_data(ddxplus_dir, f, stats)
        total_count += process_and_write_other_three_data(other_dir, f, stats)
    
    print(f"Total new training records: {total_count}")
    
    combined_file = os.path.join(output_dir, 'combined_train_data.jsonl.gz')
    print(f"Writing combined data to {combined_file}...")
    
    old_rl_file = os.path.join(data_dir, 'data_round2_rl_critic.json')
    old_sft_file = os.path.join(data_dir, 'data_round0_sft_critic.json')
    
    combined_count = 0
    
    with gzip.open(combined_file, 'wt', encoding='utf-8') as f:
        if os.path.exists(old_rl_file):
            with open(old_rl_file, 'r', encoding='utf-8') as rf:
                old_rl = json.load(rf)
            for item in old_rl:
                item['source'] = 'old_rl'
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                combined_count += 1
            print(f"Loaded {len(old_rl)} old RL records")
        
        if os.path.exists(old_sft_file):
            with open(old_sft_file, 'r', encoding='utf-8') as rf:
                old_sft = json.load(rf)
            for item in old_sft:
                item['source'] = 'old_sft'
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                combined_count += 1
            print(f"Loaded {len(old_sft)} old SFT records")
        
        print("Copying new data to combined file...")
        with gzip.open(new_data_file, 'rt', encoding='utf-8') as rf:
            for line in rf:
                f.write(line)
                combined_count += 1
    
    print(f"Total combined records: {combined_count}")
    
    new_size = os.path.getsize(new_data_file) / 1024 / 1024
    combined_size = os.path.getsize(combined_file) / 1024 / 1024
    print(f"\nFile sizes:")
    print(f"  medical_train_converted.jsonl.gz: {new_size:.2f} MB")
    print(f"  combined_train_data.jsonl.gz: {combined_size:.2f} MB")

if __name__ == '__main__':
    main()
