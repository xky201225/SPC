import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def collect_prm800_data(data_type='prm'):

    with open(prm800_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    merged_data = []
    print(f"Processing {len(dataset)} data...")
    
    for i, data in enumerate(dataset):

        if data and isinstance(data, dict):
            problem = data["problem"]
            partial_solution = data["partial_solution"]
            correct_step = data["correct_last_step"]
            incorrect_step = data["incorrect_last_step"]

            data_id = str(data["id"])
            merged_data.append({"problem": problem, "partial_solution": partial_solution, "next_step": correct_step, f"{data_type}_human_label": 1, "file_name": data_id+"_c.json"})
            merged_data.append({"problem": problem, "partial_solution": partial_solution, "next_step": incorrect_step, f"{data_type}_human_label": -1, "file_name": data_id+"_i.json"})

        else:
            print(f"line {i} does not contain valid data.")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        merged_data = [data for data in merged_data if data["file_name"] not in existing_files]
        print(f"should process {len(merged_data)} files.")

    return merged_data


def collect_process_bench_data():

    with open(process_bench_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"All {len(dataset)} data...")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        remaining_data = [data for data in dataset if data["file_name"] not in existing_files]
    else:
        remaining_data = dataset

    print(f"should process {len(remaining_data)} files.")
    return remaining_data


def collect_delta_bench_data():

    with open(delta_bench_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"All {len(dataset)} data...")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        remaining_data = [data for data in dataset if data["file_name"] not in existing_files]
    else:
        remaining_data = dataset

    print(f"should process {len(remaining_data)} files.")
    return remaining_data


def collect_medical_data(dataset_name):

    with open(medical_data_paths[dataset_name], "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"All {len(dataset)} data...")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        remaining_data = [data for data in dataset if data["file_name"] not in existing_files]
    else:
        remaining_data = dataset

    print(f"should process {len(remaining_data)} files.")
    return remaining_data

            
def generate_critique_batch(dataset):
    """
    Inputs: problem, partial solution, next step
    Outputs: label and critique
    Vesrion 2
    """

    system = """You are a helpful critic. Given a math Problem, a Partial Solution, the current Last Step of the solution, You need to provide a critique for the correctness of the Last Step.

You need to response a step-by-step analysis:
1. Analyzing the general thought of the Partial Solution. 
2. Critique. You should write a brief critique here. This part should also maintain logical coherence with the summary of the general thought of the Partial Solution.
3. Conclusion. At the end of the response, output <Answer>Correct</Answer> or <Answer>Incorrect</Answer> to represent the correctness of the Last Step."""
    
    prompt_template = """## Problem
{problem}

## Partial Solution
{partial_solution}

## Last Step
{last_step}"""


    llm = LLM(model=critic_path, gpu_memory_utilization=.90, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048, n=1)
    batch_size = 128
    
    all_inputs = []
    for data in dataset:
        if data["partial_solution"] == "":
            data["partial_solution"] = "Let's solve this problem."

        prompt = prompt_template.format(problem=data["problem"], partial_solution=data["partial_solution"], last_step=data["next_step"])

        messages = [
            {"role": "system", "content": system},
            {"role": "user","content": prompt},
        ]
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        all_inputs.append(input)
        
    for i in tqdm.tqdm(range(0, len(all_inputs), batch_size)):
        
        batch_data = dataset[i:i+batch_size]
        batch_inputs = all_inputs[i:i+batch_size]
        batch_outputs = llm.generate(batch_inputs, sampling_params)
        
        for data, input_text, output in zip(batch_data, batch_inputs, batch_outputs):
            output_samples = output.outputs
            critiques = [o.text for o in output_samples]
            save_data = data.copy()
            save_data["system"] = system
            save_data["prompt"] = input_text
            save_data["response"] = critiques
            
            with open(os.path.join(output_file), "a", encoding='utf-8') as f:
                f.write(json.dumps(save_data, ensure_ascii=False)+"\n")
            
    return True
       

def filter_critique(data_type='prm'):
        
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f.readlines()]
    lines = sorted(lines, key=lambda x: x["file_name"])
    
    data_total_num = len(lines)
    data_valid_num = 0
    correct_step_acc = []
    erroneous_step_acc = []
    
    for data in lines:
     
        correctness_label = 0
        response = data["response"][0]
        
        start_tag = "<Answer>"
        end_tag = "</Answer>"
        start_index = response.find(start_tag) + len(start_tag)
        end_index = response.find(end_tag)
        result = response[start_index:end_index]
        if result.lower() == "correct":
            correctness_label = 1
            data_valid_num += 1
        elif result.lower() == "incorrect":
            correctness_label = -1
            data_valid_num += 1
            
        if correctness_label != 0:
            if data[f"{data_type}_human_label"] == 1:
                if correctness_label == 1:
                    correct_step_acc.append(1.0)
                else:
                    correct_step_acc.append(0.0)
            elif data[f"{data_type}_human_label"] == -1:
                if correctness_label == -1:
                    erroneous_step_acc.append(1.0)
                else:
                    erroneous_step_acc.append(0.0)
    
    correct_acc = sum(correct_step_acc) / len(correct_step_acc) * 100 if correct_step_acc else 0.0
    erroneous_acc = sum(erroneous_step_acc) / len(erroneous_step_acc) * 100 if erroneous_step_acc else 0.0
    
    print("="*30)
    print(f"Correct Step Accuracy: {correct_acc:.2f}%")
    print(f"Erroneous Step Accuracy: {erroneous_acc:.2f}%")
    print(f"Average Accuracy: {(correct_acc + erroneous_acc) / 2:.2f}%")
    print(f"Harmonic Mean: {2 * correct_acc * erroneous_acc / (correct_acc + erroneous_acc) if (correct_acc + erroneous_acc) > 0 else 0:.2f}%")
    print("="*30)
    

def filter_process_bench_critique(data_type='process_bench'):
        
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f.readlines()]
    
    data_total_num = len(lines)
    data_valid_num = 0
    stat_dataset_acc = {}
    
    for data in lines:
        correctness_label = 0
        response = data["response"][0]
        
        start_tag = "<Answer>"
        end_tag = "</Answer>"
        start_index = response.find(start_tag) + len(start_tag)
        end_index = response.find(end_tag)
        result = response[start_index:end_index]
        if result.lower() == "correct":
            correctness_label = 1
            data_valid_num += 1
        elif result.lower() == "incorrect":
            correctness_label = -1
            data_valid_num += 1
            
        if correctness_label != 0:
            dataset_type = data.get("source", data.get("data_type", "unknown"))
            if dataset_type not in stat_dataset_acc:
                stat_dataset_acc[dataset_type] = []
                
            if data[f"{data_type}_human_label"] == 1:
                if correctness_label == 1:
                    stat_dataset_acc[dataset_type].append({"type": "correct", "acc": 1.0})
                else:
                    stat_dataset_acc[dataset_type].append({"type": "correct", "acc": 0.0})
            elif data[f"{data_type}_human_label"] == -1:
                if correctness_label == -1:
                    stat_dataset_acc[dataset_type].append({"type": "incorrect", "acc": 1.0})
                else:
                    stat_dataset_acc[dataset_type].append({"type": "incorrect", "acc": 0.0})
    
    print("="*30)
    for dataset_type in stat_dataset_acc:
        correct_step_acc = [d["acc"] for d in stat_dataset_acc[dataset_type] if d["type"] == "correct"]
        erroneous_step_acc = [d["acc"] for d in stat_dataset_acc[dataset_type] if d["type"] == "incorrect"]
        
        correct_acc = sum(correct_step_acc) / len(correct_step_acc) * 100 if correct_step_acc else 0.0
        erroneous_acc = sum(erroneous_step_acc) / len(erroneous_step_acc) * 100 if erroneous_step_acc else 0.0
        
        print(f"Dataset: {dataset_type}")
        print(f"  Correct Step Accuracy: {correct_acc:.2f}%")
        print(f"  Erroneous Step Accuracy: {erroneous_acc:.2f}%")
        print(f"  Average Accuracy: {(correct_acc + erroneous_acc) / 2:.2f}%")
        print(f"  Harmonic Mean: {2 * correct_acc * erroneous_acc / (correct_acc + erroneous_acc) if (correct_acc + erroneous_acc) > 0 else 0:.2f}%")
    print("="*30)


if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    spc_main_dir = os.path.dirname(current_dir)              
    spc_root_dir = os.path.dirname(spc_main_dir)             
    
    tokenizer_path = os.path.join(spc_main_dir, "check", "Qwen2.5-7B-Instruct")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from local path: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    else:
        print("Local tokenizer not found, downloading from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side='left')
    
    prm800_data_path = os.path.join(spc_root_dir, "data", "eval", "prm_eval.json")
    process_bench_data_path = os.path.join(spc_root_dir, "data", "eval", "process_bench_eval.json")
    delta_bench_data_path = os.path.join(spc_root_dir, "data", "eval", "delta_bench_eval.json")
    
    medical_data_paths = {
        "MedQA": os.path.join(spc_root_dir, "data", "eval", "MedQA.json"),
        "pubmedqa": os.path.join(spc_root_dir, "data", "eval", "pubmedqa.json"),
    }
    
    critic_path = os.path.join(spc_main_dir, "check", "SPC-Critic-2")

    print("\n" + "="*50)
    print("Testing SPC-Critic-2 on All Datasets")
    print("="*50)
    
    print("\n" + "="*20 + " ProcessBench " + "="*20)
    output_file = os.path.join(critic_path, "critique_process_bench.json")
    dataset = collect_process_bench_data()
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_process_bench_critique(data_type='process_bench')
 
    print("\n" + "="*20 + " PRM800 " + "="*20)
    output_file = os.path.join(critic_path, "critique_prm800.json")
    dataset = collect_prm800_data()
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_critique(data_type='prm')

    print("\n" + "="*20 + " DeltaBench " + "="*20)
    output_file = os.path.join(critic_path, "critique_delta_bench.json")
    dataset = collect_delta_bench_data()
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_critique(data_type='delta_bench')

    print("\n" + "="*20 + " MedQA " + "="*20)
    output_file = os.path.join(critic_path, "critique_MedQA.json")
    dataset = collect_medical_data("MedQA")
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_process_bench_critique(data_type='process_bench')

    print("\n" + "="*20 + " pubmedqa " + "="*20)
    output_file = os.path.join(critic_path, "critique_pubmedqa.json")
    dataset = collect_medical_data("pubmedqa")
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_process_bench_critique(data_type='process_bench')
