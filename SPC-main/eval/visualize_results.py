import os
import json
import matplotlib.pyplot as plt
import numpy as np


def parse_critique_file(file_path, data_type='prm'):
    with open(file_path, "r", encoding="utf-8") as f:
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
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            start_index += len(start_tag)
            result = response[start_index:end_index]
            if result.lower() == "correct":
                correctness_label = 1
                data_valid_num += 1
            elif result.lower() == "incorrect":
                correctness_label = -1
                data_valid_num += 1
        
        if correctness_label != 0:
            human_label_key = f"{data_type}_human_label"
            if human_label_key not in data:
                for key in data.keys():
                    if key.endswith("_human_label"):
                        human_label_key = key
                        break
            
            if human_label_key in data:
                if data[human_label_key] == 1:
                    if correctness_label == 1:
                        correct_step_acc.append(1)
                    else:
                        correct_step_acc.append(0)
                elif data[human_label_key] == -1:
                    if correctness_label == -1:
                        erroneous_step_acc.append(1)
                    else:
                        erroneous_step_acc.append(0)
    
    results = {
        "total": data_total_num,
        "valid_ratio": data_valid_num / data_total_num if data_total_num > 0 else 0,
        "correct_acc": sum(correct_step_acc) / len(correct_step_acc) if correct_step_acc else 0,
        "error_acc": sum(erroneous_step_acc) / len(erroneous_step_acc) if erroneous_step_acc else 0,
    }
    results["avg_acc"] = (results["correct_acc"] + results["error_acc"]) / 2
    results["harmonic_mean"] = (2 * results["correct_acc"] * results["error_acc"] / 
                                (results["correct_acc"] + results["error_acc"])) if (results["correct_acc"] + results["error_acc"]) > 0 else 0
    
    return results


def find_all_critique_files(base_path):
    critique_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith("critique_") and file.endswith(".json"):
                critique_files.append(os.path.join(root, file))
    return critique_files


def parse_filename(filename):
    basename = os.path.basename(filename)
    name = basename.replace("critique_", "").replace(".json", "")
    
    parts = name.split("_")
    
    if len(parts) >= 2:
        if parts[-1] in ["prm", "process_bench", "delta_bench"]:
            dataset_name = "_".join(parts[:-1])
            format_type = parts[-1]
        else:
            dataset_name = name
            format_type = "unknown"
    else:
        dataset_name = name
        format_type = "unknown"
    
    return dataset_name, format_type


def save_single_result_chart(dataset_name, format_type, results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Correct Acc', 'Error Acc', 'Avg Acc', 'Harmonic Mean']
    values = [results['correct_acc'], results['error_acc'], results['avg_acc'], results['harmonic_mean']]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, color=colors, width=0.6)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{dataset_name} - {format_type}\n(Total: {results["total"]}, Valid: {results["valid_ratio"]*100:.1f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{dataset_name}_{format_type}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return output_path


def print_results_table(results_dict):
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    datasets = sorted(set(k[0] for k in results_dict.keys()))
    formats = ["prm", "process_bench", "delta_bench"]
    
    print(f"\n{'Dataset':<15} {'Format':<15} {'Total':<8} {'Valid%':<8} {'Correct':<10} {'Error':<10} {'Avg':<10} {'Harmonic':<10}")
    print("-" * 95)
    
    for ds in datasets:
        for fmt in formats:
            key = (ds, fmt)
            if key in results_dict:
                r = results_dict[key]
                print(f"{ds:<15} {fmt:<15} {r['total']:<8} {r['valid_ratio']*100:>6.1f}% {r['correct_acc']:>8.3f} {r['error_acc']:>8.3f} {r['avg_acc']:>8.3f} {r['harmonic_mean']:>8.3f}")
        print("-" * 95)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spc_main_dir = os.path.dirname(current_dir)
    spc_root_dir = os.path.dirname(spc_main_dir)
    
    critic_path = os.path.join(spc_main_dir, "check", "SPC-Critic-2")
    output_dir = os.path.join(spc_main_dir, "eval", "charts")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Searching for critique files in: {critic_path}")
    critique_files = find_all_critique_files(critic_path)
    
    if not critique_files:
        print("No critique files found. Please check the path.")
        print(f"Expected path: {critic_path}")
        exit(1)
    
    print(f"Found {len(critique_files)} critique files:")
    for f in critique_files:
        print(f"  - {os.path.basename(f)}")
    
    results_dict = {}
    saved_charts = []
    
    print(f"\nGenerating charts in: {output_dir}")
    print("-" * 50)
    
    for file_path in critique_files:
        dataset_name, format_type = parse_filename(file_path)
        print(f"\nProcessing: {os.path.basename(file_path)}")
        print(f"  Dataset: {dataset_name}, Format: {format_type}")
        
        data_type = format_type
        results = parse_critique_file(file_path, data_type=data_type)
        results_dict[(dataset_name, format_type)] = results
        
        print(f"  Total: {results['total']}, Valid: {results['valid_ratio']*100:.1f}%")
        print(f"  Correct Acc: {results['correct_acc']:.3f}, Error Acc: {results['error_acc']:.3f}")
        print(f"  Avg Acc: {results['avg_acc']:.3f}, Harmonic: {results['harmonic_mean']:.3f}")
        
        chart_path = save_single_result_chart(dataset_name, format_type, results, output_dir)
        saved_charts.append(chart_path)
    
    print_results_table(results_dict)
    
    print(f"\n" + "="*50)
    print(f"Generated {len(saved_charts)} charts in: {output_dir}")
    print("="*50)
    
    results_json_path = os.path.join(spc_main_dir, "eval", "test_results_summary.json")
    json_results = {}
    for (ds, fmt), vals in results_dict.items():
        key = f"{ds}_{fmt}"
        json_results[key] = vals
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_json_path}")
