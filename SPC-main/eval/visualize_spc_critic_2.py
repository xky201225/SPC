import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_result(response):
    start_tag = "<Answer>"
    end_tag = "</Answer>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)
    
    if start_index == -1 or end_index == -1:
        return 0
    
    start_index += len(start_tag)
    result = response[start_index:end_index].strip().lower()
    
    if result == "correct":
        return 1
    elif result == "incorrect":
        return -1
    return 0


def calculate_metrics(filepath, label_key="process_bench_human_label"):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f.readlines()]
    
    total = len(lines)
    valid = 0
    correct_step_correct = 0
    correct_step_total = 0
    incorrect_step_correct = 0
    incorrect_step_total = 0
    
    for data in lines:
        response = data.get("response", [""])[0]
        pred = parse_result(response)
        
        if pred == 0:
            continue
        
        valid += 1
        label = data.get(label_key, 0)
        
        if label == 1:
            correct_step_total += 1
            if pred == 1:
                correct_step_correct += 1
        elif label == -1:
            incorrect_step_total += 1
            if pred == -1:
                incorrect_step_correct += 1
    
    if correct_step_total == 0 or incorrect_step_total == 0:
        return None
    
    correct_acc = correct_step_correct / correct_step_total
    error_acc = incorrect_step_correct / incorrect_step_total
    avg_acc = (correct_acc + error_acc) / 2
    harmonic_mean = 2 * correct_acc * error_acc / (correct_acc + error_acc) if (correct_acc + error_acc) > 0 else 0
    
    return {
        "total": total,
        "valid": valid,
        "valid_rate": valid / total if total > 0 else 0,
        "correct_acc": correct_acc,
        "error_acc": error_acc,
        "avg_acc": avg_acc,
        "harmonic_mean": harmonic_mean
    }


def plot_single_chart(results, names, title, output_path):
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    harmonic_means = [results[n]["harmonic_mean"] * 100 for n in names if n in results]
    correct_accs = [results[n]["correct_acc"] * 100 for n in names if n in results]
    error_accs = [results[n]["error_acc"] * 100 for n in names if n in results]
    valid_names = [n for n in names if n in results]
    
    x = np.arange(len(valid_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, correct_accs, width, label='Correct Step Acc', color='#27ae60')
    bars2 = ax.bar(x, error_accs, width, label='Error Step Acc', color='#e74c3c')
    bars3 = ax.bar(x + width, harmonic_means, width, label='Harmonic Mean', color='#3498db')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    for bars, vals in [(bars1, correct_accs), (bars2, error_accs), (bars3, harmonic_means)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


def visualize_spc_critic_2(model_path, model_name="SPC-Critic-2"):
    results = {}
    
    original_datasets = [
        ("ProcessBench", "critique_process_bench.json", "process_bench_human_label"),
        ("PRM800", "critique_prm800.json", "prm_human_label"),
        ("DeltaBench", "critique_delta_bench.json", "delta_bench_human_label"),
    ]
    
    medical_datasets = [
        ("MedQA", "critique_MedQA.json", "process_bench_human_label"),
        ("pubmedqa", "critique_pubmedqa.json", "process_bench_human_label"),
    ]
    
    all_datasets = original_datasets + medical_datasets
    
    for name, filename, label_key in all_datasets:
        filepath = os.path.join(model_path, filename)
        metrics = calculate_metrics(filepath, label_key)
        if metrics:
            results[name] = metrics
            print(f"{name}:")
            print(f"  Total: {metrics['total']}, Valid: {metrics['valid']} ({metrics['valid_rate']*100:.1f}%)")
            print(f"  Correct Step Acc: {metrics['correct_acc']*100:.2f}%")
            print(f"  Error Step Acc: {metrics['error_acc']*100:.2f}%")
            print(f"  Average Acc: {metrics['avg_acc']*100:.2f}%")
            print(f"  Harmonic Mean: {metrics['harmonic_mean']*100:.2f}%")
            print()
    
    if not results:
        print("No results found!")
        return
    
    original_names = ["ProcessBench", "PRM800", "DeltaBench"]
    medical_names = ["MedQA", "pubmedqa"]
    
    original_results = {k: v for k, v in results.items() if k in original_names}
    medical_results = {k: v for k, v in results.items() if k in medical_names}
    
    if original_results:
        plot_single_chart(
            original_results, 
            original_names,
            f'{model_name} - Original Benchmarks',
            os.path.join(model_path, f"{model_name}_original.png")
        )
    
    if medical_results:
        plot_single_chart(
            medical_results, 
            medical_names,
            f'{model_name} - Medical Benchmarks',
            os.path.join(model_path, f"{model_name}_medical.png")
        )
    
    return results


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spc_main_dir = os.path.dirname(current_dir)
    
    model_path = os.path.join(spc_main_dir, "check", "SPC-Critic-2")
    
    print("="*60)
    print("SPC-Critic-2 Visualization")
    print("="*60)
    print(f"Model path: {model_path}")
    print()
    
    if os.path.exists(model_path):
        visualize_spc_critic_2(model_path, "SPC-Critic-2")
    else:
        print(f"Model path not found: {model_path}")
