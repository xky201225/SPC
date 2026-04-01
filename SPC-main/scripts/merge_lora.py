import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

def merge_lora_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spc_main_dir = os.path.dirname(current_dir)
    
    base_model_path = os.path.join(spc_main_dir, "check", "SPC-Critic-2")
    lora_path = os.path.join(spc_main_dir, "saved_models", "SPC-Critic-3-Medical-LoRA")
    output_path = os.path.join(spc_main_dir, "check", "SPC-Critic-3-Medical")
    
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_path}")
    print(f"Output path: {output_path}")
    
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("\n✅ LoRA model merged successfully!")
    print(f"Merged model saved to: {output_path}")

if __name__ == "__main__":
    merge_lora_model()
