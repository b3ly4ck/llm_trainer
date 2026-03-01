import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import shutil

class RedisMetricsCallback(TrainerCallback):
    def __init__(self, task_id: int, redis_client):
        self.task_id = task_id
        self.redis_client = redis_client
        self.channel_name = f"training_metrics_{self.task_id}"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            payload = json.dumps({
                "status": "Running",
                "loss": logs["loss"],
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0
            })
            self.redis_client.publish(self.channel_name, payload)

def format_prompt(example):
    system_prompt = f"System: {example['system']}\n" if example.get('system') else ""
    user_input = f"Input: {example['input']}\n" if example.get('input') else ""
    return f"{system_prompt}User: {example['instruction']}\n{user_input}Assistant: {example['output']}"

def train_local(job_id: int, params: dict, redis_client, dataset_path: str):
    model_name = params.get("model_name")
    output_dir = f"./results/task_{job_id}"
    adapter_dir = f"./adapters/task_{job_id}_temp"
    final_model_dir = f"./models/task_{job_id}_ready"
    
    redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Initializing", "step": 0}))
    
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=params.get("epochs", 3),
            per_device_train_batch_size=params.get("batch_size", 4),
            optim="paged_adamw_32bit",
            logging_steps=1,
            learning_rate=params.get("lr", 2e-4),
            fp16=False,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=lambda x: [format_prompt(item) for item in x] if isinstance(x, list) else format_prompt(x),
            callbacks=[RedisMetricsCallback(job_id, redis_client)]
        )

        # 1. Обучаем
        trainer.train()

        # 2. Сохраняем временный адаптер
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Merging Weights"}))
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # 3. Полностью чистим память от квантованной модели
        del model
        del trainer
        torch.cuda.empty_cache()

        # 4. Загружаем чистую базовую модель и вклеиваем адаптер (Бесшовность)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model_to_merge = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = model_to_merge.merge_and_unload()

        # 5. Сохраняем финальную готовую модель и удаляем мусор
        merged_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        shutil.rmtree(adapter_dir)
        
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Completed"}))

    except Exception as e:
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Failed", "error": str(e)}))
        raise e
    finally:
        if 'base_model' in locals(): del base_model
        if 'merged_model' in locals(): del merged_model
        torch.cuda.empty_cache()