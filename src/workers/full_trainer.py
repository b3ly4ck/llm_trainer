import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer

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
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "learning_rate": logs.get("learning_rate", 0)
            })
            self.redis_client.publish(self.channel_name, payload)

def format_prompt(example):
    system_prompt = f"System: {example['system']}\n" if example.get('system') else ""
    user_input = f"Input: {example['input']}\n" if example.get('input') else ""
    return f"{system_prompt}User: {example['instruction']}\n{user_input}Assistant: {example['output']}"

def train_full(job_id: int, params: dict, redis_client, dataset_path: str):
    """
    Полное обучение (FFT) без квантования и LoRA. Обновляются все веса модели.
    Требует огромного количества видеопамяти.
    """
    model_name = params.get("model_name", "meta-llama/Llama-2-7b-hf")
    output_dir = f"./results/task_{job_id}_full"
    
    redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Initializing Full FT", "step": 0}))
    
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
            
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Загрузка модели в bfloat16 (без BitsAndBytesConfig)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=params.get("epochs", 3),
            per_device_train_batch_size=params.get("batch_size", 2), # Меньше батч, так как жрет память
            gradient_accumulation_steps=params.get("grad_accum_steps", 4),
            optim="adamw_torch",
            save_steps=params.get("save_steps", 50),
            logging_steps=1,
            learning_rate=params.get("lr", 1e-5), # Строго низкий LR для полного обучения
            weight_decay=0.01,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            max_grad_norm=1.0,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            max_seq_length=params.get("max_seq_length", 1024),
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=lambda x: [format_prompt(item) for item in x] if isinstance(x, list) else format_prompt(x),
            callbacks=[RedisMetricsCallback(job_id, redis_client)]
        )

        trainer.train()
        
        # Сохраняем всю модель целиком
        trainer.model.save_pretrained(f"./models/task_{job_id}_full_model")
        tokenizer.save_pretrained(f"./models/task_{job_id}_full_model")
        
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Completed"}))

    except Exception as e:
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Failed", "error": str(e)}))
        raise e

    finally:
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()