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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

class RedisMetricsCallback(TrainerCallback):
    """
    Коллбек для перехвата метрик во время обучения и их отправки в Redis Pub/Sub.
    Обеспечивает real-time мониторинг на фронтенде через WebSockets.
    """
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
    """
    Форматирует стандартизированную запись датасета в единый промпт для модели.
    """
    system_prompt = f"System: {example['system']}\n" if example.get('system') else ""
    user_input = f"Input: {example['input']}\n" if example.get('input') else ""
    return f"{system_prompt}User: {example['instruction']}\n{user_input}Assistant: {example['output']}"

def train_local(job_id: int, params: dict, redis_client, dataset_path: str = "processed_data/dataset.jsonl"):
    """
    Изолированный процесс локального обучения (SFT) с использованием LoRA и 4-bit квантования.
    """
    model_name = params.get("model_name", "meta-llama/Llama-2-7b-hf")
    output_dir = f"./results/task_{job_id}"
    adapter_dir = f"./adapters/task_{job_id}"
    
    # Отправка начального статуса
    redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Initializing", "step": 0}))
    
    try:
        # 1. Загрузка датасета (ожидается стандартизированный JSONL от DatasetParser)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Датасет не найден по пути: {dataset_path}")
            
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        # 2. Настройка квантования (4-bit QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # 3. Загрузка модели и токенизатора
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

        # 4. Настройка PEFT (LoRA)
        peft_config = LoraConfig(
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.1),
            r=params.get("lora_r", 8),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"] # Специфично для LLaMA-подобных архитектур
        )
        model = get_peft_model(model, peft_config)

        # 5. Параметры обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=params.get("epochs", 3),
            per_device_train_batch_size=params.get("batch_size", 4),
            gradient_accumulation_steps=params.get("grad_accum_steps", 1),
            optim="paged_adamw_32bit",
            save_steps=params.get("save_steps", 50),
            logging_steps=1, # Логируем каждый шаг для плавного графика
            learning_rate=params.get("lr", 2e-4),
            weight_decay=0.001,
            fp16=False,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none" # Отключаем wandb/tensorboard, так как используем свой RedisCallback
        )

        # 6. Инициализация SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=params.get("max_seq_length", 1024),
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=lambda x: [format_prompt(item) for item in x] if isinstance(x, list) else format_prompt(x),
            callbacks=[RedisMetricsCallback(job_id, redis_client)]
        )

        # 7. Запуск обучения
        trainer.train()

        # 8. Сохранение только весов LoRA (adapter_model.bin)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        
        # Отправка финального статуса
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Completed"}))

    except Exception as e:
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({"status": "Failed", "error": str(e)}))
        raise e

    finally:
        # Очистка памяти для предотвращения OOM при последующих запусках
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()