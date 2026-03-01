import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = r"C:\Users\smrut\Desktop\projects\Project 1 -ASTAS Autonomous Surveillance & Threat Assessment System\LLM_Enigne\Finetunning_Model\data\processed\train.jsonl"
OUTPUT_DIR = "./qlora-qwen2.5-1.5b-astas"
MAX_LENGTH = 512


# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================================
# 4-BIT CONFIG
# ==========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# ==========================================================
# LOAD BASE MODEL
# ==========================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()


# ==========================================================
# LORA CONFIG (Optimized for Qwen2.5)
# ==========================================================

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset:", DATA_PATH)
dataset = load_dataset("json", data_files=DATA_PATH)["train"]


# ==========================================================
# FORMAT FUNCTION (Assistant-only loss)
# ==========================================================

def format_sample(example):

    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        formatted_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenized["input_ids"].copy()

    # Find assistant start
    assistant_tag = "<|assistant|>"
    assistant_start = formatted_text.rfind(assistant_tag)

    if assistant_start == -1:
        labels = [-100] * len(labels)
    else:
        assistant_text = formatted_text[assistant_start:]
        assistant_tokens = tokenizer(
            assistant_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )["input_ids"]

        mask_length = len(labels) - len(assistant_tokens)
        labels[:mask_length] = [-100] * mask_length

    tokenized["labels"] = labels
    return tokenized


dataset = dataset.map(
    format_sample,
    remove_columns=dataset.column_names,
)


# ==========================================================
# TRAINING ARGS (Stable Setup)
# ==========================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    report_to="none",
    save_strategy="epoch",
)


# ==========================================================
# DATA COLLATOR
# ==========================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
)


# ==========================================================
# TRAINER
# ==========================================================

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)


# ==========================================================
# TRAIN
# ==========================================================

trainer.train()


# ==========================================================
# SAVE ADAPTER
# ==========================================================

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ QLoRA fine-tuning complete.")