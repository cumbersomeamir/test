# Importing the relevant libraries
#pip3 install pandas datasets transformers 
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, DeepSpeedPlugin, Trainer, TrainingArguments

# Reading the file
data = pd.read_excel("MedQuad dataset test.xlsx")

# Convert the pandas DataFrame to Hugging Face's Dataset
hf_dataset = Dataset.from_pandas(data)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def tokenize_function(examples):
    tokenized_prompt = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    tokenized_completion = tokenizer(examples["completion"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    return {"input_ids": tokenized_prompt["input_ids"], "attention_mask": tokenized_prompt["attention_mask"], "labels": tokenized_completion["input_ids"]}

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
print("The tokenized dataset is ", tokenized_dataset)

# Load the pre-trained GPT-Neo 1.3B model
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=100,
    evaluation_strategy="no",
    deepspeed="./ds_config.json",  # Use DeepSpeed configuration
)

    
    # Create a Trainer instance
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("finetuned-gptj-6B")

model = AutoModelForCausalLM.from_pretrained("finetuned-gptj-6B")

# Saving the Model on Hugging Face
token = "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"
model.push_to_hub("Amirkid/finetune-llama-test", use_auth_token=token)
