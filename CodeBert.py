import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load CodeBERT tokenizer and model
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load dataset
df = pd.read_csv("stegano_java_dataset.csv")

# Tokenize Java code
def tokenize_function(examples):
    return tokenizer(examples["code"], padding="max_length", truncation=True)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True)

# Split dataset (80% training, 20% testing)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./stegano_codebert")
tokenizer.save_pretrained("./stegano_codebert")
