# train.py

import argparse
import random
import os
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
)
from sklearn.metrics import f1_score, precision_score, recall_score

def set_seed(seed_value: int):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """
    Computes performance metrics for multi-label classification.
    """
    logits = pred.predictions
    labels = pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    # Apply a threshold to get binary predictions
    preds = (probs >= 0.5).astype(int)
    
    # Compute micro-averaged metrics
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    prec_micro = precision_score(labels, preds, average="micro", zero_division=0)
    rec_micro = recall_score(labels, preds, average="micro", zero_division=0)
    
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
    }

def main(args):
    """
    Main function to orchestrate the model training pipeline.
    """
    # 1. Set Seed and Device
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Dataset
    print("Loading GoEmotions dataset...")
    ds = load_dataset("mrm8488/goemotions")

    # 3. Preprocessing
    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    num_labels = len(emotion_columns)

    def create_labels_function(examples):
        labels = [[float(examples[col][i]) for col in emotion_columns] for i in range(len(examples[emotion_columns[0]]))]
        return {"labels": labels}

    print("Tokenizing and preparing dataset...")
    tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_ds = tokenized_ds.map(create_labels_function, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Split Data
    train_test_split = tokenized_ds["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # 5. Load Model
    print("Loading pre-trained model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        problem_type="multi_label_classification",
        num_labels=num_labels,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Set Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        push_to_hub=False,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8. Train and Evaluate
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    print("Evaluating final model...")
    metrics = trainer.evaluate()
    print("Final Evaluation Metrics:")
    print(metrics)

    # 9. Save Model
    print(f"Saving best model and tokenizer to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a transformers model for multi-label text classification.")
    
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the base model from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="./emotion_model_trained", help="Directory to save the trained model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    main(args)