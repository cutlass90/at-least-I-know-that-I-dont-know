from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import evaluate
import numpy as np

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')


def main():
    dataset = load_dataset("mteb/tweet_sentiment_extraction")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenize = lambda sample: tokenizer(sample['text'], padding='max_length', truncation=True)
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))
    for name, dataset in zip(['train', 'test'], [small_train_dataset, small_eval_dataset]):
        for label in range(3):
            n = sum(1 for x in dataset if x['label']==label)/len(dataset)
            print(f'{name} label={label} percentage={n}')

    small_train_dataset = small_train_dataset.map(tokenize, batched=True)
    small_eval_dataset = small_eval_dataset.map(tokenize, batched=True)
    # model = GPT2ForSequenceClassification.from_pretrained('test_trainer/checkpoint-500', num_labels=3)
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3)

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        eval_strategy='epoch',
        output_dir='test_trainer',
        per_device_train_batch_size=1,  # Reduce batch size here
        per_device_eval_batch_size=1,  # Optionally, reduce for evaluation as well
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        logging_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',  # Directory for TensorBoard logs
        logging_steps=1,  # Log every 10 steps
        report_to='tensorboard',  # Report logs to TensorBoard
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics
    )
    train_log = trainer.train()
    print('train_log', train_log)
    val_log = trainer.evaluate()
    print('val_log', val_log)
    print('Done')


if __name__ == '__main__':
    main()
