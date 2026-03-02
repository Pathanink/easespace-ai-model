import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import gc
import os
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class LearningCurveCallback(TrainerCallback):
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_f1_scores = []
        self.epochs = []
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epochs.append(state.epoch)
        
        logs = state.log_history
        
        train_logs = [log for log in logs if 'loss' in log]
        if train_logs:
            self.train_losses.append(train_logs[-1]['loss'])
        
        eval_logs = [log for log in logs if 'eval_loss' in log]
        if eval_logs:
            self.eval_losses.append(eval_logs[-1]['eval_loss'])
            if 'eval_macro_f1' in eval_logs[-1]:
                self.eval_f1_scores.append(eval_logs[-1]['eval_macro_f1'])
            elif 'eval_f1' in eval_logs[-1]:
                self.eval_f1_scores.append(eval_logs[-1]['eval_f1'])

def load_data(filename='dataset.txt'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    df = pd.read_csv(filename, sep='|', header=None, names=['text', 'label'], engine='python')
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    
    label_map = {'neg': 0, 'neu': 1, 'pos': 2}
    df['label_num'] = df['label'].str.lower().map(label_map)
    df = df.dropna(subset=['label_num'])
    df['label_num'] = df['label_num'].astype(int)
    
    print("\nData distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label_num'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_num'])
    
    return train_df, val_df, test_df, label_map

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    macro_f1 = np.mean(f1_per_class)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'f1_neg': f1_per_class[0] if len(f1_per_class) > 0 else 0,
        'f1_neu': f1_per_class[1] if len(f1_per_class) > 1 else 0,
        'f1_pos': f1_per_class[2] if len(f1_per_class) > 2 else 0
    }

def train_model(model_name, train_dataset, val_dataset, output_dir):
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"{'='*60}")
    
    clear_memory()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        problem_type="single_label_classification"
    ).to(device)
    
    learning_curve_callback = LearningCurveCallback()
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints'),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), learning_curve_callback]
    )
    
    trainer.train()
    
    final_save_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Model saved to: {final_save_path}")
    
    return model, tokenizer, trainer, learning_curve_callback

def evaluate_model(model, tokenizer, test_dataset, test_df):
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df['label_num'].values
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = np.mean(f1_per_class)
    
    results = {
        'accuracy': acc,
        'weighted_f1': f1,
        'macro_f1': macro_f1,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'f1_neg': f1_per_class[0],
        'f1_neu': f1_per_class[1],
        'f1_pos': f1_per_class[2],
        'precision_neg': precision_per_class[0],
        'precision_neu': precision_per_class[1],
        'precision_pos': precision_per_class[2],
        'recall_neg': recall_per_class[0],
        'recall_neu': recall_per_class[1],
        'recall_pos': recall_per_class[2],
        'predictions': y_pred,
        'true_labels': y_true
    }
    
    return results

def plot_learning_curves(learning_curve_data, model_name, save_path):
    epochs = learning_curve_data['epochs']
    train_losses = learning_curve_data['train_losses']
    eval_losses = learning_curve_data['eval_losses']
    eval_f1_scores = learning_curve_data['eval_f1_scores']
    
    min_len = min(len(epochs), len(train_losses), len(eval_losses), len(eval_f1_scores))
    
    if min_len == 0:
        print(f"Warning: No data to plot for {model_name}")
        return

    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    eval_losses = eval_losses[:min_len]
    eval_f1_scores = eval_f1_scores[:min_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, train_losses, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=6)
    axes[0].plot(epochs, eval_losses, 's-', label='Validation Loss', color='#e74c3c', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title(f'{model_name}: Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, eval_f1_scores, 'D-', label='Validation Macro F1', color='#2ecc71', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Macro F1 Score', fontsize=11)
    axes[1].set_title(f'{model_name}: Validation Macro F1', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Negative', 'Neutral', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_best_model_metrics(results, model_name, save_path):
    classes = ['Negative', 'Neutral', 'Positive']
    
    data = {
        'Precision': [results['precision_neg'], results['precision_neu'], results['precision_pos']],
        'Recall': [results['recall_neg'], results['recall_neu'], results['recall_pos']],
        'F1': [results['f1_neg'], results['f1_neu'], results['f1_pos']]
    }
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, data['Precision'], width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, data['Recall'], width, label='Recall', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, data['F1'], width, label='F1', color='#2ecc71', alpha=0.8)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name}: Precision, Recall, F1 per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_overall_metrics(results, model_name, save_path):
    metrics = ['Accuracy', 'Weighted F1', 'Macro F1']
    values = [results['accuracy'], results['weighted_f1'], results['macro_f1']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name}: Overall Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    
    MODEL_NAME = "WangchanBERTa"
    MODEL_PATH = "airesearch/wangchanberta-base-att-spm-uncased"
    
    output_dir = f"./wangchanberta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Starting WangchanBERTa training")
        print("=" * 60)
        
        print("\nLoading data...")
        train_df, val_df, test_df, label_map = load_data('dataset.txt')
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        
        train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label_num'].tolist(), tokenizer)
        val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label_num'].tolist(), tokenizer)
        test_dataset = SentimentDataset(test_df['text'].tolist(), test_df['label_num'].tolist(), tokenizer)
        
        model_output_dir = os.path.join(output_dir, MODEL_NAME)
        model, tokenizer, trainer, learning_curve_callback = train_model(
            MODEL_PATH, train_dataset, val_dataset, model_output_dir
        )
        
        learning_curve_data = {
            'epochs': learning_curve_callback.epochs,
            'train_losses': learning_curve_callback.train_losses,
            'eval_losses': learning_curve_callback.eval_losses,
            'eval_f1_scores': learning_curve_callback.eval_f1_scores
        }
        
        print(f"\nEvaluating {MODEL_NAME}...")
        results = evaluate_model(model, tokenizer, test_dataset, test_df)
        
        print(f"\nResults - Accuracy: {results['accuracy']:.4f}, Macro F1: {results['macro_f1']:.4f}")
        
        clear_memory()
        
        print("\nGenerating plots...")
        
        plot_learning_curves(learning_curve_data, MODEL_NAME, os.path.join(output_dir, '0_learning_curves.png'))
        
        plot_confusion_matrix(
            results['true_labels'],
            results['predictions'],
            MODEL_NAME,
            os.path.join(output_dir, '1_confusion_matrix.png')
        )
        
        plot_best_model_metrics(results, MODEL_NAME, os.path.join(output_dir, '2_per_class_metrics.png'))
        
        plot_overall_metrics(results, MODEL_NAME, os.path.join(output_dir, '3_overall_metrics.png'))
        
        print("\nSaving results...")
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'model_path': MODEL_PATH,
            'results': {k: convert_to_serializable(v) 
                       for k, v in results.items() if k not in ['predictions', 'true_labels']},
            'learning_curves': {k: convert_to_serializable(v) for k, v in learning_curve_data.items()}
        }
        
        with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"\n{MODEL_NAME} Results:")
        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  Macro F1:    {results['macro_f1']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")
        print(f"  F1 Negative: {results['f1_neg']:.4f}")
        print(f"  F1 Neutral:  {results['f1_neu']:.4f}")
        print(f"  F1 Positive: {results['f1_pos']:.4f}")
        
        print(f"\nOutput files:")
        print(f"  0_learning_curves.png")
        print(f"  1_confusion_matrix.png")
        print(f"  2_per_class_metrics.png")
        print(f"  3_overall_metrics.png")
        print(f"  summary.json")
        
        print(f"\nOutput folder: {output_dir}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()