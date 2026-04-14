from collections import Counter
from datetime import datetime

import json
from models.model import all_model_names
import numpy as np
import os
import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm 
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LongformerTokenizer
from types import SimpleNamespace
import wandb


def get_criterion(device, balanced=False, class_weights=[]):
    if not balanced:
        class_weights = class_weights.to(device)
        print('class weights ', class_weights, ' class 0 should have lower weight since it has more samples')
        
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()


def to_serializable_list(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [to_serializable_list(x) for x in data]
    else:
        return data  # assume already serializable


def evaluate_model(model, loader, model_name, device, output_file="", tune_threshold=True, best_threshold=0.5, criterion=None, tokenizer=None):
    model.eval()
    thresholds = np.arange(0.1, 1.0, 0.1) if tune_threshold else [best_threshold]
    best_f1 = 0.0
    selected_threshold = best_threshold

    best_metrics = {
        "loss": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }
    best_preds = []
    best_true = []
    best_logits = []
    best_indices = []

    with torch.no_grad():
        for t in thresholds:
            running_loss = 0.0
            running_corrects = 0
            true_labels = []
            predicted_labels = []
            logits_list = []
            indices_list = []

            for batch in loader:
                if "llama" not in model_name:
                    batch = {k: v.to(device) for k, v in batch.items()}
                outputs = run_model_pred(model, batch, model_name, tokenizer=tokenizer)
                logits = outputs.logits
                labels = batch["labels"]

                # Sanity checks
                assert logits.shape[1] == 2, f"Expected 2 output classes, got {logits.shape}"
                assert labels.dtype == torch.long, f"Expected long labels for CrossEntropyLoss, got {labels.dtype}"

                # Compute the loss using criterion
                loss = criterion(logits, labels)
                running_loss += loss.item()

                running_loss, running_corrects, true_labels, predicted_labels = update_running_metrics(
                    loss.item(), outputs, labels,
                    running_loss, running_corrects,
                    true_labels, predicted_labels,
                    threshold=t
                )

                logits_list.append(outputs.logits.detach().cpu())
                if "index" in batch:
                    indices_list.extend(batch["index"].detach().cpu().tolist())

            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            if tune_threshold and f1 > best_f1:
                best_f1 = f1
                selected_threshold = t
                best_preds = predicted_labels
                best_true = true_labels
                best_logits = torch.cat(logits_list).numpy()
                best_indices = indices_list
                best_metrics = {
                    "loss": running_loss / len(best_true),
                    "accuracy": float(running_corrects) / len(best_true),
                    "precision": precision_score(best_true, best_preds, zero_division=0),
                    "recall": recall_score(best_true, best_preds, zero_division=0),
                    "f1": f1,
                }
            if not tune_threshold:
                best_metrics = {
                    "loss": running_loss / len(true_labels),
                    "accuracy": float(running_corrects) / len(true_labels),
                    "precision": precision_score(true_labels, predicted_labels, zero_division=0),
                    "recall": recall_score(true_labels, predicted_labels, zero_division=0),
                    "f1": f1,
                }
            else:
                print(f"[Threshold {t:.1f}] F1 = {f1:.4f}")

    # Write output if needed
    if output_file:
        with open(output_file, 'w') as outfile:
            output_dict = {
                "pred_labels": to_serializable_list(best_preds),
                "y": to_serializable_list(best_true),
                "y_pred": to_serializable_list(best_logits)
            }
            if "index" in output_dict:
                output_dict["index"] = best_indices
            json.dump(output_dict, outfile)
            outfile.write("\n")


    return (
        best_metrics["loss"],
        best_metrics["accuracy"],
        best_metrics["f1"],
        best_metrics["precision"],
        best_metrics["recall"],
        selected_threshold,
    )


def inference_model(model, loader, model_name, device, threshold=0.5, tokenizer=None):
    """
    Run inference on a dataloader without computing metrics.
    
    Args:
        model: Trained model.
        loader: DataLoader (no shuffle, with index).
        model_name: Model identifier.
        device: CUDA or CPU.
        threshold: Probability threshold to convert logits to binary predictions.

    Returns:
        pred_labels: List of binary predictions.
        pred_scores: List of class 1 probabilities.
        pred_indices: List of original data indices.
    """
    model.eval()
    pred_labels = []
    pred_scores = []
    pred_indices = []

    device = model.device  # first device of the model shard

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            #if "llama" not in model_name:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = run_model_pred(model, batch, model_name, tokenizer=tokenizer)
            logits = outputs.logits
            print(f"logits {logits}")

            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1]  # Probability of class 1
            preds = (scores > threshold).long()

            pred_labels.extend(preds.cpu().tolist())
            pred_scores.extend(scores.cpu().tolist())

            if "index" in batch:
                pred_indices.extend(batch["index"].cpu().tolist())
            else:
                # Fallback: just keep count if no indices are provided
                pred_indices.extend(list(range(len(pred_labels) - len(preds), len(pred_labels))))
    print(f"inference returns: \nPREDS {pred_labels} \nSCORES {pred_scores} \nINDICES {pred_indices}")
    return pred_labels, pred_scores, pred_indices

def run_model_pred(model, batch, model_name, tokenizer=None):
    if model_name == "text_only" or "_concat" in model_name:
        # Standard HuggingFace AutoModelForSequenceClassification
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=None,  # loss handled outside
        )
        return outputs  # returns a ModelOutput with .logits

    elif "_embed" in model_name or model_name == "graph_context_all":
        print('in run model pred, at the right place')
        # BERTContextEmb returns raw logits → wrap to mimic HuggingFace output
        logits = model(batch)
        return SimpleNamespace(logits=logits)  # makes it compatible with outputs.logits

    elif "llama" in model_name:

        outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None), max_new_tokens=5)
        decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        print(f"decoded {decoded}")
        # Parse to binary labels (simple rule)
        preds = [1 if "YES" in d.upper() else 0 for d in decoded]

        return SimpleNamespace(logits=preds)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def update_running_metrics(loss, outputs, labels, running_loss, running_corrects, true_labels, predicted_labels, threshold=0.5):
    # Detach loss and add to running loss
    running_loss += loss

    # Get predicted class (argmax over logits)
    #preds = torch.argmax(outputs.logits, dim=1)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = (probs[:, 1] > threshold).long()

    # Count correct predictions
    running_corrects += torch.sum(preds == labels).item()

    # Store labels for metrics
    true_labels.extend(labels.detach().cpu().tolist())
    predicted_labels.extend(preds.detach().cpu().tolist())

    return running_loss, running_corrects, true_labels, predicted_labels


def train(args, model, pretrained_model, train_loader, val_loader, test_loader, criterion, optimizer, device, rank=0):
    num_epochs, model_name, validation, size = args.epochs, args.model_name, args.validation, args.size
    distributed = isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler)

    total_train_samples = torch.tensor(len(train_loader), device=device)
    dist.all_reduce(total_train_samples, op=dist.ReduceOp.SUM)

    if rank == 0:
        print("Total Training Set Size: ", total_train_samples)
        print("Training set size: ", len(train_loader))
        print("Validation set size: ", len(val_loader))
        print("Test set size: ", len(test_loader))
        print("Train: epochs=", num_epochs, ", dataset_name=pointdarret", ", model=", model_name, "pretrained_model=", pretrained_model)

    best_val_f1 = float('-inf')
    best_model = model 
    # Patience is the maximum number of epoch with decaying validation scores we will wait for, before early stopping the training
    patience = 10
    trigger_times = 0

    # Generate a unique model name string to save the model at
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    pretrained_model = pretrained_model.replace("/", "-")
    model_check_path = f"models/checkpoints/{model_name}_{timestamp}_{pretrained_model}_{size}.pt"

    if rank == 0:
        os.makedirs(os.path.dirname(model_check_path), exist_ok=True)
        print(f"Saving model to ", model_check_path)

    best_threshold = 0.5

    # Training loop
    for epoch in range(num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)  # required for shuffling in DDP

        running_loss = float(0)
        running_corrects = 0
        true_labels = []
        predicted_labels = []
        grad_accum_steps = 8
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(progress_bar):
            with autocast():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = run_model_pred(model, batch, model_name)
                labels = batch["labels"]
                logits = outputs.logits
                loss = criterion(logits, labels)

                # 1. Normalize loss for accumulation
                loss = loss / grad_accum_steps
                loss.backward()

            # 2. Step every grad_accum_steps iterations
            if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # 3. Update metrics — use original (unscaled) loss
            unscaled_loss = loss.detach().item() * grad_accum_steps  # to get real loss
            running_loss, running_corrects, true_labels, predicted_labels = update_running_metrics(
                unscaled_loss, outputs, labels,
                running_loss, running_corrects,
                true_labels, predicted_labels,
                threshold=best_threshold
            )

            progress_bar.set_postfix(loss=unscaled_loss)

        # Gather all predictions and labels across GPUs
        gathered_true_labels = [None for _ in range(dist.get_world_size())]
        gathered_predicted_labels = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(gathered_true_labels, true_labels)
        dist.all_gather_object(gathered_predicted_labels, predicted_labels)

        # Flatten lists
        all_true_labels = [label for sublist in gathered_true_labels for label in sublist]
        all_predicted_labels = [pred for sublist in gathered_predicted_labels for pred in sublist]

        # Aggregate loss and corrects across GPUs
        total_loss = torch.tensor(running_loss, device=device)
        total_corrects = torch.tensor(running_corrects, device=device)
        total_samples = torch.tensor(len(true_labels), device=device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_corrects, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        print(f"Total loss {total_loss}, total samples {total_samples}")

        avg_loss = total_loss.item() / total_samples.item()
        epoch_accuracy = total_corrects.item() / total_samples.item()

        if rank == 0:
            epoch_precision = precision_score(all_true_labels, all_predicted_labels, zero_division=0)
            epoch_recall = recall_score(all_true_labels, all_predicted_labels, zero_division=0)
            epoch_f1 = f1_score(all_true_labels, all_predicted_labels, zero_division=0)

            print(classification_report(all_true_labels, all_predicted_labels, digits=4))
            print(confusion_matrix(all_true_labels, all_predicted_labels))
            print("Unique predicted labels:", set(all_predicted_labels))
            print("True label distribution:", pd.Series(all_true_labels).value_counts())
            print("Predicted label distribution:", pd.Series(all_predicted_labels).value_counts())
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": epoch_accuracy,
                "train_precision": epoch_precision,
                "train_recall": epoch_recall,
                "train_f1": epoch_f1
            })
        
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, "
                f"Train Accuracy: {epoch_accuracy:.4f}, Train Precision: {epoch_precision:.4f}, "
                f"Train Recall: {epoch_recall:.4f}, Train F1 Score: {epoch_f1:.4f}")

        # If validation, compute and report the main metrics on the validation set
        if validation and rank == 0:
            avg_val_loss, val_accuracy, val_f1, val_precision, val_recall, val_best_threshold = evaluate_model(model, val_loader, model_name, device, "", tune_threshold=True, criterion=criterion)
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            })
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                f"Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")
            
            # Update best validation f1 score, best model and save checkpoint
            if val_f1 > best_val_f1:
                print("Replacing best validation F1 score from ", best_val_f1 , " to ", val_f1, " best threshold from ", best_threshold, " to ", val_best_threshold)
                best_val_f1 = val_f1
                best_model = model
                best_threshold = val_best_threshold
                trigger_times = 0
                torch.save(model.state_dict(), model_check_path)
            # Early stopping logic 
            else:
                trigger_times += 1
                if trigger_times > patience:
                    print('Early stopping!')
                    break
    if not validation and rank == 0:
        torch.save(model.state_dict(), model_check_path)

    # Finally, evaluate on the test set and report all metrics
    if rank == 0:
        print("Running evaluation ...")
        test_loss, test_accuracy, test_f1, test_precision, test_recall, selected_threshold = evaluate_model(best_model, test_loader, model_name, device, "", tune_threshold=False, best_threshold=best_threshold, criterion=criterion)
        assert selected_threshold == best_threshold

        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })
        print(f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, "
            f"Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}, Threshold: {best_threshold:.4f}")

    # Finish the run
    wandb.finish()