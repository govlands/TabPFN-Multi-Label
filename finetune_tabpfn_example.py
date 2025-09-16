"""Provides a detailed example of fine-tuning a TabPFNClassifier model.

This script demonstrates the complete workflow for fine-tuning the TabPFNClassifier
on the Covertype dataset for a multi-class classification task. It illustrates
the entire process, including loading and preparing the data, configuring the model
and optimizer, running the iterative fine-tuning loop, and evaluating model
performance using ROC AUC and Log Loss.

Note: We recommend running the fine-tuning scripts on a CUDA-enabled GPU, as full
support for the Apple Silicon (MPS) backend is still under development.
"""

from functools import partial

import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from sklearn.datasets import make_classification


def prepare_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the Covertype dataset."""
    print("--- 1. Data Preparation ---")
    # Use a smaller synthetic multi-class dataset instead of Covertype

    n_samples = min(config["num_samples_to_use"], 2000)
    X_all, y_all = make_classification(
        n_samples=n_samples,
        n_features=54,            # keep same dimensionality as Covertype
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_classes=7,
        n_clusters_per_class=1,
        flip_y=0.01,
        class_sep=1.0,
        random_state=config["random_seed"],
    )

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    splitter = partial(
        train_test_split,
        test_size=config["valid_set_ratio"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = splitter(X, y, stratify=y)

    print(
        f"Loaded and split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
    )
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_model_and_optimizer(config: dict) -> tuple[TabPFNClassifier, Optimizer, dict]:
    """Initializes the TabPFN classifier, optimizer, and training configs."""
    print("--- 2. Model and Optimizer Setup ---")
    classifier_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 4,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    classifier = TabPFNClassifier(
        **classifier_config, fit_mode="batched", differentiable_input=False
    )
    classifier._initialize_model_variables()
    # Optimizer uses finetuning-specific learning rate
    optimizer = AdamW(
        classifier.model_.parameters(), lr=config["finetuning"]["learning_rate"]
    )

    print(f"Using device: {config['device']}")
    print(f"Optimizer: Adam, Finetuning LR: {config['finetuning']['learning_rate']}")
    print("----------------------------------\n")
    return classifier, optimizer, classifier_config


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluates the model's performance on the test set."""
    eval_classifier = clone_model_for_evaluation(
        classifier, eval_config, TabPFNClassifier
    )
    eval_classifier.fit(X_train, y_train)

    try:
        probabilities = eval_classifier.predict_proba(X_test)
        roc_auc = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="weighted"
        )
        log_loss_score = log_loss(y_test, probabilities)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        roc_auc, log_loss_score = np.nan, np.nan

    return roc_auc, log_loss_score


def identity_split(X, y, random_state=None, **kwargs):
    return X, X, y, y


def main() -> None:
    """Main function to configure and run the finetuning workflow."""
    # --- Master Configuration ---
    config = {
        # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # The total number of samples to draw from the full dataset. This is useful for
        # managing memory and computation time, especially with large datasets.
        # For very large datasets the entire dataset is preprocessed and then
        # fit in memory, potentially leading to OOM errors.
        "num_samples_to_use": 100_000,
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": 42,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.3,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": 512,
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": 10,
        # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
        "learning_rate": 1e-5,
        # Meta Batch size for finetuning, i.e. how many datasets per batch. Must be 1 currently.
        "meta_batch_size": 1,
        # The number of samples within each training data split. It's capped by
        # n_inference_context_samples to align with the evaluation setup.
        "batch_size": int(
            min(
                config["n_inference_context_samples"],
                config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
            )
        ),
    }

    # --- Setup Data, Model, and Dataloader ---
    X_train, X_test, y_train, y_test = prepare_data(config)
    classifier, optimizer, classifier_config = setup_model_and_optimizer(config)

    # splitter = partial(train_test_split, test_size=1)
    splitter = identity_split
    training_datasets = classifier.get_preprocessed_datasets(
        X_train, y_train, splitter, config["finetuning"]["batch_size"]
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )
    loss_function = torch.nn.CrossEntropyLoss()

    eval_config = {
        **classifier_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            # Finetuning Step
            progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
            for (
                X_train_batch,
                X_test_batch,
                y_train_batch,
                y_test_batch,
                cat_ixs,
                confs,
            ) in progress_bar:
                X_full = X_train_batch
                y_full = y_train_batch
                n_samples = X_full[0].shape[1]
                
                if len(y_full) < 2: continue
                
                kf = KFold(n_splits=5, shuffle=True, random_state=config['random_seed'])
                for train_idx, test_idx in kf.split(np.arange(n_samples)):
                    X_fold_tr = [x[:, train_idx, :] for x in X_full]
                    X_fold_te = [x[:, test_idx, :] for x in X_full]
                    y_fold_tr = [y[:, train_idx] for y in y_full]
                    y_fold_te = y_test_batch[0][test_idx]
                    
                    y_fold_tr_n = y_fold_tr[0].squeeze(0).cpu().numpy()
                    y_fold_te_n = y_fold_te.cpu().numpy()
                    if len(np.unique(y_fold_te_n)) != len(np.unique(y_fold_tr_n)):
                        continue
                    
                    optimizer.zero_grad()
                    classifier.fit_from_preprocessed(
                        X_fold_tr, y_fold_tr, cat_ixs, confs
                    )
                    
                    preds = classifier.forward(X_fold_te, return_logits=True)
                    # print("preds.shape", preds.shape, "preds.dtype", preds.dtype)
                    if preds.shape[0] == 1:
                        preds = preds.squeeze(0)
                    preds = preds.transpose(0, 1)
                    target = torch.as_tensor(y_fold_te, dtype=torch.long, device=config['device']) 
                    # print("target.shape", target.shape, "target.dtype", target.dtype) 
                     
                    loss = loss_function(preds, target)
                    loss.backward()
                    optimizer.step()
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                # if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                #     continue  # Skip batch if splits don't have all classes

                # optimizer.zero_grad()
                # classifier.fit_from_preprocessed(
                #     X_train_batch, y_train_batch, cat_ixs, confs
                # )
                # predictions = classifier.forward(X_test_batch, return_logits=True)
                # loss = loss_function(predictions, y_test_batch.to(config["device"]))
                # loss.backward()
                # optimizer.step()

                # # Set the postfix of the progress bar to show the current loss
                # progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluation Step (runs before finetuning and after each epoch)
        epoch_roc, epoch_log_loss = evaluate_model(
            classifier, eval_config, X_train, y_train, X_test, y_test
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"📊 {status} Evaluation | Test ROC: {epoch_roc:.4f}, Test Log Loss: {epoch_log_loss:.4f}\n"
        )

    print("--- ✅ Finetuning Finished ---")


if __name__ == "__main__":
    main()