# ============================================================
# XLM-R EN -> SI
# ============================================================
#
# YOUR EXPERIMENT
#
# Stage 1:
# DynamicallyGeneratedHateDataset
#             ↓
#       XLM-R-large
#
# Stage 2:
# English-finetuned XLM-R-large
#             ↓
#        SOLD TRAIN
#
# Final:
# English -> Sinhala XLM-R
#             ↓
#        SOLD TEST
#
#
# IMPORTANT
# ------------------------------------------------------------
# English dataset:
#
# USE:
#   text
#   label
#
# IGNORE:
#   lang
#   text_trans
#
# SOLD:
#
# USE:
#   text
#   label
#
# SOLD TEST:
#   NEVER used during training
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-large"

SEED = 42

NUM_LABELS = 2

MAX_LENGTH = 128


# ============================================================
# 3. DATASET PATHS
# ============================================================

ENGLISH_FILE = (
    "../../Data New/CombineDataSets/DynamicallyGeneratedHateDataset_translation.csv"
)

SOLD_TRAIN_FILE = (
    "../../Data New/CombineDataSets/sold_train_translation.csv"
)

SOLD_TEST_FILE = (
    "../../Data New/RowData/SOLD_test.csv"
)


# ============================================================
# 4. OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = "Output/XLMR_EN_SI_DGHD"

PHASE1_DIR = os.path.join(
    OUTPUT_DIR,
    "phase1_english"
)

PHASE2_DIR = os.path.join(
    OUTPUT_DIR,
    "phase2_en_si"
)

RESULTS_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

os.makedirs(
    PHASE1_DIR,
    exist_ok=True
)

os.makedirs(
    PHASE2_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

set_seed(SEED)

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 6. DEVICE
# ============================================================

print("=" * 80)
print("DEVICE INFORMATION")
print("=" * 80)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# 7. LOAD ENGLISH DATASET
# ============================================================

def load_english_dataset(
    file_path
):

    print("\n")
    print("=" * 80)
    print("LOADING ENGLISH DATASET")
    print("=" * 80)

    df = pd.read_csv(
        file_path
    )

    print(
        "Original shape:",
        df.shape
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # YOUR DATASET:
    #
    # text
    # label
    # lang
    # text_trans
    #
    # We ONLY use:
    #
    # text
    # label
    #
    # lang       -> ignored
    # text_trans -> ignored
    # --------------------------------------------------------

    required_columns = [
        "text",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"English dataset does not "
                f"contain '{column}'"
            )

    df = df[
        [
            "text",
            "label"
        ]
    ].copy()

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "text",
            "label"
        ]
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"].str.len() > 0
    ]

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["label"]
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Only binary classification
    # --------------------------------------------------------

    df = df[
        df["label"].isin(
            [0, 1]
        )
    ]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "label"
        ]
    )

    print(
        "Duplicates removed:",
        before - len(df)
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "Final English samples:",
        len(df)
    )

    print(
        "\nEnglish label distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# 8. LOAD SOLD DATASET
# ============================================================

def load_sold_dataset(
    file_path,
    split_name
):

    print("\n")
    print("=" * 80)

    print(
        f"LOADING SOLD {split_name.upper()}"
    )

    print("=" * 80)

    df = pd.read_csv(
        file_path
    )

    print(
        "Original shape:",
        df.shape
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    required_columns = [
        "text",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"SOLD {split_name} does not "
                f"contain '{column}'"
            )

    # --------------------------------------------------------
    # Only use text + label
    # --------------------------------------------------------

    df = df[
        [
            "text",
            "label"
        ]
    ].copy()

    # --------------------------------------------------------
    # Remove missing
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "text",
            "label"
        ]
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"].str.len() > 0
    ]

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["label"]
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    df = df[
        df["label"].isin(
            [0, 1]
        )
    ]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "label"
        ]
    )

    print(
        "Duplicates removed:",
        before - len(df)
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"Final SOLD {split_name} samples:",
        len(df)
    )

    print(
        "\nLabel distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# 9. LOAD DATA
# ============================================================

english_df = load_english_dataset(
    ENGLISH_FILE
)

sold_train_full_df = load_sold_dataset(
    SOLD_TRAIN_FILE,
    "train"
)

sold_test_df = load_sold_dataset(
    SOLD_TEST_FILE,
    "test"
)


# ============================================================
# 10. IMPORTANT DATA SEPARATION
# ============================================================

print("\n")
print("=" * 80)
print("DATASET SEPARATION")
print("=" * 80)

print(
    "\nEnglish source:",
    len(english_df)
)

print(
    "SOLD train:",
    len(sold_train_full_df)
)

print(
    "SOLD test:",
    len(sold_test_df)
)

print(
    "\nSOLD TEST WILL ONLY BE USED "
    "FOR FINAL EVALUATION."
)


# ============================================================
# 11. ENGLISH TRAIN / VALIDATION
# ============================================================

english_train_df, english_val_df = (
    train_test_split(

        english_df,

        test_size=0.20,

        stratify=english_df["label"],

        random_state=SEED
    )
)


# ============================================================
# 12. SOLD TRAIN / VALIDATION
# ============================================================
#
# IMPORTANT:
#
# We split ONLY the SOLD TRAIN data.
#
# SOLD TEST remains untouched.
#
# ============================================================

sold_train_df, sold_val_df = (
    train_test_split(

        sold_train_full_df,

        test_size=0.20,

        stratify=sold_train_full_df["label"],

        random_state=SEED
    )
)


english_train_df = (
    english_train_df
    .reset_index(drop=True)
)

english_val_df = (
    english_val_df
    .reset_index(drop=True)
)

sold_train_df = (
    sold_train_df
    .reset_index(drop=True)
)

sold_val_df = (
    sold_val_df
    .reset_index(drop=True)
)


# ============================================================
# 13. PRINT SPLITS
# ============================================================

print("\n")
print("=" * 80)
print("TRAINING SPLITS")
print("=" * 80)

print("\nEnglish")

print(
    "Train:",
    len(english_train_df)
)

print(
    "Validation:",
    len(english_val_df)
)

print("\nSOLD")

print(
    "Train:",
    len(sold_train_df)
)

print(
    "Validation:",
    len(sold_val_df)
)

print(
    "Official Test:",
    len(sold_test_df)
)


# ============================================================
# 14. CONVERT TO HF DATASETS
# ============================================================

def to_hf_dataset(
    df
):

    return Dataset.from_pandas(

        df[
            [
                "text",
                "label"
            ]
        ],

        preserve_index=False
    )


english_train = to_hf_dataset(
    english_train_df
)

english_val = to_hf_dataset(
    english_val_df
)

sold_train = to_hf_dataset(
    sold_train_df
)

sold_val = to_hf_dataset(
    sold_val_df
)

sold_test = to_hf_dataset(
    sold_test_df
)


# ============================================================
# 15. TOKENIZER
# ============================================================

print("\n")
print("=" * 80)
print("LOADING XLM-RoBERTa-LARGE")
print("=" * 80)

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# 16. TOKENIZATION
# ============================================================

def tokenize(
    examples
):

    return tokenizer(

        examples["text"],

        truncation=True,

        max_length=MAX_LENGTH
    )


print(
    "\nTokenizing English..."
)

english_train = english_train.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

english_val = english_val.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)


print(
    "Tokenizing SOLD..."
)

sold_train = sold_train.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

sold_val = sold_val.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

sold_test = sold_test.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)


# ============================================================
# 17. DATA COLLATOR
# ============================================================

data_collator = (
    DataCollatorWithPadding(
        tokenizer=tokenizer
    )
)


# ============================================================
# 18. METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(

            labels,

            predictions,

            average="macro",

            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(

            labels,

            predictions,

            average="weighted",

            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {

        "accuracy": accuracy,

        "macro_f1": macro_f1,

        "macro_precision": precision,

        "macro_recall": recall,

        "weighted_f1": weighted_f1,

        "weighted_precision":
            weighted_precision,

        "weighted_recall":
            weighted_recall
    }


# ============================================================
# ============================================================
# STAGE 1
# ENGLISH DATASET -> XLM-R-LARGE
# ============================================================
# ============================================================

print("\n")
print("=" * 80)

print(
    "STAGE 1: ENGLISH FINE-TUNING"
)

print("=" * 80)


# ------------------------------------------------------------
# Load PRETRAINED XLM-R-large
# ------------------------------------------------------------

model_en = (
    AutoModelForSequenceClassification
    .from_pretrained(

        MODEL_NAME,

        num_labels=NUM_LABELS
    )
)


# ============================================================
# TRAINING SETTINGS
# ============================================================
#
# Paper-aligned:
#
# Learning rate = 2e-5
# Warmup = 10%
# Epochs = 3
#
# Hardware:
#
# RTX 2050 4GB
#
# Per-device batch = 1
# Accumulation = 16
#
# Effective batch = 16
#
# ============================================================

training_args_en = TrainingArguments(

    output_dir=PHASE1_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    weight_decay=0.0,

    warmup_ratio=0.10,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    save_total_limit=2,

    logging_steps=100,

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=SEED,

    gradient_checkpointing=True,

    dataloader_num_workers=0,
)


# ============================================================
# STAGE 1 TRAINER
# ============================================================

trainer_en = Trainer(

    model=model_en,

    args=training_args_en,

    train_dataset=english_train,

    eval_dataset=english_val,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)


# ============================================================
# TRAIN STAGE 1
# ============================================================

print(
    "\nTraining XLM-R-large on "
    "DynamicallyGeneratedHateDataset..."
)

trainer_en.train()


# ============================================================
# STAGE 1 VALIDATION
# ============================================================

print("\n")
print("=" * 80)

print(
    "STAGE 1 ENGLISH VALIDATION"
)

print("=" * 80)

english_results = (
    trainer_en.evaluate(
        english_val
    )
)

print(
    english_results
)


# ============================================================
# SAVE STAGE 1
# ============================================================

trainer_en.save_model(
    PHASE1_DIR
)

tokenizer.save_pretrained(
    PHASE1_DIR
)


print(
    "\nEnglish-finetuned model saved:"
)

print(
    PHASE1_DIR
)


# ============================================================
# ============================================================
# STAGE 2
# ENGLISH MODEL -> SOLD
# ============================================================
# ============================================================

print("\n")
print("=" * 80)

print(
    "STAGE 2: ENGLISH -> SINHALA"
)

print("=" * 80)


# ------------------------------------------------------------
# CRITICAL:
#
# LOAD STAGE 1 MODEL
#
# NOT:
#
# AutoModelForSequenceClassification
# .from_pretrained("xlm-roberta-large")
#
# ------------------------------------------------------------

model_en_si = (
    AutoModelForSequenceClassification
    .from_pretrained(

        PHASE1_DIR,

        num_labels=NUM_LABELS
    )
)


# ============================================================
# STAGE 2 SETTINGS
# ============================================================

training_args_si = TrainingArguments(

    output_dir=PHASE2_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    weight_decay=0.0,

    warmup_ratio=0.10,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    save_total_limit=2,

    logging_steps=50,

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=SEED,

    gradient_checkpointing=True,

    dataloader_num_workers=0,
)


# ============================================================
# STAGE 2 TRAINER
# ============================================================

trainer_en_si = Trainer(

    model=model_en_si,

    args=training_args_si,

    train_dataset=sold_train,

    eval_dataset=sold_val,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)


# ============================================================
# TRAIN STAGE 2
# ============================================================

print(
    "\nTraining English-finetuned "
    "XLM-R-large on SOLD..."
)

trainer_en_si.train()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

trainer_en_si.save_model(
    PHASE2_DIR
)

tokenizer.save_pretrained(
    PHASE2_DIR
)


print(
    "\nFinal XLM-R EN -> SI model saved:"
)

print(
    PHASE2_DIR
)


# ============================================================
# ============================================================
# FINAL EVALUATION
# SOLD TEST
# ============================================================
# ============================================================

print("\n")
print("=" * 80)

print(
    "FINAL EVALUATION ON SOLD TEST"
)

print("=" * 80)


# ------------------------------------------------------------
# SOLD TEST HAS NOT BEEN USED DURING:
#
# - Stage 1 training
# - Stage 1 validation
# - Stage 2 training
# - Stage 2 validation
# - Early stopping
#
# ------------------------------------------------------------

sold_test_results = (
    trainer_en_si.evaluate(
        sold_test
    )
)


print(
    "\nSOLD TEST RESULTS:"
)

print(
    json.dumps(
        sold_test_results,
        indent=4,
        default=str
    )
)


# ============================================================
# 19. PREDICTIONS
# ============================================================

predictions = (
    trainer_en_si.predict(
        sold_test
    )
)

y_true = (
    predictions.label_ids
)

y_pred = np.argmax(
    predictions.predictions,
    axis=-1
)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 80)

print(
    "SOLD TEST CLASSIFICATION REPORT"
)

print("=" * 80)

report = classification_report(

    y_true,

    y_pred,

    target_names=[
        "Not Offensive",
        "Offensive"
    ],

    digits=4,

    zero_division=0
)

print(
    report
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print(
    "\nConfusion Matrix:"
)

print(
    cm
)


# ============================================================
# 22. SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "DynamicallyGeneratedHateDataset -> SOLD",

    "model":
        MODEL_NAME,

    "source_dataset":
        "DynamicallyGeneratedHateDataset",

    "target_dataset":
        "SOLD",

    "phase_1":
        "English",

    "phase_2":
        "Sinhala",

    "english_samples":
        len(english_df),

    "sold_train_samples":
        len(sold_train_full_df),

    "sold_test_samples":
        len(sold_test_df),

    "sold_test_metrics":
        sold_test_results,

    "confusion_matrix":
        cm.tolist(),

    "classification_report":
        report,

    "seed":
        SEED,

    "max_length":
        MAX_LENGTH,

    "learning_rate":
        2e-5,

    "warmup_ratio":
        0.10,

    "epochs":
        3,

    "effective_batch_size":
        16
}


results_file = os.path.join(
    RESULTS_DIR,
    "xlmr_en_si_results.json"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)

print(
    "XLM-R EN -> SI COMPLETED"
)

print("=" * 80)

print(
    "\nSource:"
)

print(
    "DynamicallyGeneratedHateDataset"
)

print(
    "\nStage 1:"
)

print(
    "English fine-tuning"
)

print(
    "\nStage 2:"
)

print(
    "Sinhala SOLD fine-tuning"
)

print(
    "\nFinal evaluation:"
)

print(
    "Official SOLD test set"
)

print(
    "\nModel:"
)

print(
    MODEL_NAME
)

print(
    "\nSOLD Test Macro-F1:"
)

print(
    round(
        sold_test_results[
            "eval_macro_f1"
        ],
        4
    )
)

print(
    "\nSOLD Test Weighted-F1:"
)

print(
    round(
        sold_test_results[
            "eval_weighted_f1"
        ],
        4
    )
)

print(
    "\nSOLD Test Accuracy:"
)

print(
    round(
        sold_test_results[
            "eval_accuracy"
        ],
        4
    )
)

print(
    "\nFinal model:"
)

print(
    PHASE2_DIR
)

print(
    "\nResults:"
)

print(
    results_file
)

print(
    "\nXLM-R EN -> SI FINISHED."
)