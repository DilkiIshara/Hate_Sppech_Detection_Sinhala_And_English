# ============================================================
# XLM-R EN -> SI TRANSFER LEARNING
# ============================================================
#
# PAPER-ALIGNED IMPLEMENTATION
#
# Phase 1:
#       XLM-R-large
#            ↓
#       OLID English
#
# Phase 2:
#       English-finetuned XLM-R-large
#            ↓
#       SOLD Sinhala TRAIN
#
# Final evaluation:
#       Final model
#            ↓
#       SOLD Sinhala TEST
#
#
# IMPORTANT:
#
# OLID:
#     text + label
#
# SOLD:
#     text + label
#
# IGNORE:
#     lang
#     text_trans
#
# SOLD TEST IS NEVER USED FOR TRAINING.
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

# ------------------------------------------------------------
# OLID
#
# Use the OLID Level-A training data.
# ------------------------------------------------------------

OLID_TRAIN_FILE = (
    "../../Data New/CombineDataSets/DynamicallyGeneratedHateDataset_translation.csv"
)


# ------------------------------------------------------------
# SOLD
#
# IMPORTANT:
#
# sold_train = training only
# sold_test  = testing only
# ------------------------------------------------------------

SOLD_TRAIN_FILE = (
    "../../Data New/CombineDataSets/sold_train_translation.csv"
)

SOLD_TEST_FILE = (
    "../../Data New/RowData/sold_test.csv"
)


# ============================================================
# 4. OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = "OutPut/XLMR_EN_SI_PAPER"

PHASE1_DIR = os.path.join(
    OUTPUT_DIR,
    "phase1_olid_xlmr_large"
)

PHASE2_DIR = os.path.join(
    OUTPUT_DIR,
    "phase2_en_si_xlmr_large"
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
# 6. DEVICE INFORMATION
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
# 7. LOAD OLID
# ============================================================

def load_olid(
    file_path
):

    print("\n")
    print("=" * 80)
    print("LOADING OLID")
    print("=" * 80)

    # OLID is TSV
    df = pd.read_csv(
        file_path,
        sep="\t"
    )

    print(
        "Original shape:",
        df.shape
    )

    print(
        "Original columns:",
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # OLID v1 normally contains:
    #
    # id
    # tweet
    # subtask_a
    # subtask_b
    # subtask_c
    #
    # We only need:
    #
    # tweet
    # subtask_a
    # --------------------------------------------------------

    if "tweet" not in df.columns:

        raise ValueError(
            "OLID file does not contain "
            "'tweet' column."
        )

    if "subtask_a" not in df.columns:

        raise ValueError(
            "OLID file does not contain "
            "'subtask_a' column."
        )

    df = df[
        [
            "tweet",
            "subtask_a"
        ]
    ].copy()

    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "tweet": "text",
            "subtask_a": "label"
        }
    )

    # --------------------------------------------------------
    # Map OLID labels
    #
    # OFF = 1
    # NOT = 0
    # --------------------------------------------------------

    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .map(
            {
                "OFF": 1,
                "NOT": 0
            }
        )
    )

    # --------------------------------------------------------
    # Remove invalid rows
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

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Remove exact duplicates
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
        "Final OLID samples:",
        len(df)
    )

    print(
        "\nOLID label distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# 8. LOAD SOLD
# ============================================================

def load_sold(
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
        "Original columns:",
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We intentionally use ONLY:
    #
    # text
    # label
    #
    # Ignore:
    #
    # lang
    # text_trans
    # --------------------------------------------------------

    if "text" not in df.columns:

        raise ValueError(
            "SOLD dataset does not contain "
            "'text' column."
        )

    if "label" not in df.columns:

        raise ValueError(
            "SOLD dataset does not contain "
            "'label' column."
        )

    df = df[
        [
            "text",
            "label"
        ]
    ].copy()

    # --------------------------------------------------------
    # Missing values
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
    # Binary labels
    # --------------------------------------------------------

    df = df[
        df["label"].isin(
            [0, 1]
        )
    ]

    # --------------------------------------------------------
    # Duplicates
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
# 9. LOAD ALL DATA
# ============================================================

olid_df = load_olid(
    OLID_TRAIN_FILE
)

sold_train_df = load_sold(
    SOLD_TRAIN_FILE,
    "train"
)

sold_test_df = load_sold(
    SOLD_TEST_FILE,
    "test"
)


# ============================================================
# 10. DO NOT TOUCH SOLD TEST
# ============================================================

print("\n")
print("=" * 80)

print(
    "IMPORTANT DATA SPLIT"
)

print("=" * 80)

print(
    "OLID:",
    len(olid_df),
    "samples"
)

print(
    "SOLD train:",
    len(sold_train_df),
    "samples"
)

print(
    "SOLD test:",
    len(sold_test_df),
    "samples"
)

print(
    "\nSOLD TEST WILL ONLY BE USED "
    "FOR FINAL EVALUATION."
)


# ============================================================
# 11. CREATE OLID TRAIN / VALIDATION
# ============================================================
#
# The paper uses a validation portion during training.
#
# We keep SOLD completely separate.
#
# ============================================================

olid_train_df, olid_val_df = (
    __import__(
        "sklearn.model_selection",
        fromlist=[
            "train_test_split"
        ]
    ).train_test_split(

        olid_df,

        test_size=0.20,

        stratify=olid_df["label"],

        random_state=SEED
    )
)


# ============================================================
# 12. CREATE SOLD VALIDATION FROM SOLD TRAIN
# ============================================================
#
# SOLD TEST remains untouched.
#
# We use a portion of SOLD TRAIN for validation
# during Phase 2.
#
# ============================================================

sold_train_df, sold_val_df = (
    __import__(
        "sklearn.model_selection",
        fromlist=[
            "train_test_split"
        ]
    ).train_test_split(

        sold_train_df,

        test_size=0.20,

        stratify=sold_train_df["label"],

        random_state=SEED
    )
)


# Reset indexes

olid_train_df = (
    olid_train_df
    .reset_index(drop=True)
)

olid_val_df = (
    olid_val_df
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
print("FINAL TRAINING SPLITS")
print("=" * 80)

print(
    "\nOLID:"
)

print(
    "Train:",
    len(olid_train_df)
)

print(
    "Validation:",
    len(olid_val_df)
)

print(
    "\nSOLD:"
)

print(
    "Train:",
    len(sold_train_df)
)

print(
    "Validation:",
    len(sold_val_df)
)

print(
    "Test:",
    len(sold_test_df)
)


# ============================================================
# 14. CONVERT TO HUGGINGFACE DATASETS
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


olid_train = to_hf_dataset(
    olid_train_df
)

olid_val = to_hf_dataset(
    olid_val_df
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
print(
    "LOADING XLM-ROBERTA-LARGE"
)
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
    "\nTokenizing OLID..."
)

olid_train = olid_train.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

olid_val = olid_val.map(
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
# PHASE 1
# OLID -> XLM-R-LARGE
# ============================================================
# ============================================================

print("\n")
print("=" * 80)

print(
    "PHASE 1: XLM-R-LARGE -> OLID"
)

print("=" * 80)


# ------------------------------------------------------------
# Load pretrained XLM-R-large
# ------------------------------------------------------------

model_en = (
    AutoModelForSequenceClassification
    .from_pretrained(

        MODEL_NAME,

        num_labels=NUM_LABELS
    )
)


# ============================================================
# PAPER TRAINING SETTINGS
# ============================================================
#
# Paper:
#
# Model       = XLM-R-large
# Batch size  = 16
# LR          = 2e-5
# Warmup      = 10%
# Epochs      = 3
# Early stop  = evaluation loss
#
# ------------------------------------------------------------
# HARDWARE ADAPTATION
#
# Your RTX 2050 has ~4 GB VRAM.
#
# Therefore:
#
# actual batch = 1
# accumulation = 16
#
# effective batch = 16
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
# PHASE 1 TRAINER
# ============================================================

trainer_en = Trainer(

    model=model_en,

    args=training_args_en,

    train_dataset=olid_train,

    eval_dataset=olid_val,

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
# TRAIN PHASE 1
# ============================================================

print(
    "\nTraining XLM-R-large on OLID..."
)

trainer_en.train()


# ============================================================
# PHASE 1 VALIDATION
# ============================================================

print("\n")
print("=" * 80)

print(
    "PHASE 1 OLID VALIDATION"
)

print("=" * 80)

olid_results = trainer_en.evaluate(
    olid_val
)

print(
    olid_results
)


# ============================================================
# SAVE PHASE 1 MODEL
# ============================================================

trainer_en.save_model(
    PHASE1_DIR
)

tokenizer.save_pretrained(
    PHASE1_DIR
)


print(
    "\nPhase 1 model saved to:"
)

print(
    PHASE1_DIR
)


# ============================================================
# ============================================================
# PHASE 2
# OLID MODEL -> SOLD
# ============================================================
# ============================================================

print("\n")
print("=" * 80)

print(
    "PHASE 2: XLM-R ENGLISH -> SOLD SINHALA"
)

print("=" * 80)


# ------------------------------------------------------------
# CRITICAL STEP
#
# Load Phase 1 weights.
#
# DO NOT start from xlm-roberta-large again.
# ------------------------------------------------------------

model_en_si = (
    AutoModelForSequenceClassification
    .from_pretrained(

        PHASE1_DIR,

        num_labels=NUM_LABELS
    )
)


# ============================================================
# PHASE 2 TRAINING
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
# PHASE 2 TRAINER
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
# TRAIN PHASE 2
# ============================================================

print(
    "\nTraining transferred model on SOLD..."
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
    "\nFinal EN -> SI model saved to:"
)

print(
    PHASE2_DIR
)


# ============================================================
# ============================================================
# FINAL EVALUATION
# SOLD TEST ONLY
# ============================================================
# ============================================================

print("\n")
print("=" * 80)

print(
    "FINAL EVALUATION ON OFFICIAL SOLD TEST SET"
)

print("=" * 80)


# ------------------------------------------------------------
# IMPORTANT:
#
# sold_test has NEVER been used for:
#
# - training
# - model selection
# - early stopping
# - hyperparameter tuning
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
# 19. DETAILED CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 80)

print(
    "SOLD TEST CLASSIFICATION REPORT"
)

print("=" * 80)


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
# 20. CONFUSION MATRIX
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
# 21. SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "XLM-R EN -> SI",

    "model":
        MODEL_NAME,

    "source_dataset":
        "OLID Level A",

    "target_dataset":
        "SOLD",

    "phase_1":
        "OLID English",

    "phase_2":
        "SOLD Sinhala",

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
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)

print(
    "EXPERIMENT COMPLETED"
)

print("=" * 80)

print(
    "\nModel:",
    MODEL_NAME
)

print(
    "Phase 1:",
    "OLID English"
)

print(
    "Phase 2:",
    "SOLD Sinhala"
)

print(
    "Final evaluation:",
    "Official SOLD Test"
)

print(
    "\nSOLD Test Macro-F1:",
    round(
        sold_test_results[
            "eval_macro_f1"
        ],
        4
    )
)

print(
    "\nSOLD Test Weighted-F1:",
    round(
        sold_test_results[
            "eval_weighted_f1"
        ],
        4
    )
)

print(
    "\nSOLD Test Accuracy:",
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

print("\n")

print(
    "XLM-R EN -> SI FINISHED."
)