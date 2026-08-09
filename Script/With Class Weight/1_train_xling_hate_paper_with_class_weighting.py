

import os, math, random
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads if hf_transfer installed
# test
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaModel,
    get_linear_schedule_with_warmup,
)

# -------------- Config (Notebook) ----------------
config = {
    "train": "../../Data New/CombineDataSets/train.csv", # training dataset path
    "dev":   "../../Data New/CombineDataSets/test.csv", # development dataset path
    "outdir": "Output/PaperApproach_with_Class_weight_0.3", # output directory to save the best model and tokenizer
    "epochs": 15, # number of full passes over the training data
    "batch_size": 4, # number of samples per training batch (adjust based on GPU memory)
    "max_len": 192, # maximum token length for input sequences Transformer models (XLM-RoBERTa) have a fixed maximum input size (truncation/padding)
    "lr": 2e-5, # learning rate for the AdamW optimizer
    "warmup_frac": 0.1, # fraction of total training steps to linearly warm up the learning rate (10% of steps are warm-up)
    "seed": 42, # random seed for reproducibility (same results across runs)
    "model_name": "xlm-roberta-base", # pretrained transformer model to use (XLM-RoBERTa base)
    "lambda_contrast": 0.3, # weight for the contrastive loss component (0.7 means 70% contrastive, 30% cross-entropy)
    "temperature": 0.2, # contrastive similarity scores to improve representation learning.
    "early_stop_patience": 2, # number of epochs to wait for improvement before early stopping (stop if no improvement in Macro-F1 for 2 consecutive epochs)
    "freeze_warmup_epochs": 0,  # number of initial epochs to freeze the encoder for warm-up (0 means no freezing)
    "resume_download": True, # model download can continue from where it stopped if it was interrupted.
}

import torch

print("CUDA Available:", torch.cuda.is_available())
#print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Set the a random seed for reproducibility across runs (same results each time)
# This sets the seed for Python's built-in random module, NumPy, and PyTorch (both CPU and CUDA).
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import torch
print(torch.version.cuda)

# Check if CUDA (GPU) is available and set the device accordingly. If a GPU is available, it will use "cuda" for faster computations; otherwise, it will fall back to "cpu".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ---------------- Custom Dataset & Collator ----------------
class HatePairDataset(Dataset):
    def __init__(self, csv_path, is_train):
        df = pd.read_csv(csv_path)
        self.text = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.lang = df["lang"].astype(str).tolist()
        self.is_train = is_train

        if is_train:
            # ONLY training data must have translated text
            assert "text_trans" in df.columns
            self.text_trans = df["text_trans"].astype(str).tolist()
        else:
            self.text_trans = None


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        item = {
            "text": self.text[idx],
            "label": self.labels[idx],
            "lang": self.lang[idx]
        }
        if self.is_train:
            item["text_trans"] = self.text_trans[idx]
        return item


@dataclass
class Batch:
    enc_o: dict
    enc_t: dict | None
    labels: torch.Tensor
    langs: list


#----Batch collator: takes a list of samples from the dataset and processes them into a batch suitable for model input.
# It tokenizes the original and translated texts (if available), pads/truncates them to a fixed length, and converts labels to tensors.
class Collator:

    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len


    def __call__(self, batch):
        texts_o = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch]).long()
        langs = [b["lang"] for b in batch]

        enc_o = self.tok(
            texts_o,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # OPTIONAL translated text (only for training)
        if "text_trans" in batch[0]:
            texts_t = [b["text_trans"] for b in batch]
            enc_t = self.tok(
                texts_t,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
        else:
            enc_t = None

        return Batch(enc_o=enc_o, enc_t=enc_t, labels=labels, langs=langs)



# ---------------- Model ----------------
class XlingHate(nn.Module):

    def __init__(self, model_name="xlm-roberta-base", num_labels=2, class_weights=None, temperature=0.3):
        super().__init__()

        self.enc = XLMRobertaModel.from_pretrained(model_name)

        hidden = self.enc.config.hidden_size

        self.cls = nn.Linear(hidden, num_labels) #Dense Layere


        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )

        self.temperature = temperature

        if class_weights is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            )


    def encode_logits(self, **enc):

        out = self.enc(**enc)

        # CLS representation
        # h = out.last_hidden_state[:, 0, :]
        mask = enc["attention_mask"].unsqueeze(-1)
        h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1) # average pooling over the sequence length, weighted by the attention mask to ignore padding tokens.

        logits = self.cls(h)

        return h, logits


    def contrastive_loss(self, z1, z2):
       z1 = nn.functional.normalize(z1, dim=1)
       z2 = nn.functional.normalize(z2, dim=1)

       sim = torch.matmul(z1, z2.T) / self.temperature  # [N, N]

       exp_sim = torch.exp(sim)

       # positive pairs (diagonal)
       pos = torch.diag(exp_sim)

       # denominator = row sum + column sum
       denom = exp_sim.sum(dim=1) + exp_sim.sum(dim=0)

       loss = -torch.log(pos / denom)

       return loss.mean()

    def forward(self, batch: Batch, lambda_contrast: float = 0.7):

        enc_o = {k: v.to(DEVICE) for k, v in batch.enc_o.items()}
        y = batch.labels.to(DEVICE)

        h_o, logit_o = self.encode_logits(**enc_o)

        ce_o = self.ce(logit_o, y) # Compute the cross-entropy loss for the original text classification

        if batch.enc_t is not None:

            enc_t = {k: v.to(DEVICE) for k, v in batch.enc_t.items()}

            h_t, logit_t = self.encode_logits(**enc_t)


            ce_t = self.ce(logit_t, y) # Compute the cross-entropy loss for the translated text classification


            z_o = self.proj(h_o) # Project the original text representations to a lower-dimensional space for contrastive learning
            z_t = self.proj(h_t) # Project the translated text representations to the same space


            con_loss = self.contrastive_loss(z_o, z_t)


            loss = (1 - lambda_contrast) / 2 * ce_o \
                 + (1 - lambda_contrast) / 2 * ce_t \
                 + lambda_contrast * con_loss # Combine the losses: weighted sum of the original and translated cross-entropy losses and the contrastive loss. The lambda_contrast parameter controls the balance between classification and contrastive learning.

        else:

            loss = ce_o

        return loss, logit_o


# ---------------- Metrics ----------------
def macro_f1(y_true, y_pred): return f1_score(y_true, y_pred, average="macro")

def evaluate(model: XlingHate, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
    model.eval() # Set the model to evaluation mode
    ys, yh, langs_all = [], [], [] # Initialize lists to store true labels (ys), predicted labels (yh), and languages (langs_all) for all samples in the evaluation set.
    probs = []

    with torch.no_grad(): # Disable the gradient computation since we are only evaluating the model and not updating its weights.
        for b in loader:
            _, logits = model(b, lambda_contrast=0.0)  # During evaluation, we set lambda_contrast to 0.0 to only compute the classification loss.
            yhat = logits.argmax(-1).cpu().numpy().tolist() # get predicted labels by   taking the argmax of the logits (the class with the highest score) and converting it to a list.
            ys.extend(b.labels.numpy().tolist()) # Append the true labels from the batch to the ys list.
            yh.extend(yhat) # Append the predicted labels to the yh list.
            langs_all.extend(b.langs) # Append the languages of the samples to the langs_all list for per-language analysis later.
            p = torch.softmax(logits, -1)[:, 1].cpu().numpy().tolist() # Get the predicted probabilities for the positive class (hate speech) by applying softmax to the logits
            probs.extend(p)

    overall = classification_report(ys, yh, digits=3)
    print(overall)
    print("Confusion matrix:\n", confusion_matrix(ys, yh))

    f1 = macro_f1(ys, yh)

    # Compute per-language Macro-F1 scores to analyze
    per_lang = {}
    langs_unique = sorted(set(langs_all))
    for L in langs_unique:
        idx = [i for i, lg in enumerate(langs_all) if lg == L]
        if not idx: continue
        yl = [ys[i] for i in idx]
        yhl = [yh[i] for i in idx]
        f1l = macro_f1(yl, yhl)
        per_lang[L] = f1l
        print(f"[{L}] Macro-F1: {f1l:.4f}  support={len(idx)}")

    return f1, per_lang




# calculate the class weights
def compute_class_weights(csv_path: str) -> List[float]:
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().to_dict()  # {label: count}
    total = sum(counts.values())
    # inverse frequency
    w = [total / (2 * counts.get(0, 1)), total / (2 * counts.get(1, 1))]
    return w


# It controls whether the encoder (XLM-R) is allowed to learn or not.
def freeze_encoder(m: XlingHate, freeze=True):
    for p in m.enc.parameters():
        p.requires_grad = not (freeze)


# This function encapsulates the entire training loop, including data loading, model initialization, optimization, and evaluation.
def train_with_config(cfg: dict):
    set_seed(cfg["seed"])
    os.makedirs(cfg["outdir"], exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading tokenizer/model...")
    tok = XLMRobertaTokenizerFast.from_pretrained(cfg["model_name"], resume_download=cfg["resume_download"])
    # class weights from training distribution
    class_weights = compute_class_weights(cfg["train"])
    print("Class weights:", class_weights)

    model = XlingHate(
        num_labels=2,
        class_weights=class_weights,
        temperature=cfg["temperature"]
    ).to(DEVICE)

    tr_ds = HatePairDataset(cfg["train"],is_train=True)
    dv_ds = HatePairDataset(cfg["dev"],is_train=False)
    coll = Collator(tok, cfg["max_len"])
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=coll)
    dv_dl = DataLoader(dv_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=coll) # Create DataLoaders for the training and development datasets, using the custom Collator to process batches.

    # Initialize the optimizer with only parameters that require gradients
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    total_steps = max(1, cfg["epochs"] * len(tr_dl))
    sch = get_linear_schedule_with_warmup(opt, int(cfg["warmup_frac"] * total_steps), total_steps) # Set up a linear learning rate scheduler with a warm-up phase. The learning rate will linearly increase during the warm-up period (defined by warmup_frac) and then linearly decay for the rest of the training steps.
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_f1 = -1.0
    no_improve = 0

    # optional warm-up: freeze encoder for first N epochs
    if cfg["freeze_warmup_epochs"] > 0:
        print(f">> Freezing encoder for warm-up: {cfg['freeze_warmup_epochs']} epoch(s)")
        freeze_encoder(model, True) # Freeze the encoder for the specified number of warm-up epochs to allow the classification head to learn initial patterns before fine-tuning the entire model.

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        # The training loop iterates over the training DataLoader (tr_dl) for a specified number of epochs.
        # For each batch of data, it performs a forward pass through the model to compute the loss, applies backpropagation to compute gradients, and updates the model weights using the optimizer.
        # It also includes gradient clipping to prevent exploding gradients and uses mixed precision training for efficiency on GPUs.
        for step, batch in enumerate(tr_dl, 1):
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                loss, _ = model(batch, lambda_contrast=cfg["lambda_contrast"]) # Compute the loss
            scaler.scale(loss).backward() # compute gradients , how much each weight should change
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the gradients to a maximum norm of 1.0 to prevent exploding gradients, which can destabilize training.
            scaler.step(opt) # Update the model weights
            scaler.update() # Update the scale factor for the next iteration of mixed precision training.
            opt.zero_grad(set_to_none=True) # Clear the gradients.
            sch.step() # Update the learning rate according to the scheduler

            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(tr_dl)} loss {running/step:.4f}")

        # unfreeze after warm-up
        if cfg["freeze_warmup_epochs"] > 0 and epoch == cfg["freeze_warmup_epochs"]:
            print(">> Unfreezing encoder")
            freeze_encoder(model, False)

        f1, per_lang = evaluate(model, dv_dl)
        print(f"[epoch {epoch}] Macro-F1={f1:.4f} | per-lang={per_lang}")

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            no_improve = 0
            # save both model and tokenizer
            tok.save_pretrained(os.path.join(cfg["outdir"], "xlmr_xling_tok"))
            torch.save(model.state_dict(), os.path.join(cfg["outdir"], "xlmr_xling_best.pt"))
            print(f">> saved best to {cfg['outdir']}/")
        else:
            no_improve += 1
            if no_improve >= cfg["early_stop_patience"]:
                print("Early stopping triggered.")
                break


train_with_config(config)


# Throsold Tunning

# In[ ]:


import re
import unicodedata
from pathlib import Path

THR_MIN, THR_MAX, THR_STEP = 0.10, 0.90, 0.01
DEFAULT_THRESHOLD = 0.5

URL_RX = re.compile(r"(?xi)\b((?:https?://|www\d{0,3}[.])[^\s<>\"']+)")
EMAIL_RX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
MENTION_RX = re.compile(r"(^|\s)@\w+")
HASHTAG_RX = re.compile(r"(^|\s)#\w+")
MULTI_WS_RX = re.compile(r"\s+")


def shrink_repeats_en(t: str) -> str:
    return re.sub(r"([A-Za-z])\1{2,}", lambda m: m.group(1) * 2, t)


def clean_row(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFC", str(text))
    t = URL_RX.sub(" ", t)
    t = EMAIL_RX.sub(" ", t)
    t = MENTION_RX.sub(" ", t)
    t = HASHTAG_RX.sub(" ", t)
    if str(lang).strip().lower() == "en":
        t = t.lower()
        t = shrink_repeats_en(t)
    return MULTI_WS_RX.sub(" ", t).strip()


def load_eval_df(csv_path: str):
    df = pd.read_csv(csv_path)
    if "lang" not in df.columns:
        df["lang"] = "unk"
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df.apply(lambda r: clean_row(r["text"], r["lang"]), axis=1)
    df = df[df["text"].astype(str).str.len() > 0].reset_index(drop=True)
    return df


def find_best_threshold(y_true, probs, thr_min=THR_MIN, thr_max=THR_MAX, step=THR_STEP):
    best_thr, best_f1 = DEFAULT_THRESHOLD, -1.0
    rows = []
    for thr in np.arange(thr_min, thr_max + 1e-9, step):
        pred = (probs >= thr).astype(int)
        f1 = macro_f1(y_true, pred)
        rows.append({"threshold": thr, "macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1, pd.DataFrame(rows)


def load_trained_model(outdir: str):
    tok = XLMRobertaTokenizerFast.from_pretrained(str(Path(outdir) / "xlmr_xling_tok"))
    model = XlingHate(
        model_name=config["model_name"],
        temperature=config["temperature"],
    ).to(DEVICE)
    state = torch.load(Path(outdir) / "xlmr_xling_best.pt", map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return tok, model


@torch.no_grad()
def predict_probs(tok, model, texts, batch_size=32):
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i:i + batch_size],
            truncation=True,
            padding=True,
            max_length=config["max_len"],
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        batch = Batch(
            enc_o=enc,
            enc_t=None,
            labels=torch.zeros(len(enc["input_ids"]), dtype=torch.long),
            langs=["unk"] * len(enc["input_ids"]),
        )
        _, logits = model(batch, lambda_contrast=0.0)
        probs.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)


# ---- Tune on full validation (EN + SI) ----
val_df = load_eval_df(config["dev"])
y_val = val_df["label"].astype(int).tolist()
texts_val = val_df["text"].astype(str).tolist()
langs_val = val_df["lang"].astype(str).tolist()

tok_tune, model_tune = load_trained_model(config["outdir"])
val_probs = predict_probs(tok_tune, model_tune, texts_val)

baseline_f1 = macro_f1(y_val, (val_probs >= DEFAULT_THRESHOLD).astype(int))
best_thr_global, best_f1_global, thr_df = find_best_threshold(y_val, val_probs)

print(f"Validation rows: {len(val_df)} | EN={langs_val.count('en')} SI={langs_val.count('si')}")
print(f"Baseline Macro-F1 @ {DEFAULT_THRESHOLD}: {baseline_f1:.4f}")
print(f"Best GLOBAL threshold: {best_thr_global:.2f} | Macro-F1: {best_f1_global:.4f}")

# ---- Per-language thresholds ----
per_lang_thresholds = {}
print("\nPer-language optimal thresholds on validation:")
for lang in sorted(set(langs_val)):
    idx = [i for i, l in enumerate(langs_val) if l == lang]
    y_l = [y_val[i] for i in idx]
    p_l = val_probs[idx]
    thr_l, f1_l, _ = find_best_threshold(y_l, p_l)
    per_lang_thresholds[lang] = thr_l
    print(f"  [{lang}] best_thr={thr_l:.2f}  Macro-F1={f1_l:.4f}  support={len(idx)}")

print("\n--- Suitability summary ---")
print(f"Use best_thr_global={best_thr_global:.2f} for mixed EN+SI deployment.")
print(f"Use per_lang_thresholds['en']={per_lang_thresholds.get('en', DEFAULT_THRESHOLD):.2f} for English-only eval.")
print(f"Use per_lang_thresholds['si']={per_lang_thresholds.get('si', DEFAULT_THRESHOLD):.2f} for Sinhala-only eval.")
if abs(per_lang_thresholds.get("en", 0.5) - per_lang_thresholds.get("si", 0.5)) > 0.05:
    print("Note: EN and SI optimal thresholds differ — one global threshold is a compromise.")
else:
    print("Note: EN and SI optimal thresholds are similar — global threshold works well for both.")


# Evaluation

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from pathlib import Path

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("Output/PaperApproach_with_Class_weight_0.3") 
CKPT_PATH = MODEL_DIR / "xlmr_xling_best.pt"
TOK_PATH  = MODEL_DIR / "xlmr_xling_tok"    # tokenizer folder
BASE_NAME = "xlm-roberta-base"             # base encoder model
TEST_CSV  = Path("../../Data New/CombineDataSets/test.csv")
MAX_LEN   = 192
THRESHOLD = 0.38

if TOK_PATH.exists():
    tok = XLMRobertaTokenizerFast.from_pretrained(str(TOK_PATH))
    print(f"Loaded tokenizer from: {TOK_PATH}")
else:
    print("Tokenizer folder not found, falling back to base model tokenizer.")
    tok = XLMRobertaTokenizerFast.from_pretrained(BASE_NAME)

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")


class XlingPredictor(nn.Module):
    def __init__(self, model_name=BASE_NAME, num_labels=2):
        super().__init__()
        self.enc = XLMRobertaModel.from_pretrained(model_name)
        hidden = self.enc.config.hidden_size
        self.cls = nn.Linear(hidden, num_labels) #map the representation of token to logits

    def forward(self, **enc):
        out = self.enc(**enc)#encode the input text
        # Use masked mean pooling to match training evaluation
        mask = enc["attention_mask"].unsqueeze(-1)
        h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        logits = self.cls(h)
        return logits


model = XlingPredictor(BASE_NAME, num_labels=2)
state = torch.load(CKPT_PATH, map_location=DEVICE) #load the saved model checkpoint that contains the learned parameters (weights) of the model.

missing, unexpected = model.load_state_dict(state, strict=False) #load the model parameters from the checkpoint
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)

model.to(DEVICE).eval()
torch.set_grad_enabled(False) # turn off gradients calculation for inference


df = pd.read_csv(TEST_CSV)
needed = {"text", "label"}
if not needed.issubset(df.columns):
    raise ValueError(f"Test CSV missing columns: {needed - set(df.columns)}")

if "lang" not in df.columns:
    df["lang"] = "unk"

df = df.dropna(subset=["text", "label"]).reset_index(drop=True) # remove rows with missing text or label


@torch.no_grad()
def predict_batch(texts, batch_size=32):
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i:i+batch_size], return_tensors="pt",
            truncation=True, padding=True, max_length=MAX_LEN
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc)
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        probs.extend(p)
    return probs


texts = df["text"].astype(str).tolist()
y_true = df["label"].astype(int).tolist()
probs = predict_batch(texts)
y_pred = (np.array(probs) >= THRESHOLD).astype(int).tolist()

# Overall evaluation
print("\n=== Overall Classification Report ===")
print(classification_report(y_true, y_pred, digits=3))
print("\n=== Confusion matrix ===")
print(confusion_matrix(y_true, y_pred))
print("\n=== Macro-F1 (overall) ===")
print(f1_score(y_true, y_pred, average="macro"))


# Per-language evaluation
print("\n=== Per-language evaluation ===")
langs = df["lang"].astype(str).tolist()
for L in sorted(set(langs)):
    idx = [i for i, l in enumerate(langs) if l == L]
    if not idx:
        continue
    yt = np.array([y_true[i] for i in idx]) # get true labels for language L
    yp = np.array([y_pred[i] for i in idx]) # get predicted labels for language L
    acc = (yt == yp).mean()
    f1 = f1_score(yt, yp, average="macro")
    print(f"[{L}] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | support={len(idx)}")


# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from pathlib import Path

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("Output/PaperApproach_with_Class_weight_0.3") # path to the folder containing the model checkpoint and tokenizer
CKPT_PATH = MODEL_DIR / "xlmr_xling_best.pt"
TOK_PATH  = MODEL_DIR / "xlmr_xling_tok"    # tokenizer folder
BASE_NAME = "xlm-roberta-base"             # base encoder model
TEST_CSV  = Path("../../Data New/Pre processed Data/sold_test_clean.csv")
MAX_LEN   = 192
THRESHOLD = 0.38

if TOK_PATH.exists():
    tok = XLMRobertaTokenizerFast.from_pretrained(str(TOK_PATH))
    print(f"Loaded tokenizer from: {TOK_PATH}")
else:
    print("Tokenizer folder not found, falling back to base model tokenizer.")
    tok = XLMRobertaTokenizerFast.from_pretrained(BASE_NAME)

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")


class XlingPredictor(nn.Module):
    def __init__(self, model_name=BASE_NAME, num_labels=2):
        super().__init__()
        self.enc = XLMRobertaModel.from_pretrained(model_name)
        hidden = self.enc.config.hidden_size
        self.cls = nn.Linear(hidden, num_labels) #map the representation of token to logits

    def forward(self, **enc):
        out = self.enc(**enc)#encode the input text
        # Use masked mean pooling to match training evaluation
        mask = enc["attention_mask"].unsqueeze(-1)
        h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        logits = self.cls(h)
        return logits


model = XlingPredictor(BASE_NAME, num_labels=2)
state = torch.load(CKPT_PATH, map_location=DEVICE) #load the saved model checkpoint that contains the learned parameters (weights) of the model.

missing, unexpected = model.load_state_dict(state, strict=False) #load the model parameters from the checkpoint
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)

model.to(DEVICE).eval()
torch.set_grad_enabled(False) # turn off gradients calculation for inference


df = pd.read_csv(TEST_CSV)
needed = {"text", "label"}
if not needed.issubset(df.columns):
    raise ValueError(f"Test CSV missing columns: {needed - set(df.columns)}")

if "lang" not in df.columns:
    df["lang"] = "unk"

df = df.dropna(subset=["text", "label"]).reset_index(drop=True) # remove rows with missing text or label


@torch.no_grad()
def predict_batch(texts, batch_size=32):
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i:i+batch_size], return_tensors="pt",
            truncation=True, padding=True, max_length=MAX_LEN
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc)
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        probs.extend(p)
    return probs


texts = df["text"].astype(str).tolist()
y_true = df["label"].astype(int).tolist()
probs = predict_batch(texts)
y_pred = (np.array(probs) >= THRESHOLD).astype(int).tolist()

# Overall evaluation
print("\n=== Overall Classification Report ===")
print(classification_report(y_true, y_pred, digits=3))
print("\n=== Confusion matrix ===")
print(confusion_matrix(y_true, y_pred))
print("\n=== Macro-F1 (overall) ===")
print(f1_score(y_true, y_pred, average="macro"))


# Per-language evaluation
print("\n=== Per-language evaluation ===")
langs = df["lang"].astype(str).tolist()
for L in sorted(set(langs)):
    idx = [i for i, l in enumerate(langs) if l == L]
    if not idx:
        continue
    yt = np.array([y_true[i] for i in idx]) # get true labels for language L
    yp = np.array([y_pred[i] for i in idx]) # get predicted labels for language L
    acc = (yt == yp).mean()
    f1 = f1_score(yt, yp, average="macro")
    print(f"[{L}] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | support={len(idx)}")


# with Class weight Macro

