# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture --no-stderr
# !pip install numpy nnsight torch scikit-learn

import gc
import os

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

model = LanguageModel("google/gemma-3-270m-it", device_map="auto", dispatch=True)

LAYER = 8

def main():
    """### 1. Record activations on the dataset."""

    ds_harmful = load_dataset("bench-llm/or-bench", "or-bench-toxic", split="train")
    ds_harmful = ds_harmful.shuffle(seed=0)
    ds_harmful = ds_harmful.select(range(300))
    ds_harmful = ds_harmful.select_columns(["prompt"])
    ds_harmful = ds_harmful.map(lambda x: {"label": 1})

    ds_harmless = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
    ds_harmless = ds_harmless.shuffle(seed=0)
    ds_harmless = ds_harmless.select(range(len(ds_harmful)))
    ds_harmless = ds_harmless.select_columns(["prompt"])
    ds_harmless = ds_harmless.map(lambda x: {"label": 0})

    ds_combined = concatenate_datasets([ds_harmful, ds_harmless])

    gc.collect()
    torch.cuda.empty_cache()

    saved = []
    BATCH_SIZE = 16  # Process in batches to prevent OOM

    prompts = ds_combined["prompt"]

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        batch_saved = []

        # Trace a small batch of prompts
        with model.trace() as tracer:
            for t in batch_prompts:
                with tracer.invoke(t):
                    # Save the hidden state at the last token
                    h = model.model.layers[LAYER].output[0][0, -1, :].save()
                    batch_saved.append(h)

        # Move results to CPU immediately to free GPU memory
        for h in batch_saved:
            saved.append(h.detach().cpu())

        # Cleanup
        del batch_saved
        torch.cuda.empty_cache()

    H = torch.stack(saved).numpy()  # [N, d_model]

    gc.collect()
    torch.cuda.empty_cache()

    """### 2. Train a linear probe on the activations."""

    ds_combined = ds_combined.with_format("numpy")

    Xtr, Xte, ytr, yte = train_test_split(H,
                                        [int(label) for label in ds_combined["label"]],
                                        test_size=0.2,
                                        stratify=[int(label) for label in ds_combined["label"]],
                                        random_state=0)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)

    logits = clf.decision_function(Xte)

    print("Probe Accuracy (Test):", clf.score(Xte, yte))

    w = clf.coef_[0]  # [d_model]
    b = clf.intercept_

    def plot_kde(positive_scores, negative_scores):
        plt.figure(figsize=(10, 6))
        # Plotting positive scores
        sns.kdeplot(positive_scores, fill=True, bw_adjust=0.1,  # specify bandwidth here
                    color='darkblue', label='Positive')
        # Plotting negative scores
        sns.kdeplot(negative_scores, fill=True, bw_adjust=0.1,  # specify bandwidth here
                    color='darkred', label='Negative')
        # Adding legend, title, and labels
        plt.legend(prop={'size': 16}, title='Scores')
        plt.title('Logit Distribution for Positive and Negative Examples')
        plt.xlabel('Logit')
        plt.ylabel('Density')
        plt.show()

    logits_for_harmful = [logits[i] for i in range(len(yte)) if yte[i] == 1]
    logits_for_harmless = [logits[i] for i in range(len(yte)) if yte[i] == 0]

    plot_kde(logits_for_harmful, logits_for_harmless)

    """### 3. Quantize weights, bias, and activations."""

    # Symmetric int16 Quantization of Weights

    Sw = 10_000

    quantized_w = np.clip(np.round(w * Sw),
                        -np.iinfo(np.int16).max,
                        np.iinfo(np.int16).max).astype(np.int16)

    sns.kdeplot(quantized_w)
    plt.title("Quantized Weights")
    plt.show()

    sns.kdeplot(np.hstack(Xtr))
    plt.title("Activations (Training)")
    plt.show()

    # Quantization of Activations

    Sx = 100

    p999_abs = np.percentile(np.abs(np.hstack(Xtr)), 99.9)

    quantized_Xte = np.clip(np.round(np.clip(Xte, -p999_abs, p999_abs) * Sx),
                            -np.iinfo(np.int16).max,
                            np.iinfo(np.int16).max).astype(np.int16)

    print(quantized_Xte)

    # Quantization of Bias

    quantized_b = np.round(b * Sx * Sw).astype(np.int64)

    print(quantized_b)

if __name__ == "__main__":
    main()
