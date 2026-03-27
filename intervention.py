"""
Shuffle-k: Dataset → Transformer → Probe → Experiment 1
=========================================================

Run end-to-end:
    python shuffle_k.py --k 2 --n 10000 --epochs 25

Stages:
  1. Generate Shuffle-k dataset
  2. Train transformer (next-token k-hot prediction, MSE + sigmoid)
  3. For each stack i: train linear probe on hidden states → predict depth
     Uses BALANCED ACCURACY (mean per-class recall) to avoid majority-class
     inflation in both task and control probes.
     Control probe: embeddings are SHUFFLED before training so the probe
     cannot learn any embedding→depth correspondence, then evaluated on
     real (unshuffled) val labels so accuracy reflects genuine generalisation.
  4. Experiment 1: ablation sweep over alpha for each stack's probe direction.
     Plots seq_acc and pos_acc vs alpha for probe_w and random_w only.
"""

import json
import math
import random
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Shared constants
# ============================================================

BRACKET_PAIRS = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
    ("⟦", "⟧"),
    ("⟨", "⟩"),
    ("⌈", "⌉"),
    ("⌊", "⌋"),
]


# ============================================================
# 1. DATASET GENERATION
# ============================================================

def generate_one(k, length, brackets, rng, close_prob=0.6):
    """
    Generate one valid Shuffle-k string of exactly `length` tokens.
    Returns (string, depths) or None if length is odd.

    depths[i][t] = depth of stack i AFTER the t-th token (0-indexed).

    close_prob=0.6 biases the walk toward closing sooner, spreading
    depth values more evenly across the dataset.
    """
    if length % 2 != 0:
        return None

    counters      = [0] * k
    tokens        = []
    depth_history = [[] for _ in range(k)]

    for pos in range(length):
        remaining  = length - pos
        total_open = sum(counters)
        must_close = (total_open == remaining)
        must_open  = (total_open == 0)

        if must_close:
            open_stacks = [i for i in range(k) if counters[i] > 0]
            i = rng.choice(open_stacks)
            tokens.append(brackets[i][1])
            counters[i] -= 1
        elif must_open:
            i = rng.randint(0, k - 1)
            tokens.append(brackets[i][0])
            counters[i] += 1
        else:
            if rng.random() < close_prob:
                open_stacks = [i for i in range(k) if counters[i] > 0]
                i = rng.choice(open_stacks)
                tokens.append(brackets[i][1])
                counters[i] -= 1
            else:
                i = rng.randint(0, k - 1)
                tokens.append(brackets[i][0])
                counters[i] += 1

        for i in range(k):
            depth_history[i].append(counters[i])

    if any(c != 0 for c in counters):
        return None

    return "".join(tokens), depth_history


def build_dataset(n, k, min_len=2, max_len=50, seed=42):
    """
    Generate n valid Shuffle-k samples.
    Each sample: {"string": str, "depths": list[list[int]]}
      depths[i][t] = depth of stack i after token t.
    """
    brackets  = BRACKET_PAIRS[:k]
    rng       = random.Random(seed)
    even_lens = [l for l in range(min_len, max_len + 1) if l % 2 == 0]
    assert even_lens, "No even lengths in [min_len, max_len]."

    dataset, attempts = [], 0
    while len(dataset) < n and attempts < n * 20:
        attempts += 1
        result = generate_one(k, rng.choice(even_lens), brackets, rng)
        if result is not None:
            string, depths = result
            dataset.append({"string": string, "depths": depths})

    if len(dataset) < n:
        print(f"Warning: generated only {len(dataset)}/{n} samples.")
    return dataset


# ============================================================
# 2. TRANSFORMER
# ============================================================

def build_vocab(k):
    """<PAD>=0, <BOS>=1, then 2k bracket tokens."""
    tok2id = {"<PAD>": 0, "<BOS>": 1}
    for i in range(k):
        tok2id[BRACKET_PAIRS[i][0]] = len(tok2id)
        tok2id[BRACKET_PAIRS[i][1]] = len(tok2id)
    id2tok = {v: kk for kk, v in tok2id.items()}
    return tok2id, id2tok


class ShuffleKDataset(Dataset):
    """
    Each sample is one valid Shuffle-k string of length n.

    Produces (input_ids, targets, seq_len):
      input_ids : [BOS, t_0, ..., t_{n-1}]           length n+1
      targets   : k-hot vectors for positions 0..n-1  shape (n, V)
        targets[p] = valid-next-token set after seeing input_ids[0..p]
      seq_len   : n+1   (for padding/masking)
    """

    def __init__(self, samples, tok2id, k):
        self.tok2id    = tok2id
        self.k         = k
        self.V         = len(tok2id)
        self.open_ids  = [tok2id[BRACKET_PAIRS[i][0]] for i in range(k)]
        self.close_ids = [tok2id[BRACKET_PAIRS[i][1]] for i in range(k)]
        self.open_ch   = [BRACKET_PAIRS[i][0] for i in range(k)]
        self.close_ch  = [BRACKET_PAIRS[i][1] for i in range(k)]

        self.data = []
        for sample in samples:
            string    = sample["string"]
            token_ids = [tok2id["<BOS>"]] + [tok2id[c] for c in string]
            targets   = self._make_targets(string)
            self.data.append((token_ids, targets, len(token_ids)))

    def _make_targets(self, string):
        counters = [0] * self.k
        targets  = [self._khot(counters)]
        for char in string[:-1]:
            if char in self.open_ch:
                counters[self.open_ch.index(char)] += 1
            else:
                counters[self.close_ch.index(char)] -= 1
            targets.append(self._khot(counters))
        return targets

    def _khot(self, counters):
        v = [0.0] * self.V
        for i in range(self.k):
            v[self.open_ids[i]] = 1.0
            if counters[i] > 0:
                v[self.close_ids[i]] = 1.0
        return v

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def collate_fn(batch, vocab_size):
    sequences, targets, lengths = zip(*batch)
    T_max = max(lengths)
    B, V  = len(sequences), vocab_size

    input_ids = torch.zeros(B, T_max,     dtype=torch.long)
    tgt_tens  = torch.zeros(B, T_max - 1, V)
    mask      = torch.zeros(B, T_max - 1, dtype=torch.bool)

    for i, (ids, tgts, length) in enumerate(zip(sequences, targets, lengths)):
        input_ids[i, :length]  = torch.tensor(ids, dtype=torch.long)
        tgt_len                = length - 1
        tgt_tens[i, :tgt_len]  = torch.tensor(tgts)
        mask[i, :tgt_len]      = True

    return input_ids, tgt_tens, mask


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        causal = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.qkv(x).split(C, dim=-1)

        def rshp(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = rshp(Q), rshp(K), rshp(V)
        scale   = math.sqrt(self.d_head)
        attn    = (Q @ K.transpose(-2, -1)) / scale
        attn    = attn.masked_fill(self.causal_mask[:T, :T][None, None], float("-inf"))
        attn    = torch.softmax(attn, dim=-1)
        out     = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CounterTransformer(nn.Module):
    """
    Encoder-only transformer with causal mask.

    Invariant: sigmoid(decoder(get_hidden(x))) == forward(x)
    So ablating get_hidden's output and re-running decoder is a valid
    intervention — the transformer's internal computations are unchanged.
    """

    def __init__(self, vocab_size, d_embed=32, d_model=64,
                 n_layers=1, n_heads=4, d_ffn=64, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.embed_proj  = nn.Linear(d_embed, d_model, bias=False)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.layers      = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ffn) for _ in range(n_layers)]
        )
        self.norm    = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def _encode(self, input_ids):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embed_proj(self.token_embed(input_ids)) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(self, input_ids):
        return torch.sigmoid(self.decoder(self._encode(input_ids)))

    def get_hidden(self, input_ids):
        """Returns (B, T, d_model). sigmoid(decoder(get_hidden(x))) == forward(x)."""
        return self._encode(input_ids)


def pos_accuracy(preds, targets, mask):
    correct = (((preds > 0.5).float() == targets).all(dim=-1) & mask)
    return correct.sum().item() / mask.sum().item()


def seq_accuracy(preds, targets, mask):
    correct_pos = ((preds > 0.5).float() == targets).all(dim=-1)
    seq_correct = (correct_pos | ~mask).all(dim=-1)
    return seq_correct.sum().item() / len(seq_correct)


def train_transformer(model, train_loader, val_loader, epochs, lr, device):
    opt     = torch.optim.RMSprop(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_acc, best_sd = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = t_acc = n = 0
        for ids, tgts, mask in train_loader:
            ids, tgts, mask = ids.to(device), tgts.to(device), mask.to(device)
            preds = model(ids)[:, :-1, :]
            loss  = loss_fn(preds[mask], tgts[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item()
            t_acc  += pos_accuracy(preds, tgts, mask)
            n += 1

        model.eval()
        v_loss = v_acc = m = 0
        with torch.no_grad():
            for ids, tgts, mask in val_loader:
                ids, tgts, mask = ids.to(device), tgts.to(device), mask.to(device)
                preds   = model(ids)[:, :-1, :]
                v_loss += loss_fn(preds[mask], tgts[mask]).item()
                v_acc  += pos_accuracy(preds, tgts, mask)
                m += 1

        va = v_acc / m
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss={t_loss/n:.4f} acc={t_acc/n:.4f} | "
              f"val loss={v_loss/m:.4f} acc={va:.4f}")
        if va > best_acc:
            best_acc = va
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    print(f"Best val acc: {best_acc:.4f}")
    return model


# ============================================================
# 3. PROBE
# ============================================================

def extract_embeddings(model, samples, tok2id, k, stack_idx, device):
    """
    Extract (hidden_state, stack_depth) pairs for every token position.

    Alignment:
      input_ids = [BOS, t_0, ..., t_{n-1}]   length n+1
      H[0, t+1, :] = hidden state AFTER token t
      depths[stack_idx][t] = depth AFTER token t
      → pair H[0, t+1, :] with depths[stack_idx][t]
    """
    model.eval()
    embs, lbls = [], []
    with torch.no_grad():
        for sample in samples:
            string = sample["string"]
            depths = sample["depths"][stack_idx]
            ids    = [tok2id["<BOS>"]] + [tok2id[c] for c in string]
            H      = model.get_hidden(
                torch.tensor([ids], dtype=torch.long, device=device)
            )
            for t, depth in enumerate(depths):
                embs.append(H[0, t + 1, :])
                lbls.append(depth)
    return torch.stack(embs), torch.tensor(lbls, dtype=torch.long)


class EmbDataset(Dataset):
    def __init__(self, embs, lbls):
        self.embs, self.lbls = embs, lbls
    def __len__(self): return len(self.embs)
    def __getitem__(self, i): return self.embs[i], self.lbls[i]


def _balanced_accuracy(all_preds, all_labels, n_classes):
    """
    Mean per-class recall (balanced accuracy).

    A probe that always predicts the majority class scores 1/n_classes,
    regardless of how imbalanced the depth label distribution is.
    This is the correct metric when depth 0 and 1 heavily dominate.
    """
    per_class_correct = torch.zeros(n_classes)
    per_class_total   = torch.zeros(n_classes)
    for c in range(n_classes):
        mask = (all_labels == c)
        per_class_total[c]   = mask.sum().item()
        per_class_correct[c] = (all_preds[mask] == c).sum().item()
    present          = per_class_total > 0
    recall_per_class = per_class_correct[present] / per_class_total[present]
    return recall_per_class.mean().item()


def _fit_linear_probe(d_model, n_classes, tr_embs, tr_lbls, vl_embs, vl_lbls, device):
    """
    Train a linear probe; return (best_balanced_val_acc, trained_probe).

    Balanced accuracy is used so that class imbalance (many depth-0 tokens)
    does not inflate either task or control accuracy.
    """
    probe = nn.Linear(d_model, n_classes)
    nn.init.xavier_uniform_(probe.weight)
    probe      = probe.to(device)
    opt        = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn    = nn.CrossEntropyLoss()
    tr_ld      = DataLoader(EmbDataset(tr_embs, tr_lbls), batch_size=32, shuffle=True)
    vl_ld      = DataLoader(EmbDataset(vl_embs, vl_lbls), batch_size=32, shuffle=False)

    best_acc   = 0.0
    best_state = None
    for _ in range(10):
        probe.train()
        for e, l in tr_ld:
            e, l = e.to(device), l.to(device)
            opt.zero_grad(); loss_fn(probe(e), l).backward(); opt.step()

        probe.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for e, l in vl_ld:
                e, l = e.to(device), l.to(device)
                all_preds.append(probe(e).argmax(-1).cpu())
                all_labels.append(l.cpu())
        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = _balanced_accuracy(all_preds, all_labels, n_classes)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return best_acc, probe


def run_probe(embeddings, labels, stack_idx, device, seed=42):
    """
    Train task and control probes; return (task_acc, ctrl_acc, w).

    Task probe:
      Trained on real (embedding, depth) pairs.

    Control probe:
      The embeddings themselves are shuffled (row-permuted) before training,
      so the probe cannot learn any correspondence between a specific hidden
      state and its depth label. Labels stay fixed. The probe is then
      evaluated on the original (unshuffled) val embeddings with real labels.

      This is strictly stronger than label-shuffling: shuffling embeddings
      ensures the training input is independent of the training label at
      every position, so the only signal available is the marginal label
      distribution. Under balanced accuracy, even that signal yields 1/C.

    w: top right singular vector of task probe weight matrix W (n_classes x d_model).
    """
    torch.manual_seed(seed)
    n     = len(embeddings)
    idx   = torch.randperm(n)
    split = int(0.8 * n)

    tr_embs = embeddings[idx[:split]]
    tr_lbls = labels[idx[:split]]
    vl_embs = embeddings[idx[split:]]
    vl_lbls = labels[idx[split:]]       # REAL labels for val in both probes

    n_classes = int(labels.max().item()) + 1
    d_model   = embeddings.shape[1]

    # Task probe
    task_acc, task_probe = _fit_linear_probe(
        d_model, n_classes, tr_embs, tr_lbls, vl_embs, vl_lbls, device
    )

    # Control probe: shuffle the EMBEDDINGS (not labels) in the training set.
    # This breaks every (embedding, label) pair while preserving the marginal
    # distributions of both embeddings and labels independently.
    ctrl_tr_embs = tr_embs[torch.randperm(len(tr_embs))]
    ctrl_acc, _  = _fit_linear_probe(
        d_model, n_classes, ctrl_tr_embs, tr_lbls, vl_embs, vl_lbls, device
    )

    # Extract direction w from task probe weight matrix
    W = task_probe.weight.detach()          # (n_classes, d_model)
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    w = Vh[0] / Vh[0].norm()               # (d_model,) unit vector

    print(f"\n  Stack {stack_idx}:")
    print(f"    Task balanced acc    = {task_acc*100:.1f}%")
    print(f"    Control balanced acc = {ctrl_acc*100:.1f}%")
    print(f"    Selectivity          = {(task_acc - ctrl_acc)*100:.1f}%")

    return task_acc, ctrl_acc, w


# ============================================================
# 4. EXPERIMENT 1 — LINEAR SUBSPACE ABLATION
# ============================================================

def proj(h, w):
    """
    Project h onto unit vector w.
    h : (..., d)   w : (d,)
    Returns (..., d): component of h along w.
    Handles arbitrary leading dimensions via broadcasting.
    """
    return (h @ w).unsqueeze(-1) * w


def _build_eval_batch(samples, tok2id, k, vocab_size, device):
    """
    Build padded (input_ids, targets, mask) for inference-time evaluation.
    Targets computed from ground-truth counter machine.
    """
    open_ch   = [BRACKET_PAIRS[i][0] for i in range(k)]
    close_ch  = [BRACKET_PAIRS[i][1] for i in range(k)]
    open_ids  = [tok2id[c] for c in open_ch]
    close_ids = [tok2id[c] for c in close_ch]

    def khot(counters):
        v = [0.0] * vocab_size
        for i in range(k):
            v[open_ids[i]] = 1.0
            if counters[i] > 0:
                v[close_ids[i]] = 1.0
        return v

    all_ids, all_tgts, all_lens = [], [], []
    for sample in samples:
        string   = sample["string"]
        counters = [0] * k
        tgts     = [khot(counters)]
        for char in string[:-1]:
            if char in open_ch:
                counters[open_ch.index(char)] += 1
            else:
                counters[close_ch.index(char)] -= 1
            tgts.append(khot(counters))
        ids = [tok2id["<BOS>"]] + [tok2id[c] for c in string]
        all_ids.append(ids)
        all_tgts.append(tgts)
        all_lens.append(len(ids))

    T_max     = max(all_lens)
    B         = len(samples)
    input_ids = torch.zeros(B, T_max,     dtype=torch.long,  device=device)
    tgt_tens  = torch.zeros(B, T_max - 1, vocab_size,        device=device)
    mask      = torch.zeros(B, T_max - 1, dtype=torch.bool,  device=device)

    for i, (ids, tgts, length) in enumerate(zip(all_ids, all_tgts, all_lens)):
        input_ids[i, :length]  = torch.tensor(ids, dtype=torch.long, device=device)
        tgt_len                = length - 1
        tgt_tens[i, :tgt_len]  = torch.tensor(tgts, device=device)
        mask[i, :tgt_len]      = True

    return input_ids, tgt_tens, mask


def run_experiment_1(model, samples, tok2id, k, vocab_size,
                     w, embeddings, labels, stack_idx, device, seed=42):
    """
    Necessity test for stack `stack_idx`.

    Sweeps alpha in [0, 0.25, 0.5, 0.75, 1.0]:
      H     = model.get_hidden(input_ids)
      H'    = H - alpha * proj(H, direction)
      preds = sigmoid(model.decoder(H'))[:, :-1, :]
      measures pos_accuracy and seq_accuracy

    Two directions:
      probe_w  — real direction from probe (should degrade accuracy)
      random_w — random unit vector        (should not degrade accuracy)

    Intervention correctness:
      H is the post-LayerNorm residual stream, before the linear decoder.
      Only model.decoder is re-run after ablation. For this 1-layer model
      this is the complete internal representation.
    """
    torch.manual_seed(seed)

    input_ids, targets, mask = _build_eval_batch(samples, tok2id, k, vocab_size, device)

    # Activation diagnostics
    with torch.no_grad():
        H_diag = model.get_hidden(input_ids)

    B, T, d = H_diag.shape
    mask_H  = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask_H[:, 1:] = mask
    H_flat  = H_diag[mask_H]

    means         = H_flat.mean(dim=0)
    stds          = H_flat.std(dim=0)
    abs_mean      = H_flat.abs().mean(dim=0)
    frac_zero     = (H_flat.abs() < 0.01).float().mean(dim=0)
    dead_neurons  = (frac_zero > 0.99).sum().item()
    low_var       = (stds < 0.01).sum().item()
    high_var      = (stds > 0.1).sum().item()
    proj_onto_w   = H_flat @ w
    var_total     = H_flat.var(dim=0).sum().item()
    var_along_w   = proj_onto_w.var().item()

    print(f"\n---- Activation diagnostics (stack {stack_idx}) ----")
    print(f"  Valid positions : {H_flat.shape[0]}")
    print(f"  Mean |act|      : {abs_mean.mean().item():.4f}  std across neurons: {abs_mean.std().item():.4f}")
    print(f"  Mean std/neuron : {stds.mean().item():.4f}  std across neurons: {stds.std().item():.4f}")
    print(f"  Dead neurons    : {dead_neurons}/{d}")
    print(f"  Low-var (<0.01) : {low_var}/{d}")
    print(f"  High-var (>0.1) : {high_var}/{d}")
    print(f"  Var along w     : {var_along_w:.4f} / {var_total:.4f} total  ({var_along_w/var_total*100:.2f}%)")

    w_random = torch.randn_like(w)
    w_random = w_random / w_random.norm()

    directions = {
        "probe_w":  w,
        "random_w": w_random,
    }
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 — Ablation (Stack {stack_idx})")
    print(f"{'='*60}")
    print(f"  {'Alpha':>6}  {'Direction':>10}  {'Pos Acc':>9}  {'Seq Acc':>9}")
    print("  " + "-" * 42)

    results = {}
    for dir_name, direction in directions.items():
        results[dir_name] = []
        for alpha in alphas:
            with torch.no_grad():
                H     = model.get_hidden(input_ids)
                H_abl = H - alpha * proj(H, direction)
                preds = torch.sigmoid(model.decoder(H_abl))[:, :-1, :]
                p_acc = pos_accuracy(preds, targets, mask)
                s_acc = seq_accuracy(preds, targets, mask)

            print(f"  {alpha:>6.2f}  {dir_name:>10}  {p_acc*100:>8.2f}%  {s_acc*100:>8.2f}%")
            results[dir_name].append({
                "alpha":   alpha,
                "pos_acc": round(p_acc * 100, 2),
                "seq_acc": round(s_acc * 100, 2),
            })
        print()

    return results


# ============================================================
# 5. PLOTTING
# ============================================================

def plot_results(all_exp_results, alphas, k, output_path):
    """
    Save two PNG figures next to output_path:
      {base}_seq_acc.png  — seq_acc vs alpha, one subplot per stack
      {base}_pos_acc.png  — pos_acc vs alpha, one subplot per stack
    Each subplot shows probe_w (blue solid) and random_w (orange dashed).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir   = os.path.dirname(os.path.abspath(output_path))
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    dir_styles = {
        "probe_w":  ("tab:blue",   "-",  "probe $\\mathbf{w}$"),
        "random_w": ("tab:orange", "--", "random $\\mathbf{w}$"),
    }

    for metric, ylabel in [("seq_acc", "Sequence Accuracy (%)"),
                            ("pos_acc", "Position Accuracy (%)")]:
        ncols = min(k, 3)
        nrows = math.ceil(k / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle(f"{ylabel} vs $\\alpha$ — Shuffle-{k}",
                     fontsize=13, y=1.01)

        for stack_idx in range(k):
            row, col = divmod(stack_idx, ncols)
            ax       = axes[row][col]
            exp      = all_exp_results[f"stack_{stack_idx}"]

            for dir_name, (color, ls, label) in dir_styles.items():
                if dir_name not in exp:
                    continue
                vals = [e[metric] for e in exp[dir_name]]
                ax.plot(alphas, vals, color=color, linestyle=ls,
                        linewidth=2, marker="o", markersize=5, label=label)

            ax.set_title(f"Stack {stack_idx}", fontsize=11)
            ax.set_xlabel("$\\alpha$", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(alphas)
            ax.set_ylim(-5, 105)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        for idx in range(k, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{base_name}_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {save_path}")


# ============================================================
# 6. MAIN
# ============================================================

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Dataset
    print(f"\n[1/3] Generating Shuffle-{args.k} dataset ({args.n} samples)...")
    dataset = build_dataset(args.n, args.k, min_len=2, max_len=50, seed=args.seed)
    print(f"      {len(dataset)} samples generated.")

    # 2. Transformer
    print(f"\n[2/3] Training transformer ({args.epochs} epochs)...")
    tok2id, id2tok = build_vocab(args.k)
    vocab_size     = len(tok2id)

    random.shuffle(dataset)
    split        = int(0.8 * len(dataset))
    cfn          = lambda b: collate_fn(b, vocab_size)
    train_loader = DataLoader(
        ShuffleKDataset(dataset[:split], tok2id, args.k),
        batch_size=32, shuffle=True, collate_fn=cfn
    )
    val_loader   = DataLoader(
        ShuffleKDataset(dataset[split:], tok2id, args.k),
        batch_size=32, shuffle=False, collate_fn=cfn
    )
    model = CounterTransformer(
        vocab_size=vocab_size, d_embed=32, d_model=64,
        n_layers=1, n_heads=4, d_ffn=64,
    ).to(device)
    model = train_transformer(model, train_loader, val_loader,
                              epochs=args.epochs, lr=5e-3, device=device)

    # 3. Probe + Experiment 1 for every stack
    print(f"\n[3/3] Probing and ablation for all {args.k} stacks...")
    probe_samples = dataset[:10000]
    exp_samples   = dataset[:500]
    alphas        = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_probe_results = {}
    all_exp_results   = {}

    for stack_idx in range(args.k):
        print(f"\n{'='*60}")
        print(f"  Stack {stack_idx + 1} / {args.k}")
        print(f"{'='*60}")

        embeddings, labels = extract_embeddings(
            model, probe_samples, tok2id, args.k,
            stack_idx=stack_idx, device=device
        )
        print(f"  {len(embeddings)} (embedding, label) pairs extracted.")

        task_acc, ctrl_acc, w = run_probe(
            embeddings, labels, stack_idx=stack_idx,
            device=device, seed=args.seed
        )
        all_probe_results[f"stack_{stack_idx}"] = {
            "task_balanced_acc":    round(task_acc * 100, 2),
            "control_balanced_acc": round(ctrl_acc * 100, 2),
            "selectivity":          round((task_acc - ctrl_acc) * 100, 2),
        }

        exp_results = run_experiment_1(
            model      = model,
            samples    = exp_samples,
            tok2id     = tok2id,
            k          = args.k,
            vocab_size = vocab_size,
            w          = w,
            embeddings = embeddings,
            labels     = labels,
            stack_idx  = stack_idx,
            device     = device,
            seed       = args.seed,
        )
        all_exp_results[f"stack_{stack_idx}"] = exp_results

    # Save JSON
    out = {
        "k":            args.k,
        "probe":        all_probe_results,
        "experiment_1": all_exp_results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Plot
    print("\nGenerating plots...")
    plot_results(all_exp_results, alphas, args.k, args.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k",      type=int,   default=2,              help="Number of stacks")
    p.add_argument("--n",      type=int,   default=10000,          help="Dataset size")
    p.add_argument("--epochs", type=int,   default=25,             help="Training epochs")
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--output", type=str,   default="results.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()