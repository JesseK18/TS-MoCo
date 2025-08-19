import os
import torch
from torch.utils.data import DataLoader
from typing import Literal, Tuple


@torch.no_grad()
def compute_embeddings(
    encoder: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for an entire dataloader using the given encoder.

    The dataloader is expected to yield (inputs, labels).

    Returns:
        embeddings: (N, D)
        labels: (N,)
    """
    encoder.eval().to(device)

    all_embeds = []
    all_labels = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        x = x.to(device)
        # TS-MoCo encoder returns CLS embeddings via .encode
        if hasattr(encoder, "encode"):
            z = encoder.encode(x)
        else:
            # fall back to forward with K=0 and take first output as context
            z, _, _ = encoder(x, 0)
        all_embeds.append(z.detach().cpu())
        all_labels.append(y.detach().cpu())

    embeddings = torch.cat(all_embeds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


def print_embedding_info(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    split_name: str,
    class_names: list | None = None,
) -> None:
    """
    Print a concise summary of the computed embeddings and label distribution.
    """
    N, D = embeddings.shape
    device = str(embeddings.device)
    dtype = str(embeddings.dtype)

    print(f"=== Embeddings Summary ({split_name}) ===")
    print(f"shape: {tuple(embeddings.shape)} | dtype: {dtype} | device: {device}")
    print(f"num_samples: {N} | embedding_dim: {D}")

    # Global stats
    emb_min = torch.min(embeddings).item()
    emb_max = torch.max(embeddings).item()
    emb_mean = torch.mean(embeddings).item()
    emb_std = torch.std(embeddings).item()
    norms = torch.linalg.norm(embeddings, dim=1)
    print(f"value_range: [{emb_min:.4f}, {emb_max:.4f}] | mean: {emb_mean:.4f} | std: {emb_std:.4f}")
    print(f"l2_norms: mean={norms.mean().item():.4f}, std={norms.std().item():.4f}, min={norms.min().item():.4f}, max={norms.max().item():.4f}")

    # Per-dimension stats (only print fully if reasonably small)
    dim_means = torch.mean(embeddings, dim=0)
    dim_stds = torch.std(embeddings, dim=0)
    if D <= 64:
        print(f"per-dim mean (len={D}): {dim_means.tolist()}")
        print(f"per-dim std  (len={D}): {dim_stds.tolist()}")
    else:
        head = min(8, D)
        tail = min(8, D - head)
        means_preview = dim_means[:head].tolist() + (["..."] if tail > 0 else []) + (dim_means[-tail:].tolist() if tail > 0 else [])
        stds_preview = dim_stds[:head].tolist() + (["..."] if tail > 0 else []) + (dim_stds[-tail:].tolist() if tail > 0 else [])
        print(f"per-dim mean preview: {means_preview}")
        print(f"per-dim std  preview: {stds_preview}")

    # Label distribution
    unique, counts = torch.unique(labels, return_counts=True)
    print("labels distribution:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        name = class_names[u] if (class_names is not None and u < len(class_names)) else str(u)
        print(f" - class {name}: {c}")
    print("===============================")



