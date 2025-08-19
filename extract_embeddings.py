import os
import json
import argparse
import torch
import pytorch_lightning as pl

from architectures.TSMC import TSMC
from datasets.seed_dataset import SEEDDataModule
from datasets.seedIJCAI_dataset import SEEDIJCAIDataModule
from datasets.dreamer_dataset import DREAMERDataModule
from datasets.UCIHAR_dataset import UCIHARDataModule
from datasets.cho2017_dataset import Cho2017DataModule
from datasets.ucr_dataset import UCRDataModule
from functions.embeddings import compute_embeddings


def build_datamodule(args, device_params):
    dataset = args.dataset
    preprocessing = args.preprocessing
    num_workers = device_params['num_workers']
    if dataset == "SEED":
        dm = SEEDDataModule(
            device_params["ss_datapath"],
            args.train_val_split,
            preprocessing,
            "emotion",
            device_params['ss_batch_size'],
            num_workers
        )
    elif dataset == "SEEDIJCAI":
        dm = SEEDIJCAIDataModule(
            device_params['ss_emotion_ijcai_datapath'],
            preprocessing,
            device_params['ss_batch_size'],
            num_workers
        )
    elif dataset == "UCIHAR":
        dm = UCIHARDataModule(
            device_params["ss_ucihar_datapath"],
            preprocessing,
            device_params['ss_har_batch_size'],
            num_workers
        )
    elif dataset == "SEEDUC":
        dm = SEEDDataModule(
            device_params["ss_datapath"],
            args.train_val_split,
            preprocessing,
            "userID",
            device_params['ss_uc_batch_size'],
            num_workers
        )
    elif dataset == "Cho2017":
        dm = Cho2017DataModule(
            device_params["ss_mi_datapath"],
            preprocessing,
            device_params['ss_mi_batch_size'],
            num_workers
        )
    elif dataset == "DREAMER":
        dm = DREAMERDataModule(
            device_params["ss_vr_datapath"],
            preprocessing,
            device_params['ss_vr_batch_size'],
            num_workers
        )
    elif dataset == "UCR":
        dm = UCRDataModule(
            data_dir=device_params["ss_ucr_datapath"],
            dataset_name=device_params["ss_ucr_dataset_name"],
            batch_size=device_params['ss_ucr_batch_size'],
            num_workers=num_workers,
            q_split=0.15,
            seed=18,
            permute_indexes=True
        )
        dm.prepare_data()
        dm.setup()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return dm


def load_encoder(args, input_features):
    encoder = TSMC(
        pos_embeddings_alpha=args.pos_embeddings_alpha,
        input_features=input_features,
        embedding_dim=args.embedding_dim,
        n_head_token_enc=args.n_head_token_enc,
        n_head_context_enc=args.n_head_context_enc,
        depth_context_enc=args.depth_context_enc,
        max_predict_len=0,
    )
    return encoder


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings using a trained TS-MoCo encoder")
    parser.add_argument("dataset", choices=["SEED", "SEEDIJCAI", "UCIHAR", "SEEDUC", "Cho2017", "DREAMER", "UCR"], help="dataset name")
    parser.add_argument("pos_embeddings_alpha", type=float)
    parser.add_argument("embedding_dim", type=int)
    parser.add_argument("n_head_token_enc", type=int)
    parser.add_argument("n_head_context_enc", type=int)
    parser.add_argument("depth_context_enc", type=int)
    parser.add_argument("checkpoint", type=str, help="path to encoder checkpoint (.ckpt or .pt)")
    parser.add_argument("split", choices=["train", "val", "test"], help="which split to encode")
    parser.add_argument("train_val_split", choices=["random", "subject"], help="for SEED datasets")
    parser.add_argument("preprocessing", choices=["None", "standardize", "normalize"], help="preprocessing option")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device to run on")
    parser.add_argument("--save", default=None, help="output path to save embeddings (pt)")
    args = parser.parse_args()

    pl.seed_everything(33)
    with open("device_hyperparameters.json") as f:
        device_params = json.load(f)

    dm = build_datamodule(args, device_params)
    dm.setup(None)

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )

    encoder = load_encoder(args, dm.input_features)

    # Load checkpoint weights
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Filter keys if loaded from Lightning module
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("student."):
            new_state[k[len("student."):]] = v
        elif k.startswith("encoder."):
            new_state[k[len("encoder."):]] = v
        else:
            new_state[k] = v
    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    if len(unexpected) > 0:
        print(f"Warning: unexpected keys in state_dict: {unexpected}")
    if len(missing) > 0:
        print(f"Warning: missing keys in state_dict: {missing}")

    # Select dataloader
    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        # UCR uses q_dataloader naming; DataModules above alias val to q
        if hasattr(dm, "val_dataloader"):
            loader = dm.val_dataloader()
        else:
            loader = dm.q_dataloader()
    else:
        loader = dm.test_dataloader()

    embeddings, labels = compute_embeddings(encoder, loader, device=device)

    if args.save is None:
        # default save path
        root_dir = device_params['log_dir']
        os.makedirs(os.path.join(root_dir, 'embeddings'), exist_ok=True)
        base = f"{args.dataset}_emb_{args.split}.pt"
        out_path = os.path.join(root_dir, 'embeddings', base)
    else:
        out_path = args.save

    torch.save({
        'embeddings': embeddings,
        'labels': labels
    }, out_path)
    print(f"Saved embeddings to {out_path} with shape {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()


