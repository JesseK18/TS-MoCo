import os
import copy
import json
import logging
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from architectures.TSMC import TSMC
from architectures.classifier import DenseClassifier
from datasets.UCIHAR_dataset import UCIHARDataModule
from datasets.cho2017_dataset import Cho2017DataModule
from modules.encoding_module import plEncodingModule
from modules.classification_module import plClassificationModule
from datasets.seed_dataset import SEEDDataModule
from datasets.seedIJCAI_dataset import SEEDIJCAIDataModule
from datasets.dreamer_dataset import DREAMERDataModule
from datasets.ucr_dataset import UCRDataModule
from utils.restricted_float import restricted_float
from functions.embeddings import compute_embeddings, print_embedding_info
#from knockknock import email_sender
#from dotenv import load_dotenv

#load_dotenv()
#@email_sender(recipient_emails=[os.environ.get("recipient_emails")], sender_email=os.environ.get("sender_email"))
def main_supervised(args):
    logging.getLogger("lightning").setLevel(logging.WARNING)
    pl.seed_everything(7)

    ### CONFIG AND HYPERPARAMETERS
    with open("device_hyperparameters.json") as f:
        device_params = json.load(f)
    log_dir = device_params['log_dir']
    num_workers = device_params['num_workers']
    limit_train_batches = device_params['limit_train_batches']
    limit_val_batches = device_params['limit_val_batches']
    limit_test_batches = device_params['limit_test_batches']

    if args.dataset=="SEED":
        run_name = "emotion_recognition_supervised"
        datamodule = SEEDDataModule(
            device_params["ss_datapath"],
            args.train_val_split,
            args.preprocessing,
            "emotion",
            device_params['ss_batch_size'],
            num_workers
        )
    elif args.dataset=="SEEDIJCAI":
        run_name = "emotion_recognition_ijcai"
        datamodule = SEEDIJCAIDataModule(
            device_params['ss_emotion_ijcai_datapath'],
            args.preprocessing,
            device_params['ss_batch_size'],
            num_workers
        )
    elif args.dataset=="UCIHAR":
        run_name = "activity_recognition_supervised"
        datamodule = UCIHARDataModule(
            device_params["ss_ucihar_datapath"],
            args.preprocessing,
            device_params['ss_har_batch_size'],
            num_workers
        )
    elif args.dataset=="SEEDUC":
        run_name = "user_recognition_supervised"
        datamodule = SEEDDataModule(
            device_params["ss_datapath"],
            args.train_val_split,
            args.preprocessing,
            "userID",
            device_params['ss_uc_batch_size'],
            num_workers
        )
    elif args.dataset=="Cho2017":
        run_name = "motor_imagery_supervised"
        datamodule = Cho2017DataModule(
            device_params["ss_mi_datapath"],
            args.preprocessing,
            device_params['ss_mi_batch_size'],
            num_workers
        )
    elif args.dataset=="DREAMER":
        run_name = "valence_recognition_supervised"
        datamodule = DREAMERDataModule(
            device_params["ss_vr_datapath"],
            args.preprocessing,
            device_params['ss_vr_batch_size'],
            num_workers
        )
    elif args.dataset=="UCR":
        run_name = "ucr_classification_supervised"
        datamodule = UCRDataModule(
            data_dir=device_params["ss_ucr_datapath"],
            dataset_name=device_params["ss_ucr_dataset_name"],
            batch_size=device_params['ss_ucr_batch_size'],
            num_workers=num_workers,
            q_split=0.15,
            seed=18,
            permute_indexes=True
        )
        datamodule.prepare_data()
        datamodule.setup()
    else:
        raise ValueError(f'parameter dataset has to be one of ["SEED", "SEEDIJCAI", "UCIHAR", "SEEDUC", "Cho2017", "DREAMER", "UCR"], but got {args.dataset}')
    encoder = TSMC(
        pos_embeddings_alpha=args.pos_embeddings_alpha,
        input_features=datamodule.input_features,
        embedding_dim=args.embedding_dim,
        n_head_token_enc=args.n_head_token_enc,
        n_head_context_enc=args.n_head_context_enc,
        depth_context_enc=args.depth_context_enc,
        max_predict_len=0
        )
    classifier = DenseClassifier(in_features=args.embedding_dim, out_features=datamodule.n_classes)

    enc_classifier = plClassificationModule(
        encoder,
        classifier,
        datamodule.batch_size,
        args.lr,
        num_workers,
        freeze_encoder=False
    )

    supervised_trainer_checkpoint_callback = ModelCheckpoint(monitor="val_loss",dirpath=f"{log_dir}/checkpoints/{run_name}_classification", save_last=True)
    supervised_trainer_csv_logger = CSVLogger(save_dir=f"{log_dir}/csv/", name=f"{run_name}_classification")
    supervised_trainer = pl.Trainer(
        accelerator = "auto",
        default_root_dir=log_dir,
        max_epochs=args.finetune_epochs,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=args.es_after_epochs),
            supervised_trainer_checkpoint_callback
        ],
        logger=[
            supervised_trainer_csv_logger,
            TensorBoardLogger(save_dir=f"{log_dir}/tb/", name=f"{run_name}_classification", log_graph=False, default_hp_metric=False)
        ],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    supervised_trainer.fit(enc_classifier, datamodule)
    
    with open(os.path.join(supervised_trainer_csv_logger.log_dir,'best_model_path.txt'), 'w') as f:
        f.write(supervised_trainer_checkpoint_callback.best_model_path)

    # === Compute and print embeddings after supervised training ===
    try:
        extract_encoder = copy.deepcopy(enc_classifier.encoder).eval()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader() if hasattr(datamodule, 'val_dataloader') else datamodule.q_dataloader()
        test_loader = datamodule.test_dataloader()

        train_emb, train_lbl = compute_embeddings(extract_encoder, train_loader, device="cpu")
        val_emb, val_lbl = compute_embeddings(extract_encoder, val_loader, device="cpu")
        test_emb, test_lbl = compute_embeddings(extract_encoder, test_loader, device="cpu")

        class_names = getattr(datamodule, 'class_names', None)
        print_embedding_info(train_emb, train_lbl, split_name="train", class_names=class_names)
        print_embedding_info(val_emb, val_lbl, split_name="val", class_names=class_names)
        print_embedding_info(test_emb, test_lbl, split_name="test", class_names=class_names)

        emb_dir = os.path.join(log_dir, 'embeddings', run_name)
        os.makedirs(emb_dir, exist_ok=True)
        torch.save({'embeddings': train_emb, 'labels': train_lbl}, os.path.join(emb_dir, 'train.pt'))
        torch.save({'embeddings': val_emb, 'labels': val_lbl}, os.path.join(emb_dir, 'val.pt'))
        torch.save({'embeddings': test_emb, 'labels': test_lbl}, os.path.join(emb_dir, 'test.pt'))
        print(f"Saved embeddings to {emb_dir}")
    except Exception as e:
        print(f"Embedding extraction failed (supervised): {e}")

if __name__ == "__main__":
    from utils.dotdict import dotdict
    args = {
        "dataset": "DREAMER",
        "pos_embeddings_alpha": 0,
        "embedding_dim": 14,
        "n_head_token_enc": 2,
        "n_head_context_enc": 2,
        "depth_context_enc": 1,
        "lr": 1e-4,
        "finetune_epochs": 1,
        "es_after_epochs": 1,
        "train_val_split": "random",
        "preprocessing": "standardize"
        }

    main_supervised(dotdict(args))