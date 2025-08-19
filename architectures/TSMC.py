import torch
import torch.nn as nn
from architectures.transformer import TransformerEncoder
from .customLayers import TemporalSplit, OnetoManyGRU

class TSMC(nn.Module):
    def __init__(self, pos_embeddings_alpha: float, input_features: int, embedding_dim: int, n_head_token_enc: int, n_head_context_enc: int, depth_context_enc: int, max_predict_len: int) -> None:
        super(TSMC, self).__init__()
        self.max_predict_len = max_predict_len
        # self.token_encoder = TransformerEncoder(
        #     use_tokenizer=True,
        #     use_cls_token=False,
        #     use_pos_embedding=True,
        #     input_features=input_features,
        #     embedding_dim=embedding_dim,
        #     n_head=n_head_token_enc,
        #     depth=1
        # )
        self.context_encoder = TransformerEncoder(
            use_tokenizer=True,#False,
            use_cls_token=True,
            pos_embeddings_alpha=pos_embeddings_alpha,
            input_features=input_features,#embedding_dim,
            embedding_dim=embedding_dim,
            n_head=n_head_context_enc,
            depth=depth_context_enc
        )
        self.temporal_split = TemporalSplit(
            split_dim=2
        )
        self.prediction_head = OnetoManyGRU(
            embedding_dim,
            input_features,#embedding_dim,
            teacher_forcing = True,
            batch_first=True
        )

    def forward(self, x: torch.tensor, K: int):
        #tokens, _ = self.token_encoder(x)
        tokens = x
        signal, target = self.temporal_split(tokens, K)
        _, context = self.context_encoder(signal)
        prediction = self.prediction_head(context, K, x[:,:,-K:])

        return context, prediction, target

    @torch.no_grad()
    def encode(self, x: torch.tensor) -> torch.tensor:
        """
        Compute a fixed-length representation for each time series in the batch.

        Args:
            x: Input tensor of shape (batch_size, channels, timesteps)

        Returns:
            Tensor of shape (batch_size, embedding_dim) containing the CLS-token
            representations produced by the context encoder.
        """
        # Directly use the context encoder over the full input sequence
        _, context = self.context_encoder(x)
        return context