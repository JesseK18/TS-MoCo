import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TemporalSplit(nn.Identity):
    def __init__(self, split_dim=1):
        super(TemporalSplit, self).__init__()
        self.layer = nn.Identity()
        self.split_dim = split_dim

    def forward(self, input, K):
        if K == 0:
            if self.split_dim == 0: return self.layer(input[:,:,:]), self.layer(input[-1:,:,:])
            elif self.split_dim == 1: return self.layer(input[:,:,:]), self.layer(input[:,-1:,:])
            elif self.split_dim == 2: return self.layer(input[:,:,:]), self.layer(input[:,:,-1:])
            else: raise ValueError(f"split_dim must be one of [0,1,2], but got {self.split_dim}")
        else:
            if self.split_dim == 0: return self.layer(input[:-K,:,:]), self.layer(input[-K:,:,:])
            elif self.split_dim == 1: return self.layer(input[:,:-K,:]), self.layer(input[:,-K:,:])
            elif self.split_dim == 2: return self.layer(input[:,:,:-K]), self.layer(input[:,:,-K:])
            else: raise ValueError(f"split_dim must be one of [0,1,2], but got {self.split_dim}")
            

        

class OnetoManyGRU(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, teacher_forcing: bool = True, batch_first: bool = True):
        super(OnetoManyGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.teacher_forcing = teacher_forcing
        self.batch_first = batch_first
        self.prediction_head = nn.GRU(embedding_dim, embedding_dim, batch_first=batch_first)
        # Map from raw output_dim (input features C) to embedding_dim for teacher forcing inputs
        if embedding_dim == output_dim:
            self.input_tokenizer = nn.Identity()
        else:
            self.input_tokenizer = nn.Linear(output_dim, embedding_dim)
        # Always provide an untokenizer to map embeddings back to input feature space
        if embedding_dim == output_dim:
            self.untokenizer = nn.Identity()
        else:
            self.untokenizer = nn.Linear(embedding_dim, output_dim)

    def forward(self, c: torch.Tensor, K: int, x: torch.Tensor = None) -> torch.Tensor:
        if K == 0: return torch.zeros((x.size(0), x.size(1), 0))
        if self.batch_first:
            batch_size = c.size(0)    
            x_k = torch.zeros(batch_size, 1, self.embedding_dim, device=c.device)
        else:
            batch_size = c.size(1)
            x_k = torch.zeros(1, batch_size, self.embedding_dim, device=c.device)
        h_k = c.unsqueeze(0).contiguous()

        y_out = []
        for k in range(K):
            y_k, h_k = self.prediction_head(x_k, h_k)
            y_out.append(y_k)
            if self.teacher_forcing:
                # Take the next ground-truth target slice (B, C, 1) -> (B, 1, C)
                x_slice = x[:,:,-K+k:-K+k+1].transpose(1,2)
                # Project to embedding dimension expected by the GRU
                x_k = self.input_tokenizer(x_slice)
            else:
                x_k = y_k

        # y shape: (B, K, embedding_dim)
        y = torch.cat(y_out, dim=1)
        # Map back to input space (B, K, output_dim)
        y = self.untokenizer(y)
        # Return as (B, output_dim, K) to match target x_T from TemporalSplit
        y = y.transpose(1, 2)
        return y