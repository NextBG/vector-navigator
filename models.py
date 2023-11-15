import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from prettytable import PrettyTable

from vint_train.models.vint.self_attention import PositionalEncoding

class Vnav(nn.Module):
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    
class VisionEncoder(nn.Module):
    def __init__(
            self,
            context_size: int = 5,
            obs_encoder: Optional[str] = "efficientnet-b0",
            obs_encoding_size: Optional[int] = 512,
            mha_num_attention_heads: Optional[int] = 2,
            mha_num_attention_layers: Optional[int] = 2,
            mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        Vision Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # Initialize the observation encoder: EfficientNet
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError
        
        # Initialize the goal encoder: float[3] to float[obs_encoding_size]
        self.goal_encoder = nn.Linear(3, self.goal_encoding_size)

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.sa_encoder= nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

    def forward(self, obs_img: torch.tensor, goal_vec: torch.tensor) -> torch.tensor:
        # Vector goal encoding [batch_size, 3] -> [batch_size, goal_encoding_size]
        goal_enc = self.goal_encoder(goal_vec)
        goal_enc = goal_enc.unsqueeze(1) # Add a sequence dimension

        # Observation encoding [batch_size, 3*(context_size+1), width, height] -> [batch_size, context_size+1, obs_encoding_size] TODO: understand this part
        obs_img = torch.split(obs_img, 3, dim=1) # Split into 3 channels
        obs_img = torch.concat(obs_img, dim=0) # Concatenate along the batch dimension
        
        obs_enc = self.obs_encoder.extract_features(obs_img)
        obs_enc = self.obs_encoder._avg_pooling(obs_enc)
        if self.obs_encoder._global_params.include_top:
            obs_enc = obs_enc.flatten(start_dim=1)
            obs_enc = self.obs_encoder._dropout(obs_enc)
        obs_enc = self.compress_obs_enc(obs_enc)

        obs_enc = obs_enc.unsqueeze(0) # Add a batch_size dimension
        obs_enc = obs_enc.reshape((-1, self.context_size+1, self.obs_encoding_size)) # [batch_size, context_size+1, obs_encoding_size]

        # Concatenate observations and goal to form context encoding
        context_enc = torch.cat([obs_enc, goal_enc], dim=1)

        # Apply positional encoding to observation encoding
        context_enc = self.positional_encoding(context_enc)

        # Self-attention
        context_token = self.sa_encoder(context_enc)
        context_token = torch.mean(context_token, dim=1) # Average over the sequence dimension

        return context_token

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

# Test the model with random inputs
if __name__ == "__main__":
    # Initialize the model
    model = VisionEncoder()
    print(model)

    WIDTH = 120
    HEIGHT = 160
    CONTEXT_SIZE = 5
    BS = 2

    # Test the model
    obs_img = torch.rand((BS, 3*(CONTEXT_SIZE+1), WIDTH, HEIGHT))
    goal_vec = torch.rand((BS, 3))
    output = model(obs_img, goal_vec)
    print(output.shape)