import torch
import torch.nn as nn
from typing import Optional
from efficientnet_pytorch import EfficientNet
import math

class Vnav(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        noise_pred_net: nn.Module,
        ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
    
    def forward(self, func_name: str, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_vec"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        else:
            raise NotImplementedError
        return output

class VisionEncoder(nn.Module):
    def __init__(
            self,
            context_size: int = 5,
            obs_encoder: Optional[str] = "efficientnet-b0",
            obs_encoding_size: Optional[int] = 256,
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
        self.goal_encoder = nn.Linear(2, self.goal_encoding_size)

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
            norm_first=False # XXX: modified here, not sure if it's correct
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
        context_enc = self.positional_encoding(context_enc) # [batch_size, context_size+2, obs_encoding_size]

        # Self-attention
        context_token = self.sa_encoder(context_enc)
        context_token = torch.mean(context_token, dim=1) # Average over the sequence dimension

        return context_token

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x