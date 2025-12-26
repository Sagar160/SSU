import torch
import torch.nn as nn
import fvdb.nn as fvnn

class CNN_vanilla(nn.Module):
    def __init__(self, in_channels=3, features=32, out_channels=1, dropout=0.05):
        super(CNN_vanilla, self).__init__()
        
        self.activation = fvnn.SiLU(inplace=True)
        self.encoder = nn.Sequential(
            fvnn.SparseConv3d(in_channels, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation
        )
        
        self.decoder = nn.Sequential(
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, out_channels, kernel_size=1, stride=1)
        )

        self.t_conv = fvnn.SparseConv3d(
            features, features, kernel_size=3, stride=2, transposed=True) #TODO check that this is correct

    def forward(self, x, out_grid):
        enc = self.encoder(x)
        x = self.t_conv(enc, out_grid=out_grid) 
        return self.decoder(x)
    
import torch
import torch.nn as nn
import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor
import torch.nn.functional as F
import math

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        condition_channels,
        num_heads=8,
        num_head_channels=8,
    ):
        super().__init__()
        self.channels = channels
        self.condition_channels = condition_channels
        
        assert channels % num_head_channels == 0, f"Q channels {channels} not divisible by num_head_channels {num_head_channels}"
        self.num_heads = channels // num_head_channels
        self.head_dim = num_head_channels
        
        self.norm_query = fvnn.GroupNorm(min(4, channels), channels)
        self.norm_kv = fvnn.GroupNorm(1, condition_channels)
        
        self.q_proj = fvnn.Linear(channels, self.num_heads * self.head_dim)
        
        # Split KV projections to avoid chunking issues
        self.k_proj = fvnn.Linear(condition_channels, self.num_heads * self.head_dim)
        self.v_proj = fvnn.Linear(condition_channels, self.num_heads * self.head_dim)
        
        self.proj_out = fvnn.Linear(self.num_heads * self.head_dim, channels)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        length_q, ch_q = q.shape
        length_kv, ch_kv = k.shape

        assert ch_q == self.num_heads * self.head_dim, "Query dim mismatch"
        assert ch_kv == self.num_heads * self.head_dim, "Key/Value dim mismatch"

        q = q.reshape(length_q, self.num_heads, self.head_dim)
        k = k.reshape(length_kv, self.num_heads, self.head_dim)
        v = v.reshape(length_kv, self.num_heads, self.head_dim)
        
        scores = torch.einsum("qhd,khd->qhk", q, k) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        values = torch.einsum("qhk,khd->qhd", attention_weights, v)

        values = values.reshape(length_q, -1)
        return values
        
    def attention(self, q: VDBTensor, k: VDBTensor, v: VDBTensor):
        values = []
        for batch_idx in range(q.grid.grid_count):
            q_batch = q.data[batch_idx].jdata
            k_batch = k.data[batch_idx].jdata
            v_batch = v.data[batch_idx].jdata
            values.append(self._attention(q_batch, k_batch, v_batch))
            
        return fvdb.JaggedTensor(values)

    def forward(self, x: VDBTensor, cond: VDBTensor):
        # 1. Project main input for queries
        q = self.q_proj(self.norm_query(x))
        
        # 2. Project conditional input for keys and values separately
        k = self.k_proj(self.norm_kv(cond))
        v = self.v_proj(self.norm_kv(cond))
        
        # 3. Perform cross-attention
        attended_features = self.attention(q, k, v)
        
        # 4. Project output and add residual
        attended_features_vdb = VDBTensor(x.grid, attended_features)
        feature = self.proj_out(attended_features_vdb)
        return feature + x
    
# --------------------------------------------------------------------------------------------------------------------------

class CNN_vanilla_with_attention(nn.Module):
    def __init__(self, in_channels=2, features=32, out_channels=1, dropout=0.05):
        super(CNN_vanilla_with_attention, self).__init__()

        self.activation = fvnn.SiLU(inplace=True)
        
        self.attention_block = CrossAttentionBlock(channels=features, condition_channels=1) 
        
        self.encoder = nn.Sequential(
            fvnn.SparseConv3d(in_channels, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation
        )
        
        self.decoder = nn.Sequential(
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, out_channels, kernel_size=3, stride=1)
        )

        self.t_conv = fvnn.SparseConv3d(features, features, kernel_size=3, stride=2, transposed=True)

    def forward(self, x: VDBTensor, cond: VDBTensor):
        enc = self.encoder(x)
        attended_enc = self.attention_block(enc, cond)
        return self.decoder(attended_enc)

class CNN_vanilla_without_transpose(nn.Module):
    def __init__(self, in_channels=3, features=32, out_channels=1, dropout=0.05):
        super(CNN_vanilla_without_transpose, self).__init__()

        self.activation = fvnn.SiLU(inplace=True)
        self.encoder = nn.Sequential(
            fvnn.SparseConv3d(in_channels, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation
        )
        
        self.decoder = nn.Sequential(
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, out_channels, kernel_size=3, stride=1)
        )

        self.t_conv = fvnn.SparseConv3d(
            features, features, kernel_size=3, stride=2, transposed=True) #TODO check that this is correct

    def forward(self, x):
        enc = self.encoder(x)
        # x = self.t_conv(enc, out_grid=out_grid)
        # input_sdf = fvnn.VDBTensor(x.grid, x.grid.jagged_like(x.jdata[:, 0].unsqueeze(-1)))
        return self.decoder(enc)


class CNN_vanilla_with_residual(nn.Module):
    def __init__(self, in_channels=3, features=32, out_channels=1, dropout=0.05):
        super(CNN_vanilla_with_residual, self).__init__()

        self.activation = fvnn.SiLU(inplace=True)
        self.encoder = nn.Sequential(
            fvnn.SparseConv3d(in_channels, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation
        )
        
        self.decoder = nn.Sequential(
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, out_channels, kernel_size=1, stride=1)
        )

        self.t_conv = fvnn.SparseConv3d(
            features, features, kernel_size=3, stride=2, transposed=True) #TODO check that this is correct

    def forward(self, input, out_grid):
        new_centers = out_grid.grid_to_world(out_grid.ijk.float())
        up_feat = input.grid.sample_trilinear(new_centers, input.jdata[:,0].unsqueeze(-1))
        x_up = fvnn.VDBTensor(out_grid, up_feat)

        x = self.encoder(input)
        x = self.t_conv(x, out_grid=out_grid) 
        x = self.decoder(x)
        return  x + x_up


class CNN_vanilla2(nn.Module):
    def __init__(self, in_channels=3, features=32, out_channels=1, dropout=0.05):
        super(CNN_vanilla2, self).__init__()
        
        self.activation = fvnn.SiLU(inplace=True)
        self.encoder = nn.Sequential(
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            self.activation
        )
        
        self.decoder = nn.Sequential(
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            self.activation,
            fvnn.SparseConv3d(features, out_channels, kernel_size=1, stride=1)
        )

        self.t_conv = fvnn.SparseConv3d(
            in_channels, features, kernel_size=3, stride=1) #TODO check that this is correct

    def forward(self, x, out_grid):
        x = self.t_conv(x, out_grid=out_grid)
        enc = self.encoder(x)
        return self.decoder(enc)
    

class CNN_FM(nn.Module):
    def __init__(self, in_channels=11, out_channels=1, features=64, dropout=0.01):
        super(CNN_FM, self).__init__()
        self.activation = fvnn.SiLU(inplace=True)
        self.encoder = nn.Sequential(
            fvnn.SparseConv3d(in_channels, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            fvnn.BatchNorm(features),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            fvnn.BatchNorm(features),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            fvnn.BatchNorm(features),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            fvnn.BatchNorm(features),
            self.activation,

            fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
            fvnn.Dropout(dropout),
            fvnn.BatchNorm(features),
            self.activation
        )
        # grid matcher
        self.grid_matcher = fvnn.SparseConv3d(features, out_channels, kernel_size=3, stride=1)

    def forward(self, input):
        # first feature of input)
        x = self.encoder(input)
        x = self.grid_matcher(x)
        return x 
    

class CNN_VAE(nn.Module):
    def __init__(self, in_channels=3, features=32, latent_dim=256,
                 out_channels=1, dropout=0.05):
        super().__init__()
        act = fvnn.SiLU(inplace=True)

        # Encoder
        self.en1 = fvnn.SparseConv3d(in_channels, features, 7, stride=1)
        self.en1_bn = fvnn.BatchNorm(features)
        self.en1_act = act

        self.en2 = fvnn.SparseConv3d(features, features, 7, stride=2)
        self.en2_bn = fvnn.BatchNorm(features)
        self.en2_act = act

        self.en3 = fvnn.SparseConv3d(features, features, 5, stride=2)
        self.en3_bn = fvnn.BatchNorm(features)
        self.en3_act = act

        self.en4 = fvnn.SparseConv3d(features, features, 5, stride=2)
        self.en4_bn = fvnn.BatchNorm(features)
        self.en4_act = act

        self.en5 = fvnn.SparseConv3d(features, features, 5, stride=2)
        self.en5_bn = fvnn.BatchNorm(features)
        self.en5_act = act

        self.en6 = fvnn.SparseConv3d(features, features, 3, stride=2)
        self.en6_bn = fvnn.BatchNorm(features)
        self.en6_act = act

        # Latent layers
        self.mu = fvnn.SparseConv3d(features, latent_dim, kernel_size=1, stride=1)
        self.logvar = fvnn.SparseConv3d(features, latent_dim, kernel_size=1, stride=1)

        # Decoder (transpose convolutions)
        self.dec1 = fvnn.SparseConv3d(latent_dim, features, 3, stride=2, transposed=True)
        self.dec1_bn = fvnn.BatchNorm(features)
        self.dec1_act = act

        self.dec2 = fvnn.SparseConv3d(features, features, 3, stride=2, transposed=True)
        self.dec2_bn = fvnn.BatchNorm(features)
        self.dec2_act = act

        self.dec3 = fvnn.SparseConv3d(features, features, 3, stride=2, transposed=True)
        self.dec3_bn = fvnn.BatchNorm(features)
        self.dec3_act = act

        self.dec4 = fvnn.SparseConv3d(features, features, 3, stride=2, transposed=True)
        self.dec4_bn = fvnn.BatchNorm(features)
        self.dec4_act = act

        self.dec5 = fvnn.SparseConv3d(features, features, 3, stride=2, transposed=True)
        self.dec5_bn = fvnn.BatchNorm(features)
        self.dec5_act = act

        self.dec6 = fvnn.SparseConv3d(features, out_channels, 3, stride=1, transposed=False)

        # Dropout
        self.drop = fvnn.Dropout(dropout)

    def reparameterize(self, mu_vdb, logvar_vdb):
        mu = mu_vdb.jdata
        logvar = logvar_vdb.jdata
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled = mu + eps * std
        latent_vdb = fvnn.VDBTensor(mu_vdb.grid, mu_vdb.grid.jagged_like(sampled))
        return latent_vdb, mu, logvar

    def forward(self, x: fvnn.VDBTensor):
        # Encoder
        x1 = self.en1_act(self.en1_bn(self.en1(x)))
        x1 = self.drop(x1)
        x2 = self.en2_act(self.en2_bn(self.en2(x1)))
        x2 = self.drop(x2)
        x3 = self.en3_act(self.en3_bn(self.en3(x2)))
        x3 = self.drop(x3)
        x4 = self.en4_act(self.en4_bn(self.en4(x3)))
        x4 = self.drop(x4)
        x5 = self.en5_act(self.en5_bn(self.en5(x4)))
        x5 = self.drop(x5)
        x6 = self.en6_act(self.en6_bn(self.en6(x5)))
        x6 = self.drop(x6)

        # Latent
        mu_vdb = self.mu(x6)
        logvar_vdb = self.logvar(x6)
        z, mu, logvar = self.reparameterize(mu_vdb, logvar_vdb)

        # Decode, using skip-grid from encoder
        d = self.dec1_act(self.dec1_bn(self.dec1(z, out_grid=x5.grid)))
        d = self.drop(d)
        d = self.dec2_act(self.dec2_bn(self.dec2(d, out_grid=x4.grid)))
        d = self.drop(d)
        d = self.dec3_act(self.dec3_bn(self.dec3(d, out_grid=x3.grid)))
        d = self.drop(d)
        d = self.dec4_act(self.dec4_bn(self.dec4(d, out_grid=x2.grid)))
        d = self.drop(d)
        d = self.dec5_act(self.dec5_bn(self.dec5(d, out_grid=x1.grid)))
        d = self.drop(d)
        out = self.dec6(d, out_grid=x.grid)

        return out, mu, logvar