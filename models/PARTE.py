import torch
import torch.nn as nn
from utils.augmentations import SeasonDecompose_Mask
from layers.Transformer_EncDec import CrossAttnLayer, Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.losses import ContrastiveLoss
from tda_module.tda_layers import TopologicalPotentialBank


class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x
    
class TemporalProjection_Head(nn.Module):
    def __init__(self, seq_len, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        x = self.linear(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.dropout(x)
        return x.squeeze()


class MLPBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TopoEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TopoEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, cross, x_mask=x_mask, cross_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, cross, x_mask=x_mask, cross_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

    
class PATREEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.trend_trans = nn.Sequential(*[MLPBlock(configs.d_model, configs.d_ff, configs.dropout) for _ in range(configs.e_layers)])
        self.season_trans = nn.Sequential(*[MLPBlock(configs.d_model, configs.d_hidden, configs.dropout) for _ in range(configs.e_layers)])

        self.trend_encoder = TopoEncoder(
            [
                CrossAttnLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.season_encoder = TopoEncoder(
            [
                CrossAttnLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_hidden,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.trend_mul = nn.Parameter(torch.zeros(configs.d_model))
        self.season_mul = nn.Parameter(torch.zeros(configs.d_model))
        
        self.component_gating_layer = nn.Sequential(
                                        nn.Linear(2*configs.d_model, 2), 
                                        nn.Softmax(dim=-1)
                                        ) 
        
    def _combine_trend_season(self, zt, zs):
        gate_input = torch.cat([zt, zs], dim=-1) # b, l, 2*d_model
        weights = self.component_gating_layer(gate_input) # b, l, 2 
        a = weights[:, :, 0].unsqueeze(-1)
        b = weights[:, :, 1].unsqueeze(-1)
        z = a * zt + b * zs
        return z

    def _pretrain_stage(self, masked_xt, masked_xs_list, tda_features, tda_phi):
        enc_zt = self.trend_encoder(masked_xt, tda_phi)[0]
        enc_zs_list = [self.season_encoder(masked_xs, tda_features)[0] for masked_xs in masked_xs_list]
        xt = self.trend_trans(masked_xt)
        xs_list = [self.season_trans(masked_xs) for masked_xs in masked_xs_list]
        zt = xt + self.trend_mul * enc_zt
        zs_list = [xs + self.season_mul * enc_zs for xs, enc_zs in zip(xs_list, enc_zs_list)]
        z_list = [self._combine_trend_season(zt, zs) for zs in zs_list]
        return z_list

    def _finetune(self, xt, xs, tda_features, tda_phi):      
        enc_zt = self.trend_encoder(xt, tda_phi)[0] # b, l, d_model
        enc_zs = self.season_encoder(xs, tda_features)[0] # b, l, d_model
        xt = self.trend_trans(xt)
        xs = self.season_trans(xs)
        zt = xt + self.trend_mul * enc_zt
        zs = xs + self.season_mul * enc_zs
        z = self._combine_trend_season(zt, zs) # b, l, d_model
        return z
                
    def forward(self, xt, xs, tda_features, tda_phi, task_name="finetune"):
        
        if task_name == "pretrain":
            return self._pretrain_stage(xt, xs, tda_features, tda_phi)
        elif task_name == "finetune":
            return self._finetune(xt, xs, tda_features, tda_phi)
        else:
            raise ValueError(f"Unknown task_name: {task_name}")

        
class Model(nn.Module):
    """
    PATRE
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.seg_len = configs.seg_len
        assert self.seq_len % self.seg_len == 0, f"seq_len ({self.seq_len}) must be divisible by seg_len ({self.seg_len})"
        self.n_patches = self.seq_len // self.seg_len
        self.output_attention = configs.output_attention
        self.n_kernels = configs.n_kernels
        self.configs = configs
        self.use_tda = configs.use_tda
        self.topo_loss = ContrastiveLoss(configs.tau)

        # Embedding
        self.trend_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.season_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.tda_layers= TopologicalPotentialBank(n_kernels = configs.n_kernels, pooling = configs.tda_pooling)

        if self.task_name == 'pretrain':
            self.patch_weights = nn.Parameter(torch.zeros(self.n_patches))
            self.x_embedding = nn.Linear(1, configs.d_model)
            self.tda_embedding = nn.Linear(configs.n_kernels, configs.d_model)
        elif self.task_name == 'finetune':
            self.tda_embedding = nn.Linear(configs.n_kernels, configs.d_model)

        # mask and ts decomposition
        self.SeasonDecompose_Mask = SeasonDecompose_Mask(configs.seq_len, configs.kernel_size, configs.seg_len, configs.p_tmask, configs.topk)
        # Encoder
        self.encoder = PATREEncoder(configs)

        self.pred_head = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)

    def _combine_patches(self, z_list):

        # (K,)
        weights = torch.softmax(self.patch_weights, dim=0)
        z = sum(w * zp for w, zp in zip(weights, z_list))
    
        return z

    def _size_check(self, x_enc, pd_tensor, pd_mask):
        if pd_tensor.dim() == 5:
            pd_tensor = pd_tensor.unsqueeze(1)
        if pd_mask.dim() == 4:
            pd_mask = pd_mask.unsqueeze(1)
        
        assert x_enc.ndim == 3, f"x_enc must be 3D (B, L, D), got {x_enc.shape}"
        assert pd_tensor.ndim == 6, f"pd_tensor must be 6D (B, ?, ?, ?, ?, ?), got {pd_tensor.shape}"
        assert pd_mask.ndim == 5, f"pd_mask must be 5D (B, ?, ?, ?, ?), got {pd_mask.shape}"
        
        return pd_tensor, pd_mask

    def _pretrain(self, x_enc, x_mark_enc, pd_tensor, pd_mask):
        
        pd_tensor, pd_mask = self._size_check(x_enc, pd_tensor, pd_mask)
        bs, seq_len, n_vars = x_enc.shape
        _, n_patches, _,  _, max_points, _ = pd_tensor.shape # (B, n_patches, n_vars, 2, max_points, 2)

        assert seq_len % n_patches == 0, f"seq_len ({seq_len}) must be divisible by n_patches ({n_patches})"

        # mean subtraction
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
    
        # channel independent
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(-1, seq_len, 1)
        xt_enc, xs_enc = self.SeasonDecompose_Mask.extract_season_trend(x_enc)
        masked_xt, masked_xs_list = self.SeasonDecompose_Mask.mask(xt_enc, xs_enc)
        xt_enc = self.trend_embedding(masked_xt)
        xs_enc_list = [self.season_embedding(masked_xs) for masked_xs in masked_xs_list]
        
        pd_tensor = pd_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()
        pd_mask = pd_mask.permute(0, 2, 1, 3, 4).contiguous()
        pd_tensor = pd_tensor.view(-1, max_points, 2)
        pd_mask = pd_mask.view(-1, max_points)
        
        tda_repre = self.tda_layers(pd_tensor, pd_mask)
        features = tda_repre["features"]
        phi = tda_repre["responses"]
        
        features_enc = self.tda_embedding(features.contiguous().view(-1, n_patches * 2, self.n_kernels))
        phi_enc = self.tda_embedding(phi.reshape(-1, n_patches * 2 * max_points, self.n_kernels))

        z_list = self.encoder(xt_enc, xs_enc_list, features_enc, phi_enc, self.task_name)
        z = self.x_embedding(x_enc)
        topo_loss = self.topo_loss(z, z_list)
        
        dec_out = self._combine_patches(z_list)
        dec_out = dec_out.reshape(bs, n_vars, seq_len, -1)
        dec_out = self.pred_head(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out + means

        return dec_out, topo_loss
        
        
    def _forecast(self, x_enc, x_mark_enc, pd_tensor, pd_mask):

        pd_tensor, pd_mask = self._size_check(x_enc, pd_tensor, pd_mask)
    
        bs, seq_len, n_vars = x_enc.shape
        _, n_patches, _,  _, max_points, _ = pd_tensor.shape # (B, n_patches, n_vars, 2, max_points, 2)
    
        # mean subtraction
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
    
        # channel independent
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(-1, seq_len, 1)

        pd_tensor = pd_tensor.permute(0, 2, 1, 3, 4, 5).contiguous()
        pd_mask = pd_mask.permute(0, 2, 1, 3, 4).contiguous()
        pd_tensor = pd_tensor.view(-1, max_points, 2)
        pd_mask = pd_mask.view(-1, max_points)
    
        # decomposition
        xt_enc, xs_enc = self.SeasonDecompose_Mask.extract_season_trend(x_enc)
    
        # TDA
        tda_repre = self.tda_layers(pd_tensor, pd_mask)
        features = tda_repre["features"]
        phi = tda_repre["responses"]
    
        features_enc = self.tda_embedding(features.contiguous().view(-1, n_patches * 2, self.n_kernels))
        phi_enc = self.tda_embedding(phi.reshape(-1, n_patches * 2 * max_points, self.n_kernels))
    
        # embedding
        xt_enc = self.trend_embedding(xt_enc)
        xs_enc = self.season_embedding(xs_enc)
    
        # encoder
        enc_out = self.encoder(xt_enc, xs_enc, features_enc, phi_enc)
        enc_out = enc_out.reshape(bs, n_vars, seq_len, -1)
    
        # decoder
        dec_out = self.pred_head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
    
        # mean restoration
        dec_out = dec_out + means
    
        return dec_out

    def forward(self, x_enc, x_mark_enc, pd_tensor, pd_mark):
    
        if self.task_name == 'pretrain':
            return self._pretrain(x_enc, x_mark_enc, pd_tensor, pd_mark)
        elif self.task_name == 'finetune':
            dec_out = self._forecast(x_enc, x_mark_enc, pd_tensor, pd_mark)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise NotImplementedError(f"Unknown task_name: {self.task_name}")
