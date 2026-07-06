"""Plain-torch re-authoring of Streaming Sortformer v2's forward_for_export core, module-named to
match the NeMo checkpoint keys 1:1 (strict load). Runs in the coreai-models venv (torch 2.9).

Graph = pre_encode (dw_striding subsampling, 8x) -> xscale(sqrt(512)) -> [cat spkcache | chunk_pe]
      -> 17L non-causal rel-pos Conformer (batch_norm conv) -> 512->192 proj
      -> 18L post-LN Transformer (abs MHA + relu FF) -> head(relu, 192->192, 192->4, sigmoid).

AOSC / spkcache sort / streaming state live on the HOST (out of graph). This module is the pure
tensor path. Gated stage-by-stage against chunk_io.npz (captured from NeMo) in gate_reauthor.py.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- dims (from _nemo/model_config.yaml) ----
MEL, SUB_CH = 128, 256
HID, HEADS, HDIM, FF, NLAYERS = 512, 8, 64, 2048, 17   # conformer
CONV_K = 9
TF_HID, TF_HEADS, TF_HDIM, TF_INNER, TF_LAYERS = 192, 8, 24, 768, 18  # transformer
N_SPK = 4
XSCALE = math.sqrt(HID)
NEG_INF = -10000.0


# --------------------------------------------------------------------------- pre_encode
class PreEncode(nn.Module):
    """NeMo dw_striding ConvSubsampling: conv2d stem + 2 depthwise-separable stages (8x on time+freq)
    then a linear to HID. Sequential indices 0,2,3,5,6 match the ckpt (1,4,7 are ReLU)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, SUB_CH, 3, stride=2, padding=1),                       # 0
            nn.ReLU(),                                                          # 1
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),   # 2 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, 1),                                       # 3 pointwise
            nn.ReLU(),                                                          # 4
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),   # 5 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, 1),                                       # 6 pointwise
            nn.ReLU(),                                                          # 7
        )
        self.out = nn.Linear(SUB_CH * (MEL // 8), HID)

    def forward(self, x):  # x [B,Tf,MEL]
        x = x.unsqueeze(1)                                    # [B,1,Tf,MEL]
        x = self.conv(x)                                      # [B,SUB_CH,T,MEL//8]
        b, c, t, f = x.shape
        x = x.transpose(1, 2).reshape(b, t, c * f)            # [B,T,SUB_CH*MEL//8]
        return self.out(x)                                    # [B,T,HID]


# --------------------------------------------------------------------------- conformer
class ConformerFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(HID, FF)
        self.linear2 = nn.Linear(FF, HID)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class ConformerConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(HID, 2 * HID, 1)
        self.depthwise_conv = nn.Conv1d(HID, HID, CONV_K, padding=(CONV_K - 1) // 2, groups=HID)
        self.batch_norm = nn.BatchNorm1d(HID)
        self.pointwise_conv2 = nn.Conv1d(HID, HID, 1)

    def forward(self, x, conv_mask):  # x [B,T,HID], conv_mask [B,1,T] float (1=valid,0=pad)
        x = x.transpose(1, 2)                                 # [B,HID,T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if conv_mask is not None:
            x = x * conv_mask                                 # zero padded frames before depthwise
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class RelPosMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(HID, HID)
        self.linear_k = nn.Linear(HID, HID)
        self.linear_v = nn.Linear(HID, HID)
        self.linear_out = nn.Linear(HID, HID)
        self.linear_pos = nn.Linear(HID, HID, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.pos_bias_v = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.s_d_k = math.sqrt(HDIM)

    @staticmethod
    def rel_shift(x):  # x [B,H,T,2T-1]
        b, h, t, p = x.shape
        x = F.pad(x, (1, 0))
        x = x.view(b, h, p + 1, t)
        x = x[:, :, 1:].view(b, h, t, p)
        return x

    def forward(self, x, pos_emb, att_bias):  # x [B,T,HID], pos_emb [1,2T-1,HID], att_bias [B,1,T,T] additive
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, HEADS, HDIM)
        k = self.linear_k(x).view(B, T, HEADS, HDIM).transpose(1, 2)   # [B,H,T,dk]
        v = self.linear_v(x).view(B, T, HEADS, HDIM).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, HEADS, HDIM).transpose(1, 2)  # [1,H,2T-1,dk]
        q_u = (q + self.pos_bias_u).transpose(1, 2)                    # [B,H,T,dk]
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = q_u @ k.transpose(-2, -1)                          # [B,H,T,T]
        matrix_bd = q_v @ p.transpose(-2, -1)                          # [B,H,T,2T-1]
        matrix_bd = self.rel_shift(matrix_bd)[:, :, :, :T]
        scores = (matrix_ac + matrix_bd) / self.s_d_k
        if att_bias is not None:
            scores = scores + att_bias                                # 0 valid / NEG_INF at padded keys
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, HID)
        return self.linear_out(out)


class ConformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(HID)
        self.feed_forward1 = ConformerFF()
        self.norm_self_att = nn.LayerNorm(HID)
        self.self_attn = RelPosMHA()
        self.norm_conv = nn.LayerNorm(HID)
        self.conv = ConformerConv()
        self.norm_feed_forward2 = nn.LayerNorm(HID)
        self.feed_forward2 = ConformerFF()
        self.norm_out = nn.LayerNorm(HID)

    def forward(self, x, pos_emb, att_bias, conv_mask):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, att_bias)
        x = x + self.conv(self.norm_conv(x), conv_mask)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_encode = PreEncode()
        self.layers = nn.ModuleList([ConformerLayer() for _ in range(NLAYERS)])

    def forward(self, concat, pos_emb, att_bias, conv_mask):
        x = concat * XSCALE                      # NeMo applies xscale inside pos_enc, on the whole concat
        for blk in self.layers:
            x = blk(x, pos_emb, att_bias, conv_mask)
        return x


# --------------------------------------------------------------------------- transformer
class TFMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_net = nn.Linear(TF_HID, TF_HID)
        self.key_net = nn.Linear(TF_HID, TF_HID)
        self.value_net = nn.Linear(TF_HID, TF_HID)
        self.out_projection = nn.Linear(TF_HID, TF_HID)
        self.attn_scale = math.sqrt(math.sqrt(TF_HDIM))

    def _heads(self, x):  # [B,T,HID] -> [B,H,T,dk]
        B, T, _ = x.shape
        return x.view(B, T, TF_HEADS, TF_HDIM).permute(0, 2, 1, 3)

    def forward(self, x, att_bias):  # att_bias [B,1,1,T] additive (0 valid / -1e4 pad key)
        q = self._heads(self.query_net(x)) / self.attn_scale
        k = self._heads(self.key_net(x)) / self.attn_scale
        v = self._heads(self.value_net(x))
        scores = q @ k.transpose(-2, -1)
        if att_bias is not None:
            scores = scores + att_bias
        probs = torch.softmax(scores, dim=-1)
        ctx = (probs @ v).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], TF_HID)
        return self.out_projection(ctx)


class TFPositionWiseFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_in = nn.Linear(TF_HID, TF_INNER)
        self.dense_out = nn.Linear(TF_INNER, TF_HID)

    def forward(self, x):
        return self.dense_out(F.relu(self.dense_in(x)))


class TFEncoderBlock(nn.Module):
    """Post-LN: Attn -> +res -> LN1 -> FF -> +res -> LN2."""

    def __init__(self):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(TF_HID, eps=1e-5)
        self.first_sub_layer = TFMultiHeadAttention()
        self.layer_norm_2 = nn.LayerNorm(TF_HID, eps=1e-5)
        self.second_sub_layer = TFPositionWiseFF()

    def forward(self, x, att_bias):
        a = self.first_sub_layer(x, att_bias) + x
        a = self.layer_norm_1(a)
        o = self.second_sub_layer(a) + a
        return self.layer_norm_2(o)


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TFEncoderBlock() for _ in range(TF_LAYERS)])

    def forward(self, x, att_bias):
        for blk in self.layers:
            x = blk(x, att_bias)
        return x


# --------------------------------------------------------------------------- head + top
class SortformerModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_proj = nn.Linear(HID, TF_HID)
        self.first_hidden_to_hidden = nn.Linear(TF_HID, TF_HID)
        self.single_hidden_to_spks = nn.Linear(TF_HID, N_SPK)
        self.hidden_to_spks = nn.Linear(2 * TF_HID, N_SPK)  # unused at inference (loaded for strictness)

    def sigmoids(self, h):  # forward_speaker_sigmoids (dropout is identity at eval)
        h = F.relu(h)
        h = self.first_hidden_to_hidden(h)
        h = F.relu(h)
        return torch.sigmoid(self.single_hidden_to_spks(h))


def build_pos_emb(T: int, dtype=torch.float32) -> torch.Tensor:
    """RelPositionalEncoding table for length T: positions (T-1)..-(T-1), even=sin/odd=cos, base 1e4."""
    div_term = torch.exp(torch.arange(0, HID, 2, dtype=torch.float32) * -(math.log(10000.0) / HID))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float32).unsqueeze(1)   # [2T-1,1]
    pe = torch.zeros(2 * T - 1, HID)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0).to(dtype)                                      # [1,2T-1,HID]


class Sortformer(nn.Module):
    """Full forward_for_export core. Masks/pos_emb are supplied by the caller (host builds them per
    chunk from lengths) so the exported graph stays static."""

    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder()
        self.transformer_encoder = TransformerEncoder()
        self.sortformer_modules = SortformerModules()

    # ---- individual stages (for gating) ----
    def pre_encode(self, mel_chunk):
        return self.encoder.pre_encode(mel_chunk)

    def conformer_proj(self, concat, pos_emb, att_bias, conv_mask):
        x = self.encoder(concat, pos_emb, att_bias, conv_mask)
        return self.sortformer_modules.encoder_proj(x)

    def infer(self, fc, tf_att_bias, out_mask):
        x = self.transformer_encoder(fc, tf_att_bias)
        preds = self.sortformer_modules.sigmoids(x)
        return preds * out_mask

    @staticmethod
    def masks_from_valid(valid):
        """valid [B,T] float {0,1} -> (conf_att_bias [B,1,T,T], conv_mask [B,1,T],
        tf_att_bias [B,1,1,T], out_mask [B,T,1]). All additive/multiplicative, export-friendly."""
        both = valid.unsqueeze(2) * valid.unsqueeze(1)           # [B,T,T] 1 where q&k both valid
        conf_att_bias = ((1.0 - both) * NEG_INF).unsqueeze(1)    # [B,1,T,T] 0 valid / NEG_INF pad
        conv_mask = valid.unsqueeze(1)                           # [B,1,T]
        tf_att_bias = ((1.0 - valid) * NEG_INF).view(valid.shape[0], 1, 1, -1)   # [B,1,1,T]
        out_mask = valid.unsqueeze(-1)                           # [B,T,1]
        return conf_att_bias, conv_mask, tf_att_bias, out_mask

    # ---- full static graph (pre_encode + simple cat + conformer + proj + transformer + head) ----
    def forward(self, chunk_mel, spkcache, valid, pos_emb):
        conf_att_bias, conv_mask, tf_att_bias, out_mask = self.masks_from_valid(valid)
        chunk_pe = self.encoder.pre_encode(chunk_mel)                        # [1,Pe,512] (pre-xscale)
        concat = torch.cat([spkcache, chunk_pe], dim=1)                     # [1, 188+Pe, 512]
        fc = self.conformer_proj(concat, pos_emb, conf_att_bias, conv_mask)  # applies xscale inside
        preds = self.infer(fc, tf_att_bias, out_mask)                       # [1,378,4]
        return preds, chunk_pe   # host needs chunk_pe to update the speaker cache (as forward_for_export)


def load_ckpt(model: Sortformer, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    want = {k: v for k, v in sd.items() if not k.startswith("preprocessor.")}
    missing, unexpected = model.load_state_dict(want, strict=False)
    miss = [m for m in missing if not m.endswith("num_batches_tracked")]
    assert not miss and not unexpected, f"weight map mismatch: missing={miss[:8]} unexpected={unexpected[:8]}"
    return model
