"""Tree-attention VERIFY kernel (the structural, LOSSLESS spec-decode lever) — correctness de-risk.

EAGLE-3-style spec-decode drafts a TREE of candidate tokens; verifying them in ONE forward needs each
tree node to attend to the prefix + its ANCESTORS only (not siblings/other branches). That = a
multi-query (q=T) flash attention with an ARBITRARY additive mask (tree-ancestor mask), a generalization
of the q=1 flash-decode SDPA (gemma4_dense_metal_sdpa). The batched forward amortizes the weight read
over T tree tokens (the speed) and verify keeps the output distribution EXACT (lossless — no quality cost,
unlike 4-bit quant). This file de-risks the KERNEL only (GPU-parity vs a torch masked-attention reference
at small dims; no model, no EAGLE head).

Run: ../../../coreai-models/.venv/bin/python -u tree_attn_verify.py
"""
import asyncio, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter
from coreai.runtime import NDArray

_MAX_EPT = 8  # head_dim/32 upper bound (D=256 -> 8); per-lane register slice

# One SIMD-group (32 lanes) per (head h=gid.y, query t=gid.z). lane=gid.x owns ept=D/32 head dims.
# DSL axes are reversed vs torch: Q torch [H,T,D] -> A[d,t,h]=torch Q[h,t,d]; K/V torch [nkv,Sk,D] ->
# K[d,j,kv]=torch K[kv,j,d]; MASK torch [T,Sk] -> M[j,t]=torch MASK[t,j]; CTX torch [H,T,D] -> CTX[d,t,h].
# GQA: query head h -> kv slot h/(H/nkv). scale baked into qx. fp32 online softmax. mask is ADDITIVE
# (0 = attend, -inf = block) — the tree-ancestor mask (or any custom mask).
_TREE_SRC = r"""
    const uint D    = A.get_extent(0);     // head_dim
    const uint T    = A.get_extent(1);     // queries (tree size)
    const uint H    = A.get_extent(2);     // heads
    const uint Sk   = K.get_extent(1);     // keys (prefix + tree)
    const uint nkv  = K.get_extent(2);
    const uint lane = gid.x;               // 0..31
    const uint h    = gid.y;               // head
    const uint t    = gid.z;               // query / tree node
    const uint ept  = D / 32;              // <= __MAXEPT__
    const uint kv   = h / (H / nkv);
    const float scale = __SCALE__;

    float qx[__MAXEPT__];
    for (uint i = 0; i < ept; ++i) qx[i] = float(A[lane * ept + i, t, h]) * scale;

    float o[__MAXEPT__];
    for (uint i = 0; i < ept; ++i) o[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (uint j = 0; j < Sk; ++j) {
        float p = 0.0f;
        for (uint i = 0; i < ept; ++i) p += qx[i] * float(K[lane * ept + i, j, kv]);
        float s = simd_sum(p) + float(M[j, t]);   // masked score (M additive: 0 or -inf)
        float mnew = max(m, s);
        float corr = exp(m - mnew);
        float e    = exp(s - mnew);
        l = l * corr + e;
        for (uint i = 0; i < ept; ++i) o[i] = o[i] * corr + e * float(V[lane * ept + i, j, kv]);
        m = mnew;
    }
    float inv = 1.0f / l;
    for (uint i = 0; i < ept; ++i) CTX[lane * ept + i, t, h] = TYPE(o[i] * inv);
"""


def _tree_defn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # q [H,T,D]; k,v [nkv,Sk,D]; mask [T,Sk] additive. GQA block map. fp32 softmax. -> [H,T,D]
    H, T, D = q.shape
    nkv = k.shape[0]
    rep = H // nkv
    kk = k.repeat_interleave(rep, dim=0).float()   # [H,Sk,D]
    vv = v.repeat_interleave(rep, dim=0).float()
    scale = D ** -0.5
    scores = torch.einsum("htd,hjd->htj", q.float(), kk) * scale + mask.float()[None]  # [H,T,Sk]
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("htj,hjd->htd", w, vv).to(q.dtype)


def build_kernel(D):
    return TorchMetalKernel(
        name="tree_attn_verify", input_names=["A", "K", "V", "M"], result_names=["CTX"],
        src=_TREE_SRC.replace("__MAXEPT__", str(_MAX_EPT)).replace("__SCALE__", repr(float(D ** -0.5))),
        torch_defn=_tree_defn,
        metal_params=[MetalParameter("gid", "uint3", "thread_position_in_grid")],
        template_dtypes={"A": "TYPE"})


class TreeAttn(nn.Module):
    def __init__(self, kernel):
        super().__init__(); self.kernel = kernel

    def forward(self, q, k, v, mask):
        H, T, D = q.shape
        return self.kernel(q, k, v, mask, threads_per_grid=(32, H, T),
                           threads_per_thread_group=(32, 1, 1), result_shapes=[[H, T, D]])


def make_tree_mask(T, prefix):
    # a simple binary tree over T nodes: node i's parent = (i-1)//2; node attends prefix (all) + ancestors.
    Sk = prefix + T
    mask = torch.full((T, Sk), float("-inf"))
    mask[:, :prefix] = 0.0                          # all tree nodes see the whole prefix
    for i in range(T):
        j = i
        while j >= 0:                               # self + ancestor chain
            mask[i, prefix + j] = 0.0
            j = (j - 1) // 2 if j > 0 else -1
    return mask


async def main():
    torch.manual_seed(0)
    H, nkv, D = 8, 2, 256           # GQA 4:1, head_dim 256
    T, prefix = 15, 40              # 15-node tree (depth-4 binary), 40-token prefix
    Sk = prefix + T
    q = torch.randn(H, T, D, dtype=torch.float16)
    k = torch.randn(nkv, Sk, D, dtype=torch.float16)
    v = torch.randn(nkv, Sk, D, dtype=torch.float16)
    mask = make_tree_mask(T, prefix)                # [T, Sk] additive
    ref = _tree_defn(q, k, v, mask).float().numpy().reshape(-1)

    kernel = build_kernel(D)
    ep = torch.export.export(TreeAttn(kernel).eval(), (q, k, v, mask)).run_decompositions(get_decomp_table())
    conv = TorchConverter(); conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["A", "K", "V", "M"], output_names=["CTX"])
    prog = conv.to_coreai(); prog.optimize()
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "tree.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            out = await fn({"A": NDArray(q.numpy()), "K": NDArray(k.numpy()),
                            "V": NDArray(v.numpy()), "M": NDArray(mask.numpy())})
            g = list(out.values())[0].numpy().astype(np.float32).reshape(-1)
    maxdiff = float(np.abs(g - ref).max())
    cos = float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-12))
    print(f"== tree-attn verify kernel ==  H={H} nkv={nkv} D={D} T={T} prefix={prefix}")
    print(f"  GPU vs torch masked-attn: maxdiff={maxdiff:.2e} cos={cos:.6f} -> "
          f"{'PASS ✅' if (maxdiff < 2e-2 and cos > 0.9999) else 'FAIL ❌'}")


if __name__ == "__main__":
    asyncio.run(main())
