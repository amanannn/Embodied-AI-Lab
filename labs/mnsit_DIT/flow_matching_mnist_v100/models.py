"""
【模型定义模块 - models.py】

【作用】
    定义条件生成模型：ConditionalUNet（主力）和 DiT（可选扩展）。
    两者均支持时间步 t 和类别标签 label 的条件注入。
    通过命令行参数 --model_type unet|dit 切换。

【关键类】
    ConditionalUNet  - 条件 U-Net（含时间/类别嵌入 + 跳跃连接）
    DiT              - 轻量级 Diffusion Transformer（自适应 LayerNorm）
    TimestepEmbedding - 正弦位置编码 + MLP 时间嵌入
    ResBlock         - U-Net 基本残差块（含 FiLM 条件调制）

【输入输出】
    输入:  x       [B, 1, 28, 28]  噪声图像
           t       [B]              时间步 (0~1 连续值)
           label   [B]              类别标签 (0~9)
    输出:  v_pred  [B, 1, 28, 28]  预测的速度场
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 时间步嵌入：将标量时间 t 编码为高维向量
# ============================================================
class TimestepEmbedding(nn.Module):
    """
    正弦位置编码 + 两层 MLP。
    将连续时间步 t ∈ [0,1] 映射为高维特征向量。
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: 嵌入维度（同时也是正弦编码的基础维度）
        """
        super().__init__()
        self.dim = dim
        # 两层 MLP，中间扩展 4 倍维度
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),                              # Swish 激活函数
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 时间步，值域 [0, 1]
        Returns:
            emb: [B, dim*4] 时间嵌入向量
        """
        # 正弦位置编码（与 Transformer 位置编码相同原理）
        half_dim = self.dim // 2
        # 频率按指数递减: 1 / (10000^(2i/d))
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        # t: [B] → [B, 1] * [1, half_dim] → [B, half_dim]
        arg = t.float().unsqueeze(1) * freq.unsqueeze(0)
        embedding = torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)
        # 通过 MLP 进一步变换
        return self.mlp(embedding)


# ============================================================
# 残差块：U-Net 的基本构建单元（含 FiLM 条件调制）
# ============================================================
class ResBlock(nn.Module):
    """
    两卷积残差块，通过 FiLM (Feature-wise Linear Modulation) 注入时间条件。
    结构: GroupNorm → SiLU → Conv → +time_scale/shift → GroupNorm → SiLU → Conv → +residual
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, use_upsample: bool = False):
        """
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            time_emb_dim: 时间嵌入维度（用于 FiLM 调制）
            use_upsample: 是否在此块后上采样（解码器中使用）
        """
        super().__init__()
        self.use_upsample = use_upsample

        # 第一个卷积组
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # 时间嵌入投影 → scale 和 shift（FiLM 调制参数）
        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)

        # 第二个卷积组
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # 跳跃连接：如果通道数不一致，用 1×1 卷积对齐
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        # 可选上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if use_upsample else None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W] 特征图
            time_emb: [B, time_emb_dim] 时间嵌入
        Returns:
            out: [B, C_out, H', W'] 变换后的特征图
        """
        residual = self.skip(x)

        # 第一个卷积 + FiLM 调制
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # FiLM: 学习到的 scale 和 shift 调制特征
        scale1, shift1 = self.time_proj(time_emb).chunk(2, dim=1)
        h = h * (1 + scale1.unsqueeze(-1).unsqueeze(-1)) + shift1.unsqueeze(-1).unsqueeze(-1)

        # 第二个卷积
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # 残差连接
        out = h + residual

        # 可选上采样
        if self.use_upsample:
            out = self.upsample(out)

        return out


# ============================================================
# 条件 U-Net：编码器-瓶颈-解码器 + 跳跃连接
# ============================================================
class ConditionalUNet(nn.Module):
    """
    条件 U-Net 模型，专为 28×28 MNIST 设计。

    架构概览:
        [编码器]                     [解码器]
        1→64  (28×28)  ──skip──→  64→1   (28×28)
        64→128(14×14)  ──skip──→ 256→128(28×28)
        128→256(7×7)   ──skip──→ 512→256(14×14)
               └── 瓶颈 256→256 (7×7) ──┘

    时间步 t 和类别标签通过 FiLM 注入每个 ResBlock。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,              # 基础通道数
        channel_mult: tuple = (1, 2, 4),      # 各层通道倍增 [64, 128, 256]
        time_dim: int = 128,                   # 时间嵌入维度
        num_classes: int = 10,                 # MNIST 类别数
    ):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes

        # 计算各层通道数
        chs = [base_channels * m for m in channel_mult]  # [64, 128, 256]

        # --- 时间与类别嵌入 ---
        self.time_embed = TimestepEmbedding(dim=time_dim)
        time_emb_dim = time_dim * 4                       # MLP 输出维度
        self.class_embed = nn.Embedding(num_classes, time_emb_dim)

        # --- 编码器（下采样路径）---
        self.in_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            ResBlock(chs[0], chs[1], time_emb_dim),
            nn.Conv2d(chs[1], chs[1], kernel_size=3, stride=2, padding=1),  # 28→14
        )
        self.down2 = nn.Sequential(
            ResBlock(chs[1], chs[2], time_emb_dim),
            nn.Conv2d(chs[2], chs[2], kernel_size=3, stride=2, padding=1),  # 14→7
        )

        # --- 瓶颈层 ---
        self.mid_block1 = ResBlock(chs[2], chs[2], time_emb_dim)
        self.mid_block2 = ResBlock(chs[2], chs[2], time_emb_dim)

        # --- 解码器（上采样路径）---
        # 上采样块（输入通道 ×2 因为拼接了跳跃连接）
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),                     # 7→14
            nn.Conv2d(chs[2], chs[1], kernel_size=3, padding=1),
        )
        self.up2_block = ResBlock(chs[1] * 2, chs[1], time_emb_dim)          # ×2 因为 cat skip

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),                     # 14→28
            nn.Conv2d(chs[1], chs[0], kernel_size=3, padding=1),
        )
        self.up1_block = ResBlock(chs[0] * 2, chs[0], time_emb_dim)

        # --- 输出层 ---
        self.out_norm = nn.GroupNorm(num_groups=min(32, chs[0]), num_channels=chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x:     [B, 1, 28, 28]  输入图像（噪声/中间状态）
            t:     [B]              时间步 ∈ [0, 1]
            label: [B]              类别标签 0~9

        Returns:
            v:     [B, 1, 28, 28]  预测的速度场
        """
        # 1. 构建条件嵌入: time_emb + class_emb
        t_emb = self.time_embed(t)                          # [B, time_dim*4]
        c_emb = self.class_embed(label)                     # [B, time_dim*4]
        cond_emb = t_emb + c_emb                            # 相加融合

        # 2. 编码器路径
        e0 = self.in_conv(x)                                # [B, 64, 28, 28]
        e1 = self.down1[0](e0, cond_emb)                    # ResBlock
        e1 = self.down1[1](e1)                              # 下采样 → [B, 128, 14, 14]
        e2 = self.down2[0](e1, cond_emb)                    # ResBlock
        e2 = self.down2[1](e2)                              # 下采样 → [B, 256, 7, 7]

        # 3. 瓶颈层
        mid = self.mid_block1(e2, cond_emb)                 # [B, 256, 7, 7]
        mid = self.mid_block2(mid, cond_emb)                # [B, 256, 7, 7]

        # 4. 解码器路径（拼接跳跃连接）
        d2 = self.up2(mid)                                  # [B, 128, 14, 14]
        d2 = torch.cat([d2, e1], dim=1)                     # [B, 256, 14, 14] cat skip
        d2 = self.up2_block(d2, cond_emb)                   # [B, 128, 14, 14]

        d1 = self.up1(d2)                                   # [B, 64, 28, 28]
        d1 = torch.cat([d1, e0], dim=1)                     # [B, 128, 28, 28] cat skip
        d1 = self.up1_block(d1, cond_emb)                   # [B, 64, 28, 28]

        # 5. 输出
        out = self.out_norm(d1)
        out = F.silu(out)
        out = self.out_conv(out)                            # [B, 1, 28, 28]

        return out


# ============================================================
# 轻量级 DiT (Diffusion Transformer)
# 使用自适应 LayerNorm (adaLN) 注入时间与类别条件
# ============================================================
class AdaRMSNorm(nn.Module):
    """
    自适应 RMSNorm：根据条件嵌入动态调整 scale 和 shift。
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        # 确保维度对齐: cond 可能是 [B, dim] 或 [B, N, dim]
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    DiT Transformer 块：自注意力 + FFN，均通过 adaLN 注入条件。
    """

    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = AdaRMSNorm(dim, cond_dim)
        self.norm2 = AdaRMSNorm(dim, cond_dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN + 残差
        h = self.norm2(x, cond)
        h = self.mlp(h)
        x = x + h
        return x


class DiT(nn.Module):
    """
    轻量级 Diffusion Transformer (DiT)。

    将 28×28 图像切分为 4×4 的 patch（共 49 个 token），
    通过 6 层 Transformer + 自适应 LayerNorm 注入时间/类别条件，
    最后反嵌入还原为 28×28 速度场。

    参数量约 8M，V100 可轻松训练。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        cond_dim: int = 256,
        num_classes: int = 10,
        time_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 计算 patch 数量：28 / 4 = 7 → 7×7 = 49
        self.img_size = 28
        self.num_patches = (self.img_size // patch_size) ** 2  # 49

        # Patch 嵌入：Conv2d 一步完成切分和嵌入
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        # 时间嵌入
        self.time_embed = TimestepEmbedding(dim=time_dim)
        # 类别嵌入
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        # 条件投影（融合时间+类别）
        self.cond_proj = nn.Sequential(
            nn.Linear(time_dim * 4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Transformer 块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, cond_dim, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])

        # 最终归一化与输出投影
        self.final_norm = AdaRMSNorm(hidden_dim, cond_dim)
        self.unpatchify = nn.Sequential(
            nn.Linear(hidden_dim, patch_size * patch_size * out_channels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, 1, 28, 28]
            t:     [B]
            label: [B]
        Returns:
            v:     [B, 1, 28, 28]
        """
        B = x.shape[0]

        # 1. Patch 嵌入: [B, 1, 28, 28] → [B, hidden_dim, 7, 7] → [B, 49, hidden_dim]
        patches = self.patch_embed(x)                       # [B, C, 7, 7]
        patches = patches.flatten(2).transpose(1, 2)        # [B, 49, C]
        patches = patches + self.pos_embed                  # 加位置编码

        # 2. 条件嵌入
        t_emb = self.time_embed(t)                          # [B, time_dim*4]
        c_emb = self.class_embed(label)                     # [B, cond_dim]
        cond = self.cond_proj(t_emb) + c_emb                # [B, cond_dim]

        # 3. Transformer 编码
        h = patches
        for block in self.blocks:
            h = block(h, cond)

        # 4. 输出头
        h = self.final_norm(h, cond)                        # [B, 49, hidden_dim]
        tokens = self.unpatchify(h)                         # [B, 49, patch_size²]

        # 5. 重组为图像: [B, 49, 16] → [B, 1, 28, 28]
        tokens = tokens.reshape(B, self.num_patches, self.patch_size, self.patch_size)
        # 将 1D patch 序列重组为 2D 网格
        grid_size = int(self.num_patches ** 0.5)            # 7
        img = tokens.reshape(
            B, grid_size, grid_size, self.patch_size, self.patch_size
        )
        img = img.permute(0, 1, 3, 2, 4).reshape(B, 1, self.img_size, self.img_size)

        return img


# ============================================================
# 模型工厂函数：根据名称创建对应模型
# ============================================================
def create_model(model_type: str = "unet", **kwargs) -> nn.Module:
    """
    模型工厂：根据字符串创建 ConditionalUNet 或 DiT。

    Args:
        model_type: "unet" 或 "dit"
        **kwargs: 传递给具体模型构造函数的参数

    Returns:
        model: nn.Module 实例
    """
    model_type = model_type.lower()
    if model_type == "unet":
        return ConditionalUNet(**kwargs)
    elif model_type == "dit":
        return DiT(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: '{model_type}'，可选: 'unet', 'dit'")


# ============================================================
# 模块自测：验证模型前向传播与参数统计
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用设备: {device}")

    # 测试输入
    B = 4
    x = torch.randn(B, 1, 28, 28).to(device)
    t = torch.rand(B).to(device)
    label = torch.randint(0, 10, (B,)).to(device)

    # --- 测试 ConditionalUNet ---
    print("\n" + "=" * 60)
    print("测试 ConditionalUNet")
    print("=" * 60)
    unet = ConditionalUNet().to(device)
    with torch.no_grad():
        out = unet(x, t, label)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"   输入: {x.shape} → 输出: {out.shape}")
    print(f"   参数量: {n_params:,} (~{n_params/1e6:.1f}M)")
    assert out.shape == x.shape, f"形状不匹配: {out.shape} != {x.shape}"
    print("   ✅ 前向传播通过")

    # --- 测试 DiT ---
    print("\n" + "=" * 60)
    print("测试 DiT (Diffusion Transformer)")
    print("=" * 60)
    dit = DiT().to(device)
    with torch.no_grad():
        out = dit(x, t, label)
    n_params = sum(p.numel() for p in dit.parameters())
    print(f"   输入: {x.shape} → 输出: {out.shape}")
    print(f"   参数量: {n_params:,} (~{n_params/1e6:.1f}M)")
    assert out.shape == x.shape, f"形状不匹配: {out.shape} != {x.shape}"
    print("   ✅ 前向传播通过")

    # --- 测试 create_model 工厂函数 ---
    print("\n" + "=" * 60)
    print("测试 create_model 工厂函数")
    print("=" * 60)
    for mtype in ["unet", "dit"]:
        m = create_model(mtype).to(device)
        with torch.no_grad():
            o = m(x, t, label)
        print(f"   create_model('{mtype}'): {type(m).__name__} → {o.shape} ✅")
