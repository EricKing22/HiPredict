from __future__ import annotations

import torch.nn as nn
import torch
import torch.nn.functional as F


# Conv blocks with residual connections
class ResidualBlock2d(nn.Module):
    """
    Local convolutional feature block with an identity shortcut.
    The shortcut preserves signal while deeper convolutions learn residual structure.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convs: int = 2,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ):
        super().__init__()
        if not 1 <= n_convs <= 4:
            raise ValueError("n_convs must be an integer from 1 to 4 (inclusive)")

        padding = (kernel_size - 1) // 2


        def make_norm(c: int):
            return LayerNorm2d(c)

        blocks = []
        in_c = in_channels
        for i in range(n_convs):
            out_c = out_channels
            blocks.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False, padding_mode="reflect"))
            blocks.append(make_norm(out_c))
            if i < n_convs - 1:
                blocks.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.main = nn.Sequential(*blocks)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels),
            )

        self.drop = nn.Dropout2d(p=p_drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x)
        out = out + self.shortcut(x)
        out = self.drop(out)
        out = self.relu(out)
        return out


# Normalisation which ensures all pixels are mathematically equivalent, translationally invariant.
class LayerNorm2d(nn.Module):
    """
    Layer normalization over channel dimension for 2D feature maps.
    Translation equivariant - normalizes per-pixel over channels only.
    """
    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


# Anti-aliased downsampling layer
class LowPassDown2(nn.Module):
    """
    Downsample after channel-wise Gaussian blur.
    The blur reduces aliasing before spatial resolution is halved.
    """
    f: torch.Tensor

    def __init__(self, c, k=5, sigma=1.0):
        super().__init__()
        coords = torch.arange(k).float() - (k - 1) / 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d.view(k, 1) @ g1d.view(1, k)
        self.register_buffer('f', g2d.view(1, 1, k, k))
        self.c = c
        self.k = k

    def forward(self, x):
        w = self.f.expand(self.c, 1, self.k, self.k)
        pad = self.k // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x = F.conv2d(x, w, stride=1, groups=self.c)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


# Fast local box averaging for patch-safe brightness/context estimates
class RobustSpatialAvg(nn.Module):
    """
    Sliding spatial mean computed with cumulative sums.
    Used where local, translation-equivariant statistics are needed.
    """
    def __init__(self, channels, k_h, k_w, eps=1e-6, pad_mode="constant"):
        super().__init__()
        assert k_h % 2 == 1 and k_w % 2 == 1
        self.c = channels
        self.k_h = k_h
        self.k_w = k_w
        self.eps = eps
        self.pad_mode = pad_mode
        self._den_cache = {}

    def _box_sum(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            acc_dtype = torch.float32
        else:
            acc_dtype = x.dtype

        pad_h = self.k_h // 2
        pad_w = self.k_w // 2

        if self.pad_mode == "reflect":
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        else:
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

        s = torch.cumsum(x, dim=2, dtype=acc_dtype)
        s = torch.cumsum(s, dim=3, dtype=acc_dtype)
        s = F.pad(s, (1, 0, 1, 0), mode="constant", value=0.0)

        kh = self.k_h
        kw = self.k_w
        result = s[:, :, kh:, kw:] - s[:, :, :-kh, kw:] - s[:, :, kh:, :-kw] + s[:, :, :-kh, :-kw]
        return result.to(orig_dtype)

    def _den(self, x: torch.Tensor) -> torch.Tensor | float:
        if self.pad_mode == "reflect":
            # fixed full support everywhere after reflect pad
            return float(self.k_h * self.k_w)

        H, W = x.shape[-2], x.shape[-1]
        key = (H, W, x.dtype, x.device)
        den = self._den_cache.get(key, None)
        if den is None:
            ones = torch.ones((1, 1, H, W), device=x.device, dtype=x.dtype)
            den = self._box_sum(ones).clamp_min(self.eps)
            self._den_cache[key] = den
        return den

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.c
        num = self._box_sum(x)
        den = self._den(x)
        return num / den


# Explicit local brightness head anchored to neighbouring RED channels
class BrightnessHeadLocal(nn.Module):
    """
    Single, identifiable, local brightness mechanism.

    Predicts a spatial brightness map from local target-band means plus bounded
    RED3/RED5 side-calibration features. The direct anchor is:

      m_base = 0.5 * (mu_ir + mu_bg) + blended(red - mid)

    A small learned residual lets metadata/geometry adjust that calibration
    without giving the head raw side-strip image content.
    """
    def __init__(self, n_meta=0, hidden=32, beta_scale=0.1):
        super().__init__()
        self.n_meta = n_meta
        self.beta_scale = beta_scale

        ctx_ch = 2 + 6 + 3 + n_meta
        self.net = nn.Sequential(
            nn.Conv2d(ctx_ch, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

        # Initialize residual to zero; the calibration baseline still passes through.
        last = self.net[-1]
        assert isinstance(last, nn.Conv2d)
        assert last.bias is not None
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, mu_band_map, calib_ctx, geometry, x_meta_spatial=None):
        if x_meta_spatial is None:
            ctx = torch.cat([mu_band_map, calib_ctx, geometry], dim=1)
        else:
            ctx = torch.cat([mu_band_map, calib_ctx, geometry, x_meta_spatial], dim=1)

        delta = self.beta_scale * self.net(ctx)
        mu_mid = 0.5 * (mu_band_map[:, 0:1] + mu_band_map[:, 1:2])
        red_minus_mid = calib_ctx[:, 2:3]
        return mu_mid + red_minus_mid + delta


class SideCalibrationContext(nn.Module):
    """
    Builds bounded side-neighbour calibration maps for the convolutional U-Net.

    RED3/RED5 side strips are top-aligned to the target patch, reduced across
    their narrow overlap width, smoothed vertically, and blended by horizontal
    position so they act as calibration context rather than target image content.
    """
    def __init__(self, smooth_kernel=65, calib_clip=4.0, ratio_eps=0.08, chroma_clip=3.0):
        super().__init__()
        self.smooth_kernel = smooth_kernel
        self.calib_clip = calib_clip
        self.ratio_eps = ratio_eps
        self.chroma_clip = chroma_clip
        self.out_channels = 6

    def _target_rows(self, x_neigh, height):
        if x_neigh.size(-2) >= height:
            return x_neigh[:, :, :height, :]
        return F.interpolate(x_neigh, size=(height, x_neigh.size(-1)), mode='bilinear', align_corners=False)

    def _smooth_vertical(self, x):
        k = min(int(self.smooth_kernel), int(x.size(-2)))
        if k % 2 == 0:
            k -= 1
        if k <= 1:
            return x

        pad = k // 2
        pad_mode = "reflect" if x.size(-2) > pad else "replicate"
        x = F.pad(x, (0, 0, pad, pad), mode=pad_mode)
        return F.avg_pool2d(x, kernel_size=(k, 1), stride=1)

    def _side_calibration_features(self, bg, ir, red):
        mid = 0.5 * (ir + bg)
        delta_ib = ir - bg
        denom = torch.where(
            delta_ib >= 0,
            delta_ib.clamp_min(self.ratio_eps),
            delta_ib.clamp_max(-self.ratio_eps),
        )
        chroma = (red - bg) / denom
        chroma = chroma.clamp(-self.chroma_clip, self.chroma_clip)

        return torch.cat(
            [
                red - ir,
                red - bg,
                red - mid,
                chroma,
                ir - bg,
            ],
            dim=1,
        )

    def _summarise_side_calibration(self, bg, ir, red, height, width):
        feats = self._side_calibration_features(bg, ir, red)
        feats = feats.median(dim=3, keepdim=True).values
        feats = self._smooth_vertical(feats)
        feats = feats.clamp(-self.calib_clip, self.calib_clip)
        return feats.expand(-1, -1, height, width)

    @staticmethod
    def _geometry_channels(x_band, patch_x=None, obs_width=None):
        B, Cb, Hb, Wb = x_band.shape
        local_x = torch.arange(Wb, device=x_band.device, dtype=x_band.dtype).view(1, 1, 1, Wb)

        if patch_x is None or obs_width is None:
            x_frac = local_x / max(Wb - 1, 1)
        else:
            patch_x = patch_x.to(device=x_band.device, dtype=x_band.dtype).view(B, 1, 1, 1)
            obs_width = obs_width.to(device=x_band.device, dtype=x_band.dtype).view(B, 1, 1, 1)
            x_frac = (patch_x + local_x) / (obs_width - 1.0).clamp_min(1.0)

        x_frac = x_frac.clamp(0.0, 1.0).expand(B, 1, Hb, Wb)
        left_weight = 1.0 - x_frac
        right_weight = x_frac
        return torch.cat([x_frac, left_weight, right_weight], dim=1)

    def forward(self, x_band, x_neigh, patch_x=None, obs_width=None):
        B, Cb, height, width = x_band.shape
        neigh = self._target_rows(x_neigh, height)
        geometry = self._geometry_channels(x_band, patch_x=patch_x, obs_width=obs_width)

        r5_bg = neigh[:, 0:1]
        r5_ir = neigh[:, 1:2]
        r5_red = neigh[:, 2:3]
        r3_bg = neigh[:, 3:4]
        r3_ir = neigh[:, 4:5]
        r3_red = neigh[:, 5:6]

        left_r3 = self._summarise_side_calibration(r3_bg, r3_ir, r3_red, height, width)
        right_r5 = self._summarise_side_calibration(r5_bg, r5_ir, r5_red, height, width)

        blended = geometry[:, 1:2] * left_r3 + geometry[:, 2:3] * right_r5
        disagreement = (left_r3 - right_r5).abs().mean(dim=1, keepdim=True)
        disagreement = disagreement.clamp(0.0, self.calib_clip)
        calib_ctx = torch.cat([blended, disagreement], dim=1)
        return calib_ctx, geometry


# Residual U-Net with an explicit RED3/RED5 brightness calibration head
class UNetCNN_conv(nn.Module):
    """
    Predicts RED4 from IR/BG bands plus aligned neighbour context.
    The U-Net learns spatial detail; the brightness head supplies local DC level from
    neighbouring RED observations.
    """
    def __init__(self, in_ch, out_ch=1, base_ch=64, depth=4, n_res_blocks=1, kernel_size=3, p_drop=0.25, n_convs=2, align_corners=False, n_meta=0):
        super().__init__()
        assert 0 <= p_drop < 1.0, "p_drop should be a probability"
        assert 0 < n_convs <= 4, "n_convs should be an integer from 1 to 4 (inclusive)"

        # ------------------------- config --------------------------
        self.enc_cfg, self.bot_cfg = stage_hyperparams(
            depth=depth,
            n_convs=n_convs,
            p_drop=p_drop,
            n_res_blocks=n_res_blocks,
        )
        self.dec_cfg = list(reversed(self.enc_cfg))
        self.align_corners = align_corners
        self.unet_in_ch = in_ch - n_meta
        self.n_meta = n_meta
        assert self.unet_in_ch == 8, f"UNET expects 8 image channels, got {self.unet_in_ch}."

        self.side_calib = SideCalibrationContext()
        self.n_calib_features = self.side_calib.out_channels

        # -------------------------  Encoder - Down-sampling path -------------------------
        self.encoder = nn.ModuleList()
        enc_channels = []

        curr_in = self.unet_in_ch
        curr_out = base_ch
        for cfg in self.enc_cfg:
            blocks = []
            for _ in range(int(cfg["n_res_blocks"])):
                blocks.append(ResidualBlock2d(curr_in, curr_out, kernel_size=kernel_size, p_drop=cfg["p_drop"], n_convs=int(cfg["n_convs"])))
                curr_in = curr_out
            self.encoder.append(nn.Sequential(*blocks))
            enc_channels.append(curr_out)
            curr_out *= 2

        self.down_ops = nn.ModuleList()
        # Anti-alias for the first two stages then standard avg pool
        for i, c in enumerate(enc_channels):
            if i == 0:
                self.down_ops.append(LowPassDown2(c, k=5, sigma=0.6))
            elif i == 1:
                self.down_ops.append(LowPassDown2(c, k=5, sigma=0.5))
            else:
                self.down_ops.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # -------------------------  Bottleneck -------------------------
        self.bottleneck = ResidualBlock2d(
            enc_channels[-1],
            base_ch * (2**depth),
            kernel_size=kernel_size,
            p_drop=self.bot_cfg["p_drop"],
            n_convs=int(self.bot_cfg["n_convs"])
        )

        # ------------------------- Decoder - Up-sampling path -------------------------
        self.decoder = nn.ModuleList()
        dec_channels = enc_channels[::-1]  # Reverse the channels for decoder

        for i, cfg in enumerate(self.dec_cfg):
            out_c = dec_channels[i]

            if i == 0:
                in_c = base_ch * (2**depth) + enc_channels[-1]  # First decoder input is the bottleneck + last encoder output
                stage_drop = self.bot_cfg["p_drop"]             # Use bottleneck dropout for the first decoder stage
            else:
                in_c = dec_channels[i - 1] + enc_channels[depth - i - 1]
                stage_drop = cfg["p_drop"]                      # No dropout in the remaining decoder stages (enc vals)

            blocks = []
            for j in range(int(cfg["n_res_blocks"])):
                blk_in = in_c if j == 0 else out_c
                blk_drop = stage_drop if j == 0 else cfg["p_drop"]
                blocks.append(
                    ResidualBlock2d(
                        blk_in,
                        out_c,
                        kernel_size=kernel_size,
                        p_drop=blk_drop,
                        n_convs=int(cfg["n_convs"])
                    )
                )
            self.decoder.append(nn.Sequential(*blocks))

        # ------------------------- Final convolution -------------------------
        self.final_conv = nn.Conv2d(base_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='reflect')

        # ------------------------- Mean head to calibrate global frequency -------------------------
        self.mean_head = BrightnessHeadLocal(n_meta=n_meta, hidden=32, beta_scale=0.1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        # ------------------------- local spatial averages for translation equivariance -------------------------
        self.local_avg_kernel = 129    # Must be odd
        self.output_local_avg = RobustSpatialAvg(1, self.local_avg_kernel, self.local_avg_kernel, pad_mode="reflect")
        self.band_spatial_avg = RobustSpatialAvg(2, self.local_avg_kernel, self.local_avg_kernel, pad_mode="reflect")

        # ---------------------------------------------------------------------------------------------

    def forward(self, x_band, x_neigh, x_meta, patch_x=None, obs_width=None):
        """
        x_band:  (B, 2, Hb, Wb)   IR10, BG12 patch at target
        x_neigh: (B, 6, Hn, Wn)   2 neighbours (3 chans each)
        x_meta:  (B, n_meta) or None  camera / obs metadata
        """

        # ------------------------- build neighbour context maps -------------------------
        B, Cb, Hb, Wb = x_band.shape
        Bn, Cn, Hn, Wn = x_neigh.shape

        assert Cb == 2, f"expected 2 band channels, got {Cb}"
        assert Cn == 6, f"expected 6 neighbour channels, got {Cn}"

        calib_ctx, geometry = self.side_calib(x_band, x_neigh, patch_x=patch_x, obs_width=obs_width)
        if getattr(self, "disable_neigh_ctx", False):
            calib_ctx = torch.zeros_like(calib_ctx)

        # --- spatial band context maps via dedicated band averaging ---
        mu_band_map = self.band_spatial_avg(x_band)            # [B, 2, Hb, Wb] - local means at each pixel

        # Build 8-channel image input for U-Net:
        # 2 target bands + 6 bounded side-calibration maps.
        x_img = torch.cat([x_band, calib_ctx], dim=1)

        # ------------------------- UNet encoder / bottleneck / decoder -------------------------
        skips = []
        x = x_img
        for i, enc in enumerate(self.encoder):
            x = enc(x)                                          # residual blocks
            skips.append(x)                                     # save skip feature
            x = self.down_ops[i](x)                             # anti-aliased / avg pooling downsample

        x = self.bottleneck(x)                                  # bottleneck residual block

        for i, dec in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='nearest')  # upsample by 2
            if i < 1:
                # box smooth after upsample to suppress replication spectrum
                x = F.avg_pool2d(
                    F.pad(x, (1, 1, 1, 1), mode='reflect'),
                    kernel_size=3,
                    stride=1,
                )
            skip = skips[-(i + 1)]                              # corresponding skip from encoder
            x = torch.cat([x, skip], dim=1)                     # concat skip
            x = dec(x)                                          # decoder residual blocks

        out = self.final_conv(x)                                # raw HF + LF corrected output

        # --------------------------- center output using local spatial averages -------------------------
        mu_raw_map = self.output_local_avg(out)    # [B, 1, H, W] - per-pixel local mean
        self.mu_raw = mu_raw_map.detach()

        if getattr(self, "disable_dc_center", False):
            out_centered = out
        else:
            out_centered = out - mu_raw_map         # locally centered before DC correction

        # ------------------------- build features for mean head -------------------------
        if self.n_meta > 0:
            assert x_meta is not None, "x_meta must be provided when n_meta > 0"
            x_meta_spatial = x_meta.view(B, -1, 1, 1).expand(-1, -1, Hb, Wb)    # [B, n_meta, H, W]
        else:
            x_meta_spatial = None
        m_hat = self.mean_head(mu_band_map, calib_ctx, geometry, x_meta_spatial)

        out = out_centered + m_hat
        out = out + self.global_bias

        # ------------------------- bookkeeping for debugging -------------------------
        self.m_hat_live = m_hat
        self.m_hat = m_hat.detach()              # Now [B, 1, H, W] instead of [B, 1, 1, 1]
        self.out_centered = out_centered.detach()
        self.lf_bias = torch.zeros_like(m_hat).detach()     # For diagnostics
        self.global_bias_value = self.global_bias.item()

        return out


# Residual U-Net with learned local mean and low-frequency correction
class UNetCNN_local(nn.Module):
    """
    Patch-safe RED4 predictor using local input averages for brightness calibration.
    A learned low-frequency branch can add spatial bias after local centering.
    """
    def __init__(self, in_ch, out_ch=1, base_ch=64, depth=4, n_res_blocks=1, kernel_size=3, p_drop=0.25, n_convs=2, align_corners=False, n_meta=0):
        super().__init__()
        assert 0 <= p_drop < 1.0, "p_drop should be a probability"
        assert 0 < n_convs <= 4, "n_convs should be an integer from 1 to 4 (inclusive)"

        # ------------------------- config --------------------------
        self.enc_cfg, self.bot_cfg = stage_hyperparams(
            depth=depth,
            n_convs=n_convs,
            p_drop=p_drop,
            n_res_blocks=n_res_blocks,
        )
        self.dec_cfg = list(reversed(self.enc_cfg))
        self.align_corners = align_corners
        self.unet_in_ch = in_ch - n_meta
        self.n_meta = n_meta
        assert self.unet_in_ch == 8, f"UNET expects 8 image channels, got {self.unet_in_ch}."

        # ------------------------- neighbour context aggregator -------------------------
        self.neigh_kernel_h = 257   # must be odd
        self.neigh_kernel_w = 33    # must be odd

        self.neigh_spatial_avg = RobustSpatialAvg(6, self.neigh_kernel_h, self.neigh_kernel_w, pad_mode="reflect")
        self.neigh_width_avg = RobustSpatialAvg(6, 1, 5, pad_mode="reflect")

        # -------------------------  Encoder - Down-sampling path -------------------------
        self.encoder = nn.ModuleList()
        enc_channels = []

        curr_in = self.unet_in_ch
        curr_out = base_ch
        for cfg in self.enc_cfg:
            blocks = []
            for _ in range(int(cfg["n_res_blocks"])):
                blocks.append(ResidualBlock2d(curr_in, curr_out, kernel_size=kernel_size, p_drop=cfg["p_drop"], n_convs=int(cfg["n_convs"])))
                curr_in = curr_out
            self.encoder.append(nn.Sequential(*blocks))
            enc_channels.append(curr_out)
            curr_out *= 2

        self.down_ops = nn.ModuleList()
        # Anti-alias for the first two stages then standard avg pool
        for i, c in enumerate(enc_channels):
            if i == 0:
                self.down_ops.append(LowPassDown2(c, k=5, sigma=1.0))
            elif i == 1:
                self.down_ops.append(LowPassDown2(c, k=5, sigma=0.9))
            else:
                self.down_ops.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # -------------------------  Bottleneck -------------------------
        self.bottleneck = ResidualBlock2d(
            enc_channels[-1],
            base_ch * (2**depth),
            kernel_size=kernel_size,
            p_drop=self.bot_cfg["p_drop"],
            n_convs=int(self.bot_cfg["n_convs"])
        )

        # ------------------------- Decoder - Up-sampling path -------------------------
        self.decoder = nn.ModuleList()
        dec_channels = enc_channels[::-1]  # Reverse the channels for decoder

        for i, cfg in enumerate(self.dec_cfg):
            out_c = dec_channels[i]

            if i == 0:
                in_c = base_ch * (2**depth) + enc_channels[-1]  # First decoder input is the bottleneck + last encoder output
                stage_drop = self.bot_cfg["p_drop"]             # Use bottleneck dropout for the first decoder stage
            else:
                in_c = dec_channels[i - 1] + enc_channels[depth - i - 1]
                stage_drop = cfg["p_drop"]                      # No dropout in the remaining decoder stages (enc vals)

            blocks = []
            for j in range(int(cfg["n_res_blocks"])):
                blk_in = in_c if j == 0 else out_c
                blk_drop = stage_drop if j == 0 else cfg["p_drop"]
                blocks.append(
                    ResidualBlock2d(
                        blk_in,
                        out_c,
                        kernel_size=kernel_size,
                        p_drop=blk_drop,
                        n_convs=int(cfg["n_convs"])
                    )
                )
            self.decoder.append(nn.Sequential(*blocks))

        # ------------------------- Final convolution -------------------------
        self.final_conv = nn.Conv2d(base_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='reflect')

        # ------------------------- Local mean calibration for patch-safe inference -------------------------
        self.local_avg_kernel = 129    # Must be odd
        self.output_local_avg = RobustSpatialAvg(1, self.local_avg_kernel, self.local_avg_kernel, pad_mode="reflect")
        self.input_local_avg = RobustSpatialAvg(8 + n_meta, self.local_avg_kernel, self.local_avg_kernel, pad_mode="reflect")

        self.mean_head = nn.Sequential(
            nn.Conv2d(8 + n_meta, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

        last = self.mean_head[-1]
        assert isinstance(last, nn.Conv2d)
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

        # ------------------------- lf conv for spatial variation in mean head -------------------------
        self.lf_conv = nn.Conv2d(dec_channels[-1], 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.lf_conv.weight)
        if self.lf_conv.bias is not None:
            nn.init.zeros_(self.lf_conv.bias)
        # ---------------------------------------------------------------------------------------------

    def forward(self, x_band, x_neigh, x_meta, patch_x=None, obs_width=None):

        B, Cb, Hb, Wb = x_band.shape
        Bn, Cn, Hn, Wn = x_neigh.shape

        assert Cb == 2, f"expected 2 band channels, got {Cb}"
        assert Cn == 6, f"expected 6 neighbour channels, got {Cn}"

        # Build aligned neighbour context maps from strip input
        neigh_ctx_full = self.neigh_spatial_avg(x_neigh)       # [B, 6, Hn, Wn]
        neigh_ctx_full = self.neigh_width_avg(neigh_ctx_full)  # [B, 6, Hn, Wn]

        # Collapse width and align height to target patch
        neigh_col = neigh_ctx_full.mean(dim=3, keepdim=True)  # [B, 6, Hn, 1]

        if neigh_col.shape[2] >= Hb:
            start = (neigh_col.shape[2] - Hb) // 2
            neigh_col_cropped = neigh_col[:, :, start:start + Hb, :]
        else:
            neigh_col_cropped = F.interpolate(neigh_col, size=(Hb, 1), mode='bilinear', align_corners=False)

        neigh_ctx_aligned = neigh_col_cropped.expand(-1, -1, -1, Wb)  # [B, 6, Hb, Wb]

        # Build standard 8-channel UNet image input
        x_img = torch.cat([x_band, neigh_ctx_aligned], dim=1)          # [B, 8, Hb, Wb]

        skips = []
        x = x_img
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            skips.append(x)
            x = self.down_ops[i](x)

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i < 1:
                x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel_size=3, stride=1)       # box smooth after upsample. Unit stride smoother to suppress the replication spectrum
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            if i == len(self.decoder) - 1:
                lf_low = self.lf_conv(x)


        out = self.final_conv(x)

        mu_raw = self.output_local_avg(out)                  # [B,1,H,W]
        out_centered = out - mu_raw

        if self.n_meta > 0:
            assert x_meta is not None, "x_meta must be provided when n_meta > 0"
            x_meta_spatial = x_meta.view(B, -1, 1, 1).expand(-1, -1, Hb, Wb)
            mean_in = torch.cat([x_img, x_meta_spatial], dim=1)   # [B, 8 + n_meta, H, W]
        else:
            x_meta_spatial = None
            mean_in = x_img                                        # [B, 8, H, W]

        assert mean_in.shape[1] == 8 + self.n_meta

        mean_in = self.input_local_avg(mean_in)
        m_hat = self.mean_head(mean_in)                            # [B, 1, H, W]
        out = out_centered + m_hat

        lf_bias = F.interpolate(lf_low, size=out.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        lf_bias = lf_bias - self.output_local_avg(lf_bias)

        out = out + lf_bias

        self.m_hat_live = m_hat
        self.m_hat = m_hat.detach()
        self.mu_raw = mu_raw.detach()
        self.out_centered = out_centered.detach()
        self.lf_bias = lf_bias.detach()
        return out


# Basic encoder-decoder U-Net baseline
class BasicUNet(nn.Module):
    """
    U-Net baseline over target bands plus low-frequency side calibration.

    RED3/RED5 side strips are reduced to smoothed overlap-calibration features
    before they reach the image branch, so they provide spectral scale/context
    without acting as copyable row-brightness images. The ratio feature is kept
    but bounded separately because it is the most volatile calibration channel.
    """
    def __init__(self, in_ch, out_ch=1, base_ch=64, kernel_size=3, n_meta=None):
        super().__init__()
        # in_ch not used as meta not used in this model
        self.calib_smooth_kernel = 65
        self.calib_clip = 4.0
        self.calib_ratio_eps = 0.08
        self.chroma_clip = 3.0
        n_geometry_channels = 3
        n_calib_features = 6
        model_in_ch = 2 + n_geometry_channels + n_calib_features

        # Neighbor prep is handled by smoothed side-calibration features in forward.

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(model_in_ch, base_ch)
        self.enc2 = self.conv_block(base_ch, base_ch * 2)
        self.enc3 = self.conv_block(base_ch * 2, base_ch * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_ch * 4, base_ch * 8)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_ch * 8, base_ch * 4)     # 256 (up) + 256 (skip)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_ch * 4, base_ch * 2)     # 128 (up) + 128 (skip)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_ch * 2, base_ch)  # 64 (up) + 64 (skip)

        # Final Output Layer
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _target_rows(self, x_neigh, height):
        if x_neigh.size(-2) >= height:
            return x_neigh[:, :, :height, :]
        return F.interpolate(x_neigh, size=(height, x_neigh.size(-1)), mode='bilinear', align_corners=False)

    def _smooth_vertical(self, x):
        k = min(int(self.calib_smooth_kernel), int(x.size(-2)))
        if k % 2 == 0:
            k -= 1
        if k <= 1:
            return x

        pad = k // 2
        pad_mode = "reflect" if x.size(-2) > pad else "replicate"
        x = F.pad(x, (0, 0, pad, pad), mode=pad_mode)
        return F.avg_pool2d(x, kernel_size=(k, 1), stride=1)

    def _side_calibration_features(self, bg, ir, red):
        mid = 0.5 * (ir + bg)
        delta_ib = ir - bg
        denom = torch.where(
            delta_ib >= 0,
            delta_ib.clamp_min(self.calib_ratio_eps),
            delta_ib.clamp_max(-self.calib_ratio_eps),
        )
        chroma = (red - bg) / denom
        chroma = chroma.clamp(-self.chroma_clip, self.chroma_clip)

        return torch.cat(
            [
                red - ir,
                red - bg,
                red - mid,
                chroma,
                ir - bg,
            ],
            dim=1,
        )

    def _summarise_side_calibration(self, bg, ir, red, height, width):
        feats = self._side_calibration_features(bg, ir, red)
        feats = feats.median(dim=3, keepdim=True).values
        feats = self._smooth_vertical(feats)
        feats = feats.clamp(-self.calib_clip, self.calib_clip)
        return feats.expand(-1, -1, height, width)

    def _context_calibration(self, x_neigh, height, width, left_weight, right_weight):
        neigh = self._target_rows(x_neigh, height)

        r5_bg = neigh[:, 0:1]
        r5_ir = neigh[:, 1:2]
        r5_red = neigh[:, 2:3]
        r3_bg = neigh[:, 3:4]
        r3_ir = neigh[:, 4:5]
        r3_red = neigh[:, 5:6]

        left_r3 = self._summarise_side_calibration(r3_bg, r3_ir, r3_red, height, width)
        right_r5 = self._summarise_side_calibration(r5_bg, r5_ir, r5_red, height, width)

        blended = left_weight * left_r3 + right_weight * right_r5
        disagreement = (left_r3 - right_r5).abs().mean(dim=1, keepdim=True)
        disagreement = disagreement.clamp(0.0, self.calib_clip)
        return torch.cat([blended, disagreement], dim=1)

    @staticmethod
    def _geometry_channels(x_band, patch_x=None, obs_width=None):
        B, Cb, Hb, Wb = x_band.shape
        local_x = torch.arange(Wb, device=x_band.device, dtype=x_band.dtype).view(1, 1, 1, Wb)

        if patch_x is None or obs_width is None:
            x_frac = local_x / max(Wb - 1, 1)
        else:
            patch_x = patch_x.to(device=x_band.device, dtype=x_band.dtype).view(B, 1, 1, 1)
            obs_width = obs_width.to(device=x_band.device, dtype=x_band.dtype).view(B, 1, 1, 1)
            x_frac = (patch_x + local_x) / (obs_width - 1.0).clamp_min(1.0)

        x_frac = x_frac.clamp(0.0, 1.0).expand(B, 1, Hb, Wb)
        left_weight = 1.0 - x_frac
        right_weight = x_frac
        return torch.cat([x_frac, left_weight, right_weight], dim=1)

    def forward(self, x_band, x_neigh, x_meta=None, patch_x=None, obs_width=None):
        B, Cb, Hb, Wb = x_band.shape
        geometry = self._geometry_channels(x_band, patch_x=patch_x, obs_width=obs_width)
        context = self._context_calibration(
            x_neigh,
            Hb,
            Wb,
            left_weight=geometry[:, 1:2],
            right_weight=geometry[:, 2:3],
        )
        x = torch.cat([x_band, geometry, context], dim=1)

        # Encoder
        s1 = self.enc1(x)
        p1 = F.max_pool2d(s1, 2)

        s2 = self.enc2(p1)
        p2 = F.max_pool2d(s2, 2)

        s3 = self.enc3(p2)
        p3 = F.max_pool2d(s3, 2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Skip Connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# Linear IR plus neighbouring RED baseline
class Linear(nn.Module):
    """
    Applies a learned linear convolution to IR10 and a row-mean RED reference.
    Provides a minimal trainable baseline for brightness transfer.
    """
    def __init__(self, in_ch, out_channels=1, kernel_size=1, n_meta=None, base_ccd="RED5"):
        super().__init__()
        assert isinstance(kernel_size, int), f"kernel_size must be an integer, got {type(kernel_size)}: {kernel_size}"
        assert base_ccd in ["RED5", "RED3"], f"base must be 'RED5' or 'RED3', got {base_ccd}"
        assert in_ch == 2, f"Expected in_ch to be 2 (IR10 + base CCD), got {in_ch} (Can be set in config)"
        self.linear_transform = nn.Conv2d(in_ch, out_channels, kernel_size, padding=0)
        self.base_idx = 2 if base_ccd == "RED5" else 5   # Idx of R5/R3 in x_neigh

    def forward(self, x_band, x_neigh, x_meta=None, patch_x=None, obs_width=None):
        ir = x_band[:, 0, :, :]                                      # shape => (B, Hb, Wb) - IR10
        base = x_neigh[:, self.base_idx, :, :]                       # shape => (B, Hn, Wn) - R3 or R5

        # isis2np saves context patches with the same top row as the target patch,
        # then extends them downward. The target-aligned crop is therefore the first Hb rows.
        base = base[:, :ir.size(1), :]
        base_row_means = base.mean(dim=2, keepdim=True)    # shape => (B, Hb, 1) - mean of R3/R5 across width for each row
        base_expanded = base_row_means.expand(-1, -1, ir.size(2))  # shape => (B, Hb, Wb)

        x = torch.cat((ir.unsqueeze(1), base_expanded.unsqueeze(1)), dim=1)   # shape => (B, 2, H, W)
        out = self.linear_transform(x)
        return out


# Linear baseline with broadcast metadata channels
class LinearWithMeta(nn.Module):
    """
    Extends the linear baseline by concatenating spatially broadcast metadata.
    This tests whether metadata improves the simple IR-to-RED mapping.
    """
    def __init__(self, in_ch, out_channels=1, kernel_size=1, n_meta=None, base_ccd="RED5"):
        super().__init__()
        assert isinstance(kernel_size, int), f"kernel_size must be an integer, got {type(kernel_size)}: {kernel_size}"
        assert base_ccd in ["RED5", "RED3"], f"base must be 'RED5' or 'RED3', got {base_ccd}"
        assert in_ch == 2, f"Expected in_ch to be 2 (IR10 + base CCD), got {in_ch} (Can be set in config)"

        self.base_idx = 2 if base_ccd == "RED5" else 5   # Idx of R5/R3 in x_neigh
        self.n_meta = 0 if n_meta is None else int(n_meta)
        self.linear_transform = nn.Conv2d(in_ch + self.n_meta, out_channels, kernel_size, padding=0)

    def forward(self, x_band, x_neigh, x_meta=None, patch_x=None, obs_width=None):
        ir = x_band[:, 0, :, :]                                      # shape => (B, Hb, Wb) - IR10
        base = x_neigh[:, self.base_idx, :, :]                       # shape => (B, Hn, Wn) - R3 or R5

        # isis2np saves context patches with the same top row as the target patch,
        # then extends them downward. The target-aligned crop is therefore the first Hb rows.
        base = base[:, :ir.size(1), :]
        base_row_means = base.mean(dim=2, keepdim=True)    # shape => (B, Hb, 1) - mean of R3/R5 across width for each row
        base_expanded = base_row_means.expand(-1, -1, ir.size(2))  # shape => (B, Hb, Wb)

        cat_list = [
            ir.unsqueeze(1),
            base_expanded.unsqueeze(1),
        ]
        if self.n_meta > 0:
            assert x_meta is not None, "x_meta must be provided when n_meta > 0"
            assert x_meta.ndim == 2, f"x_meta must have shape [B, n_meta], got {x_meta.shape}"
            assert x_meta.size(1) == self.n_meta, f"Expected x_meta.shape[1] == {self.n_meta}, got {x_meta.size(1)}"

            x_meta_img = x_meta[:, :, None, None].expand(-1, -1, ir.size(1), ir.size(2))  # (B, n_meta, H, W)
            cat_list.append(x_meta_img)

        x = torch.cat(cat_list, dim=1)                         # (B, 2 + n_meta, H, W)
        out = self.linear_transform(x)
        return out


# Analytic SYN4 brightness-scaling baseline
class SYN4(nn.Module):
    """
    Scales IR10 so its mean brightness matches a chosen RED reference.
    This is non-trainable and preserves IR10 spatial structure.

    The multiplicative brightness gain MUST be formed in physical DN space.
    Inputs arrive normalized as (DN - center) / scale; in that mean-centered
    space the per-patch mean of IR can pass through zero, making the gain
    ref_mean / ir_mean explode (bipolar saturation, per-image NMAE -> 1e5). We
    therefore denormalize IR/ref to DN with the global (center, scale), form the
    gain on strictly-positive DN brightness means, then renormalize the output so
    the downstream `out * scale + center` denormalization stays valid.

    Requires global_robust normalization: a single (center, scale) applies to
    every sample, so the per-sample affine equals these constants.
    """
    def __init__(self, center, scale, eps=1e-9, base="RED5", r3_idx=5, r5_idx=2):
        super().__init__()
        assert center is not None and scale is not None, (
            "SYN4 requires the global_robust (center, scale); construct via "
            "SYN4.from_global_stats(...) or get_model('SYN4').")
        assert float(scale) > 0, f"SYN4 scale must be positive, got {scale}"
        self.center = float(center)
        self.scale = float(scale)
        self.eps = eps
        self.base = base
        self.r3_idx = r3_idx
        self.r5_idx = r5_idx
        assert base in ["RED5", "RED3", "RED4"], f"base must be 'RED5' or 'RED3' or 'RED4', got {base}"

    @classmethod
    def from_global_stats(cls, device="cpu", stats_path=None, **kwargs):
        """Build SYN4 with the global_robust (center, scale) from global_stats.json.

        Single construction path so every call site denormalizes with the same
        affine the dataset used. Raises if the stats are not global_robust.
        """
        from hipredict.data.normalization import load_global_normalization_stats, _resolve_global_stats_path
        center, scale = load_global_normalization_stats(_resolve_global_stats_path({}, stats_path))
        return cls(center=center, scale=scale, **kwargs).to(device)

    def forward(self, x_band, x_neigh, x_meta, y=None, patch_x=None, obs_width=None):
        # x_neigh is shape (B, 2, H, W, 4)
        B, C, H, W = x_band.shape

        ir = x_band[:, 0, :, :]      # shape => (B, H, W) - IR10 (normalized)

        if self.base == "RED5":
            ref = x_neigh[:, self.r5_idx, :, :]    # shape => (B, H, W) - RED5
        elif self.base == "RED3":
            ref = x_neigh[:, self.r3_idx, :, :]    # shape => (B, H, W) - RED3
        else:   # self.base == "RED4" (Already asserted in __init__)
            assert y is not None, "y (RED4) must be provided when base is 'RED4'"
            ref = y[:, 0, :, :]    # shape => (B, H, W) - RED4

        # Denormalize to DN so the gain denominator is a true (positive) brightness
        # mean instead of a mean-centered value that can cross zero.
        ir_dn = ir * self.scale + self.center
        ref_dn = ref * self.scale + self.center

        ir_mean = ir_dn.mean(dim=(1, 2), keepdim=True)      # mean IR10 brightness (DN)
        ref_mean = ref_dn.mean(dim=(1, 2), keepdim=True)    # mean REDx brightness (DN)

        scale_factor = ref_mean / (ir_mean + self.eps)
        out_dn = ir_dn * scale_factor

        # Renormalize so downstream `out * scale + center` recovers out_dn.
        out = (out_dn - self.center) / self.scale
        return out.unsqueeze(1)        # shape => (B, 1, H, W)


def stage_hyperparams(depth: int, n_convs: int, p_drop: float, n_res_blocks: int):
    enc_cfg = [dict(n_res_blocks=n_res_blocks, n_convs=n_convs, p_drop=0.0) for _ in range(depth)]
    bottleneck_cfg = dict(n_res_blocks=n_res_blocks, n_convs=n_convs, p_drop=min(p_drop, 0.2))
    return enc_cfg, bottleneck_cfg


if __name__ == "__main__":
    # Test the model
    model = SYN4.from_global_stats({})
    batch = torch.randn(32, 256, 1024, 2)  # Example input (batch_size, height, width, channels)
    adjusted = model(batch)
    print("Adjusted shape:", adjusted.shape)  # Should be (32, 256, 1024, 1)
