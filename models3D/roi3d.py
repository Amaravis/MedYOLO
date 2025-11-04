# models/roi3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _reorder_boxes_for_theta(labels: torch.Tensor, box_format: str) -> torch.Tensor:
    assert labels.dim() == 2 and labels.size(-1) in (6, 8), "labels should be (R,6) or (R,8)"

    
    boxes = labels[:, 2:8] if labels.size(-1) == 8 else labels  # [z, x, y, d, w, h] or already (R,6)

    if box_format == "zyx_dhw":      
        return boxes
    elif box_format == "zxy_dwh":
        zc, xc, yc, d, w, h = boxes.unbind(dim=1)
        return torch.stack([zc, yc, xc, d, h, w], dim=1)
    else:
        raise ValueError(f"Unknown box_format: {box_format}")


def _boxes_any_to_theta_3d(boxes_norm_for_theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    zc, yc, xc, d, h, w = boxes_norm_for_theta.unbind(dim=1)

    
    w = w.clamp(min=eps, max=1.0)
    h = h.clamp(min=eps, max=1.0)
    d = d.clamp(min=eps, max=1.0)
    xc = xc.clamp(0.0, 1.0)
    yc = yc.clamp(0.0, 1.0)
    zc = zc.clamp(0.0, 1.0)

   
    tx = 2.0 * xc - 1.0
    ty = 2.0 * yc - 1.0
    tz = 2.0 * zc - 1.0

    zero = torch.zeros_like(w)
    theta = torch.stack([
        torch.stack([w,   zero, zero, tx], dim=1),  # x-row (width fraction)
        torch.stack([zero, h,   zero, ty], dim=1),  # y-row (height fraction)
        torch.stack([zero, zero, d,   tz], dim=1),  # z-row (depth fraction)
    ], dim=1)  # (R,3,4)
    return theta


class ROICrop3D(nn.Module):
    """
    Differentiable 3D crop-and-resize using affine_grid + grid_sample.

    Args:
      out_size: (D_out, H_out, W_out)
      box_format:
         'zyx_dhw' -> boxes are (zc, yc, xc, d, h, w)
         'zxy_dwh' -> boxes are (zc, xc, yc, d, w, h)  [MedYOLO]
    """
    def __init__(self, out_size=(16, 32, 32), mode='bilinear', padding_mode='zeros',
                 align_corners=True, box_format='zxy_dwh'):
        super().__init__()
        self.out_size = tuple(out_size)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.box_format = box_format  # <<< set this to 'zxy_dwh' for MedYOLO

    def forward(self, feats: torch.Tensor, boxes):
        """
        feats:  (B, C, D, H, W)
        boxes:  list of length B with (Ri,6) OR tensor (B,R,6)
                in the format specified by self.box_format.
        Returns:
          patches:   (sum_R, C, D_out, H_out, W_out)
          index_map: (sum_R, 2) long -> (b_ix, r_ix)
        """
        B, C, D, H, W = feats.shape

        # normalize input to list-of-tensors
        if isinstance(boxes, torch.Tensor):
            assert boxes.dim() == 3 and boxes.size(0) == B and boxes.size(-1) == 6
            boxes_list = [boxes[b] for b in range(B)]
        else:
            boxes_list = boxes
            assert len(boxes_list) == B

        patches, index_map = [], []
        for b, boxes_b in enumerate(boxes_list):
            if boxes_b.numel() == 0:
                continue

            # Reorder MedYOLO format -> theta format
            boxes_theta = _reorder_boxes_for_theta(boxes_b, self.box_format)   # -> (zc,yc,xc,d,h,w)
            theta = _boxes_any_to_theta_3d(boxes_theta)                        # (Rb,3,4)

            # grid & sample
            grid = F.affine_grid(theta, size=(boxes_b.size(0), C, *self.out_size),
                                 align_corners=self.align_corners)
            x_b = feats[b:b+1].expand(boxes_b.size(0), -1, -1, -1, -1)
            patch_b = F.grid_sample(
                x_b, grid, mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners
            )  # (Rb, C, D_out, H_out, W_out)

            patches.append(patch_b)
            index_map.append(
                torch.stack([
                    torch.full((boxes_b.size(0),), b, dtype=torch.long, device=feats.device),
                    torch.arange(boxes_b.size(0), device=feats.device, dtype=torch.long)
                ], dim=1)
            )

        if len(patches) == 0:
            return feats.new_zeros((0, C, *self.out_size)), feats.new_zeros((0, 2), dtype=torch.long)

        patches = torch.cat(patches, dim=0)
        index_map = torch.cat(index_map, dim=0)
        return patches, index_map


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, embed_dim=256, patch=(2,2,2), stride=None):
        super().__init__()
        if stride is None: stride = patch
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=stride, padding=0, bias=True)
    def forward(self, x):               # x: (N,C,D,H,W)
        x = self.proj(x)                # (N,E,d',h',w')
        B, E, Dp, Hp, Wp = x.shape
        return x.flatten(2).transpose(1, 2)   # (N, Npatch, E)

class ROIEmbedHead3D(nn.Module):
    """
    - Crops 3D patches around GT boxes (normalized) from one or more tapped feature maps.
    - Converts each patch to tokens via PatchEmbed3D, pools to one vector, fuses across scales, projects.
    """
    def __init__(self, in_channels, crop_size=(16,32,32), embed_dim=256, patch=(2,2,2),
                 proj_dim=256, pooling='mean', padding_mode='zeros'):
        super().__init__()
        if isinstance(in_channels, int): in_channels = [in_channels]
        self.cropper = ROICrop3D(out_size=crop_size, padding_mode=padding_mode, align_corners=True)
        self.patch_embeds = nn.ModuleList([PatchEmbed3D(c, embed_dim, patch) for c in in_channels])
        self.proj = nn.Linear(embed_dim * len(in_channels), proj_dim)
        self.pooling = pooling

    @torch.no_grad()
    def _check_sizes(self, feats):
        # Optional sanity check: all feats share B and are 5D
        B = feats[0].shape[0]
        for f in feats: assert f.dim()==5 and f.shape[0]==B

    def forward(self, feats, boxes):
        """
        feats:  list of tapped features, each (B,C,D,H,W)
        boxes:  list[Tensor(Ri,6)] or Tensor(B,R,6) — normalized (zc,yc,xc,d,h,w)
        returns:
          emb: (sum_i Ri, proj_dim)  — one embedding per ROI
          idx: (sum_i Ri, 2)         — (b_ix, r_ix) mapping for bookkeeping
        """
        self._check_sizes(feats)
        # Crop on each scale using the SAME normalized boxes (works because grid coords are normalized)
        patches_per_scale = []
        idx_ref = None
        for f in feats:
            patches, idx = self.cropper(f, boxes)     # (sum_R, C, Dout,Hout,Wout)
            patches_per_scale.append(patches)
            if idx_ref is None:
                idx_ref = idx
            else:
                # defensively ensure the same ROI set/order
                assert torch.equal(idx, idx_ref), "ROI order mismatch across scales"

        # Patch → tokens → pool per ROI for each scale
        pooled_per_scale = []
        for pe, p in zip(self.patch_embeds, patches_per_scale):
            tok = pe(p)                         # (sum_R, Npatch, E)
            if self.pooling == 'mean':
                v = tok.mean(dim=1)             # (sum_R, E)
            elif self.pooling == 'max':
                v = tok.max(dim=1).values
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
            pooled_per_scale.append(v)

        # Fuse scales and project
        z = torch.cat(pooled_per_scale, dim=1)   # (sum_R, E * n_scales)
        z = self.proj(z)                          # (sum_R, proj_dim)
        z = F.normalize(z, dim=1)                 # handy for cosine/contrastive losses
        return z, idx_ref

