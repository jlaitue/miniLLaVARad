"""BiomedCLIP-CXR vision trunk. Weights come from huggingface.co/microsoft/llava-rad."""

import timm
import torch
import torch.nn as nn

NUM_PATCHES = 1369  # 37 x 37 at 518 pixels with patch size 14
EMBED_DIM = 768
TRUNK_PREFIX = "module.visual.trunk."


class VisionEncoder(nn.Module):
    """Returns patch tokens from the penultimate block, as LLaVA-Rad does (select_layer=-2)."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.trunk = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m", pretrained=False, img_size=518, num_classes=0, global_pool=""
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        trunk_state = {k[len(TRUNK_PREFIX):]: v for k, v in state.items() if k.startswith(TRUNK_PREFIX)}
        self.trunk.load_state_dict(trunk_state, strict=True)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images):
        tokens = self.trunk.patch_embed(images)
        tokens = self.trunk._pos_embed(tokens)
        tokens = self.trunk.norm_pre(tokens)
        for block in self.trunk.blocks[:-1]:
            tokens = block(tokens)
        return tokens[:, 1:]
