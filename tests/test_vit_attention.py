"""Architecture-only ViT tests; these do not validate missing T26 weights."""
import torch

from src.modules.vit_attention import ViTLungAttention


def test_vit_average_pool_baseline_matches_timm_head():
    model = ViTLungAttention(use_attention=False, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits, attention, attention_logits = model(x)
        direct = model.backbone(x)
    assert logits.shape == (1, 4)
    assert attention is None and attention_logits is None
    torch.testing.assert_close(logits, direct, atol=1e-5, rtol=1e-5)


def test_vit_lung_gate_has_patch_grid_and_classifier_path():
    model = ViTLungAttention(use_attention=True, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits, attention, attention_logits = model(x)
    assert logits.shape == (1, 4)
    assert attention.shape == (1, 1, 14, 14)
    assert attention_logits.shape == attention.shape
    assert torch.isfinite(logits).all()
