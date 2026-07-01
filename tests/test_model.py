import torch

from rcsgcn.model import RCSGCN


def test_model_forward_shape_on_synthetic_input():
    torch.manual_seed(42)
    model = RCSGCN().eval()
    data = torch.randn(2, 12, 16, 25, 1)
    with torch.no_grad():
        logits = model(data)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()

