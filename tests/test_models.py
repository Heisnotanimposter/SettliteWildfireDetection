import torch
from wildfire_detection.models.unet import UNet
from wildfire_detection.models.faster_rcnn import build_faster_rcnn


def test_unet_forward_shape():
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_faster_rcnn_instantiation():
    model = build_faster_rcnn(num_classes=2, pretrained=False)
    assert model is not None
    images = [torch.randn(3, 128, 128)]
    targets = [{"boxes": torch.tensor([[10., 10., 30., 30.]]), "labels": torch.tensor([1])}]
    model.train()
    loss_dict = model(images, targets)
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict
