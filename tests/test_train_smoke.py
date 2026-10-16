import torch

from rapidstamp_train.models import SealDetTiny, SealRecTiny
from rapidstamp_train.train import detection_loss, recognition_loss


def test_detector_loss_is_finite_and_decreases():
    torch.manual_seed(0)
    model = SealDetTiny()
    images = torch.rand(4, 3, 128, 128)
    boxes = [[(40.0, 30.0, 50.0, 50.0)], [], [(20.0, 20.0, 40.0, 36.0)], [(70.0, 60.0, 30.0, 30.0)]]
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    first = None
    last = None
    for _ in range(3):
        opt.zero_grad()
        loss = detection_loss(model(images), boxes)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert last < first


def test_recognizer_loss_is_finite():
    torch.manual_seed(0)
    model = SealRecTiny()
    images = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([0, 3, 5, 8])
    loss = recognition_loss(model(images), labels)
    assert torch.isfinite(loss)
    loss.backward()
