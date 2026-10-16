import io
import json
import zipfile
from pathlib import Path

import torch

from rapidstamp_train.models import SealRec, adapt_recognizer
from rapidstamp_train.realdata import convert_rest_zip, guess_seal, read_charset_file, write_charset_file


def test_guess_seal_keywords_and_aspect():
    assert guess_seal("全国统一发票监制章", 400, 220)[0] == "invoice"
    assert guess_seal("某某财务专用章", 400, 220)[0] == "finance"
    assert guess_seal("上海市公证处", 300, 300)[0] == "notary"
    assert guess_seal("贵州纯玩旅行社有限公司", 420, 240)[1] == "ellipse"
    assert guess_seal("北京华信标研检测技术服务有限公司", 300, 300) == ("round_official", "circle")


def test_convert_rest_zip_one_crop(tmp_path: Path):
    jpg = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (96, 96), (180, 40, 40)).save(jpg, format="JPEG")
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as nested:
        nested.writestr("train_images/7.jpg", jpg.getvalue())
    outer_path = tmp_path / "rest.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr("train/train_images.zip", inner.getvalue())
        outer.writestr(
            "train/train_gts.json",
            json.dumps({"gt_7": [{"points": [[8, 8], [80, 8], [80, 40], [8, 40]], "transcription": "测试公司"}]}),
        )
    dst = tmp_path / "out"
    stats = convert_rest_zip(outer_path, dst, seed=1)
    assert stats["train"] + stats["val"] == 1
    labels = list((dst / "det").rglob("*.json"))
    assert len(labels) == 1
    meta = json.loads(labels[0].read_text(encoding="utf-8"))
    assert meta["source"] == "rest"
    assert meta["seals"][0]["text"] == "测试公司"
    tsv = list((dst / "rec").rglob("labels.tsv"))
    assert "测试公司" in tsv[0].read_text(encoding="utf-8")


def test_charset_roundtrip_keeps_space(tmp_path: Path):
    path = tmp_path / "charset.txt"
    write_charset_file(path, ["A", " ", "字"])
    assert read_charset_file(path) == ["A", " ", "字"]


def test_adapt_recognizer_keeps_old_rows():
    old_chars = ["甲", "乙"]
    new_chars = ["甲", "丙", "乙"]
    old = SealRec(1 + len(old_chars))
    with torch.no_grad():
        old.fc.weight.zero_()
        old.fc.bias.zero_()
        old.fc.weight[1].fill_(1.5)
        old.fc.weight[2].fill_(2.5)
        old.fc.bias[2] = 0.25
    state = {"model": old.state_dict(), "charset": old_chars}
    new = adapt_recognizer(state, new_chars)
    assert torch.allclose(new.fc.weight[1], old.fc.weight[1])
    assert torch.allclose(new.fc.weight[3], old.fc.weight[2])
    assert new.fc.bias[3].detach().item() == 0.25
    assert new.fc.weight[2].detach().abs().sum().item() == 0.0
