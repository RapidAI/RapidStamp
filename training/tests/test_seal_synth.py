import random
from pathlib import Path

import numpy as np
import pytest

from rapidstamp_train.direct_aug import Charset
from rapidstamp_train.name_corpus import (
    accept_name,
    clean_line,
    flatten_places,
    generate_names,
    uniform_strings,
)
from rapidstamp_train.seal_render import FontBank, render_seal, ring_angles

FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
CHARS = list("杭州市西湖区市场监督管理局有限公司科技华新东城北京市人民政府生态环境")


def _charset() -> Charset:
    return Charset(CHARS)


def test_clean_line_drops_the_corpus_header():
    assert clean_line("\ufeffBy@萌名NameMoe") == ""
    assert clean_line("2022.11.30") == ""
    assert clean_line("  杭州市市场监督管理局 ") == "杭州市市场监督管理局"


def test_accept_rejects_blocked_templates_and_long_strings():
    charset = _charset()
    blocked = {"杭州市西湖区市场监督管理局"}
    assert not accept_name("杭州市西湖区市场监督管理局", charset, blocked)
    assert not accept_name("某某有限公司", charset, blocked)
    assert not accept_name("甲" * 26, charset, blocked)
    assert not accept_name("AB有限公司", charset, blocked)
    assert accept_name("杭州市市场监督管理局", charset, blocked)


def test_uniform_strings_cover_every_character_and_skip_blocked():
    charset = Charset(["甲", "乙", "丙", "丁"])
    blocked = set()
    rows = uniform_strings(charset, blocked, per_char=2, rng=random.Random(0))
    for char in "甲乙丙丁":
        assert sum(char in row for row in rows) >= 2
    assert all(row not in blocked for row in rows)


def test_generated_agencies_stay_inside_the_charset():
    charset = Charset(list("北京市东城区市场监督管理局公安局人民政府西湖"))
    places = flatten_places(
        [
            {
                "name": "北京市",
                "children": [
                    {"name": "市辖区", "children": [{"name": "东城区"}]},
                ],
            }
        ]
    )
    assert "东城区" in places
    assert "市辖区" not in places
    _companies, agencies = generate_names(random.Random(1), places, charset, set(), 0, 5)
    assert len(agencies) == 5
    assert all(accept_name(name, charset, set()) for name in agencies)


def test_ring_arc_is_centered_on_the_top_and_open_at_the_bottom():
    phi0, span = ring_angles(12, random.Random(0))
    mid = (phi0 + span / 2.0) % 360.0
    assert mid == pytest.approx(270.0)
    assert (90.0 - phi0) % 360.0 >= span
    assert (270.0 - phi0) % 360.0 < span


@pytest.mark.skipif(not FONT.is_file(), reason="wenquanyi font is not installed")
def test_render_is_a_square_red_seal_and_repeats():
    fonts = FontBank([FONT])
    rng = random.Random(4)
    image = render_seal("杭州市市场监督管理局", rng, fonts)
    assert image.mode == "RGB"
    assert image.size[0] == image.size[1]
    arr = np.asarray(image)
    red = ((arr[:, :, 0] > arr[:, :, 1] + 20) & (arr[:, :, 0] > 100)).sum()
    assert red > 200
    again = np.asarray(render_seal("杭州市市场监督管理局", random.Random(4), fonts))
    assert int(arr.sum()) == int(again.sum())
