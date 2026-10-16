# rapidstamp

CPU seal detection and recognition. Training lives in the repository's `training/` package; this wheel is the inference API.

```python
from rapidstamp import SealEngine

engine = SealEngine.from_pretrained(backend="numpy")  # or "native" / "onnx"
for seal in engine.infer(rgb_uint8):
    print(seal.score, seal.text, seal.x, seal.y, seal.w, seal.h)
```

`backend="native"` loads `librapidstamp` (build `cpp/` or set `RAPIDSTAMP_LIB`). `backend="onnx"` uses the sibling `.onnx` files through ONNX Runtime. The numpy backend is the reference and has no extra dependencies beyond NumPy.

See the repository `README.md` and `docs/research.md` for the model choice, datasets, and the C API.
