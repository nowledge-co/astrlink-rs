"""Generate synthetic, independent AstrLink reference fixtures.

Run with uv run --no-project --with onnxruntime==1.23.2 --with tokenizers==0.23.3 python -I tools/oracle.py MODEL_DIR OUTPUT
ORT is a fixture tool, not a library dependency.
"""
import hashlib
import importlib.metadata
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

root, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((pathlib.Path(__file__).resolve().parents[1] / "model.json").read_text())
assets = {}
for role in ("model", "data", "tokenizer"):
    pin = manifest[role]
    path = root / pin["file"]
    with path.open("rb") as asset:
        data = asset.read(pin["size"] + 1)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != pin["size"] or digest != pin["sha256"]:
        raise SystemExit(f"{path} differs from model.json; download the pinned assets before generating fixtures")
    assets[pin["file"]] = digest
tokenizer = Tokenizer.from_file(str(root / manifest["tokenizer"]["file"]))
tokenizer.no_truncation()
tokenizer.no_padding()
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(root / manifest["model"]["file"]), options, providers=["CPUExecutionProvider"])
texts = [
    "Contact Alice Smith at alice@example.com.",
    "联系人：张三，邮箱 zhangsan@example.com。",
    "連絡先は田中太郎、メール tanaka@example.com です。",
    "담당자 김민수, 이메일 minsu@example.com 입니다.",
    "  café é ＡＢＣ ① 👩🏽‍💻\n\tadmin@example.org\r\n",
    "<s>hi</s> <mask> x <mask>",
    "",
    "hello " * 255,
]
cases = []
for text in texts:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [0, *encoded.ids, 2]
    offsets = [[len(text[:a].encode()), len(text[:b].encode())] for a, b in encoded.offsets]
    feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64), "token_type_ids": np.zeros((1, len(ids)), dtype=np.int64)}
    feed = {item.name: feed[item.name] for item in session.get_inputs()}
    logits = session.run(None, feed)[0][0]
    cases.append({"text": text, "ids": encoded.ids, "offsets": offsets, "logits": logits.tolist()})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"oracle": ort.__version__, "tokenizer": importlib.metadata.version("tokenizers"), "assets": assets, "cases": cases}, ensure_ascii=False, indent=2) + "\n")
print(f"Generated {len(cases)} synthetic reference cases: {output}")
