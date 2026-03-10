"""UniTable extraction via RunPod Flash — runs on remote GPU, called locally."""

import asyncio
import base64
import io
import re
import warnings
from pathlib import Path
from functools import partial

from runpod_flash import Endpoint, GpuType


@Endpoint(
    name="unitable-extract",
    gpu=GpuType.NVIDIA_L4,
    workers=(0, 1),
    idle_timeout=300,
    dependencies=[
        "torch",
        "torchvision",
        "tokenizers",
        "Pillow",
        "huggingface_hub",
    ],
    system_dependencies=["git"],
)
async def extract_table(data: dict) -> dict:
    """Runs on RunPod GPU. Accepts base64 image, returns HTML table."""
    import os
    import re
    import subprocess
    import sys
    import warnings
    from pathlib import Path
    from functools import partial

    import torch
    from torch import nn
    import tokenizers as tk
    from PIL import Image
    from torchvision import transforms
    from huggingface_hub import hf_hub_download

    warnings.filterwarnings("ignore")

    # --- Setup: clone repo + download weights (cached after first run) ---
    REPO_DIR = Path("/tmp/unitable")
    WEIGHTS_DIR = REPO_DIR / "experiments" / "unitable_weights"
    VOCAB_DIR = REPO_DIR / "vocab"

    if not REPO_DIR.exists():
        subprocess.check_call(
            ["git", "clone", "-q", "https://github.com/poloclub/unitable.git", str(REPO_DIR)]
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for wf in ["unitable_large_structure.pt", "unitable_large_bbox.pt", "unitable_large_content.pt"]:
        if not (WEIGHTS_DIR / wf).exists():
            hf_hub_download(repo_id="poloclub/UniTable", filename=wf, local_dir=str(WEIGHTS_DIR))

    # Import UniTable source
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
    from src.utils import (
        subsequent_mask, pred_token_within_range, greedy_sampling,
        bbox_str_to_token_list, cell_str_to_token_list, html_str_to_token_list,
        build_table_from_html_and_cell, html_table_template,
    )
    from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN, INVALID_CELL_TOKEN

    # --- Load models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, patch_size, nhead, dropout = 768, 16, 12, 0.2

    def load_model(vocab_path, max_seq_len, weights_path):
        vocab = tk.Tokenizer.from_file(str(vocab_path))
        backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
        encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout,
                          activation="gelu", norm_first=True, nlayer=12, ff_ratio=4)
        decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout,
                          activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)
        model = EncoderDecoder(
            backbone=backbone, encoder=encoder, decoder=decoder,
            vocab_size=vocab.get_vocab_size(), d_model=d_model,
            padding_idx=vocab.token_to_id("<pad>"), max_seq_len=max_seq_len,
            dropout=dropout, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        model = model.to(device).eval()
        return vocab, model

    structure_vocab, structure_model = load_model(VOCAB_DIR / "vocab_html.json", 784, WEIGHTS_DIR / "unitable_large_structure.pt")
    bbox_vocab, bbox_model = load_model(VOCAB_DIR / "vocab_bbox.json", 1024, WEIGHTS_DIR / "unitable_large_bbox.pt")
    content_vocab, content_model = load_model(VOCAB_DIR / "vocab_cell_6k.json", 200, WEIGHTS_DIR / "unitable_large_content.pt")

    # --- Helpers ---
    def to_tensor(image, size):
        T = transforms.Compose([
            transforms.Resize(size), transforms.ToTensor(),
            transforms.Normalize([0.86597056, 0.88463002, 0.87491087],
                                 [0.20686628, 0.18201602, 0.18485524])])
        return T(image).to(device).unsqueeze(0)

    def decode(model, img_tensor, prefix, max_len, eos_id, whitelist=None, blacklist=None):
        with torch.no_grad():
            memory = model.encode(img_tensor)
            ctx = torch.tensor(prefix, dtype=torch.int32).repeat(img_tensor.shape[0], 1).to(device)
        for _ in range(max_len):
            if all(eos_id in k for k in ctx):
                break
            with torch.no_grad():
                mask = subsequent_mask(ctx.shape[1]).to(device)
                logits = model.generator(model.decode(memory, ctx, tgt_mask=mask, tgt_padding_mask=None))[:, -1, :]
            logits = pred_token_within_range(logits.detach(), white_list=whitelist, black_list=blacklist)
            _, next_tok = greedy_sampling(logits)
            ctx = torch.cat([ctx, next_tok], dim=1)
        return ctx

    # --- Decode image ---
    img_bytes = base64.b64decode(data["image_base64"])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_size = image.size

    # Stage 1: Structure
    t = to_tensor(image, (448, 448))
    pred = decode(structure_model, t,
        [structure_vocab.token_to_id("[html]")], 512,
        structure_vocab.token_to_id("<eos>"),
        [structure_vocab.token_to_id(i) for i in VALID_HTML_TOKEN])
    pred_html = html_str_to_token_list(
        structure_vocab.decode(pred.cpu().numpy()[0], skip_special_tokens=False))

    # Stage 2: Bbox
    t = to_tensor(image, (448, 448))
    pred = decode(bbox_model, t,
        [bbox_vocab.token_to_id("[bbox]")], 1024,
        bbox_vocab.token_to_id("<eos>"),
        [bbox_vocab.token_to_id(i) for i in VALID_BBOX_TOKEN[:449]])
    pred_bbox = bbox_str_to_token_list(
        bbox_vocab.decode(pred.cpu().numpy()[0], skip_special_tokens=False))
    ratio = [img_size[0] / 448, img_size[1] / 448] * 2
    pred_bbox = [[int(round(c * r)) for c, r in zip(b, ratio)] for b in pred_bbox]

    # Stage 3: Content
    if pred_bbox:
        cell_tensors = torch.cat([to_tensor(image.crop(b), (112, 448)) for b in pred_bbox], dim=0)
        pred = decode(content_model, cell_tensors,
            [content_vocab.token_to_id("[cell]")], 200,
            content_vocab.token_to_id("<eos>"),
            blacklist=[content_vocab.token_to_id(i) for i in INVALID_CELL_TOKEN])
        pred_cell = [cell_str_to_token_list(c) for c in
            content_vocab.decode_batch(pred.cpu().numpy(), skip_special_tokens=False)]
        pred_cell = [re.sub(r'(\d).\s+(\d)', r'\1.\2', c) for c in pred_cell]
    else:
        pred_cell = []

    # Combine
    code = build_table_from_html_and_cell(pred_html, pred_cell)
    html = html_table_template("".join(code))

    return {"html": html, "device": str(device)}


# --- Local helper to call the endpoint ---
def extract_html_remote(image: "Image.Image") -> str:
    """Send a PIL image to RunPod, get back HTML. Call from local code."""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    result = asyncio.run(extract_table({"image_base64": img_b64}))
    return result["html"]
