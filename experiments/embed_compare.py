"""Embedder comparison: X-CLIP vs Qwen3-VL-Embedding-2B vs SigLIP2.

Single-file, local-only experiment. No DB, no vector store.
Gallery: 1 Titanic target (.webp from user) + 8 curated distractor photos.
Metrics: cosine similarity table, rank of target, margin pos-neg.

Run locally:  python experiments/embed_compare.py
Run on Colab: colab run --gpu T4 experiments/embed_compare.py
"""
import io
import json
import os
import subprocess
import sys
import urllib.request

TARGET_URL = "https://ik.imagekit.io/annasblackhat/project/footage-engine/even-at-the-bottom-of-the-ocean-she-is-still-majestic-v0-0tm5g0ubz74b1.webp"

# Curated distractors: fixed picsum seeds -> reproducible real photos,
# semantically far from a deep-ocean shipwreck (none are wrecks).
DISTRACTORS = {
    "neg_beach": "https://picsum.photos/seed/titanic-exp-beach/640/360",
    "neg_city": "https://picsum.photos/seed/titanic-exp-city/640/360",
    "neg_forest": "https://picsum.photos/seed/titanic-exp-forest/640/360",
    "neg_desert": "https://picsum.photos/seed/titanic-exp-desert/640/360",
    "neg_food": "https://picsum.photos/seed/titanic-exp-food/640/360",
    "neg_portrait": "https://picsum.photos/seed/titanic-exp-portrait/640/360",
    "neg_cat": "https://picsum.photos/seed/titanic-exp-cat/640/360",
    "neg_mountain": "https://picsum.photos/seed/titanic-exp-mountain/640/360",
}

POS_QUERIES = [
    "ship sinking deep ocean",
    "bow of Titanic wreck on ocean floor",
    "rusted shipwreck bow in deep abyss",
    "The bow section of the RMS Titanic resting on the dark ocean floor, "
    "rusted hull with anchor housing and railings, deep indigo abyssal tones, seabed sediment",
]
NEG_QUERIES = [
    "sunny beach",
    "cat sleeping on sofa",
    "city street at night",
]


def ensure_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import sentence_transformers  # noqa: F401
        import PIL  # noqa: F401
        return
    except ImportError:
        pass
    print("[setup] installing torch/transformers/sentence-transformers ...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "-q", "install",
         "torch", "transformers>=4.57", "sentence-transformers",
         "pillow", "numpy", "requests"]
    )


def download(url: str, dest: str, timeout: int = 60) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "footage-engine-exp/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def l2norm(vecs):
    import numpy as np
    v = np.asarray(vecs, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


def cosine(a, b):
    import numpy as np
    return float(np.dot(np.asarray(a, float), np.asarray(b, float)))


class XclipWrapper:
    name = "xclip-base-patch32"
    dim = 512

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[xclip] loading microsoft/xclip-base-patch32 on {self.device} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(self.device)
        self.model.eval()

    def _feat(self, out):
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0, :]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def embed_image(self, pil_img):
        from PIL import Image
        torch = self.torch
        img = pil_img.convert("RGB")
        frames = [img] * 8  # same fake-video protocol as footage_engine/embeddings/xclip.py
        inputs = self.processor(videos=[frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = self._feat(self.model.get_video_features(**inputs))
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().tolist()

    def embed_text(self, text):
        torch = self.torch
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = self._feat(self.model.get_text_features(**inputs))
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().tolist()


class Siglip2Wrapper:
    name = "siglip2-base-patch16-512"
    dim = 512

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-base-patch16-512"
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[siglip2] loading {mid} on {self.device} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid).to(self.device)
        self.model.eval()

    def _pool(self, out):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if v is not None:
                    return v
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0, :]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def embed_image(self, pil_img):
        torch = self.torch
        inputs = self.processor(images=pil_img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            feat = self._pool(self.model.get_image_features(**inputs))
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().tolist()

    def embed_text(self, text):
        torch = self.torch
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            feat = self._pool(self.model.get_text_features(**inputs))
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().tolist()


class QwenWrapper:
    name = "Qwen3-VL-Embedding-2B"
    dim = 2048

    def __init__(self, device="cuda"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[qwen] loading Qwen/Qwen3-VL-Embedding-2B on {self.device} ...", flush=True)
        self.model = SentenceTransformer(
            "Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True,
            device=self.device, model_kwargs={"dtype": "float16" if self.device == "cuda" else "float32"},
        )
        # probe calling conventions once
        self._img_mode = self._probe_image_mode()

    def _probe_image_mode(self):
        from PIL import Image
        tiny = Image.new("RGB", (32, 32), color=(10, 20, 30))
        for mode, fn in [
            ("images_kw", lambda: self.model.encode(images=[tiny])),
            ("list_pil", lambda: self.model.encode([tiny])),
            ("dict_image", lambda: self.model.encode([{"image": tiny}])),
        ]:
            try:
                out = fn()
                if out is not None and len(out) == 1:
                    print(f"[qwen] image encode mode: {mode}", flush=True)
                    return mode
            except Exception as e:
                print(f"[qwen] mode {mode} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        raise RuntimeError("No working Qwen image encode convention found")

    def embed_image(self, pil_img):
        img = pil_img.convert("RGB")
        if self._img_mode == "images_kw":
            out = self.model.encode(images=[img], convert_to_numpy=True, normalize_embeddings=True)
        elif self._img_mode == "dict_image":
            out = self.model.encode([{"image": img}], convert_to_numpy=True, normalize_embeddings=True)
        else:
            out = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in out[0]]

    def embed_text(self, text):
        out = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                                prompt_name="query" if "query" in (self.model.prompts or {}) else None)
        return [float(x) for x in out[0]]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="xclip,siglip2,qwen",
                    help="comma subset of xclip,siglip2,qwen")
    args, _unknown = ap.parse_known_args()
    wanted = {m.strip().lower() for m in args.models.split(",")}

    ensure_deps()
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} cuda={torch.cuda.is_available()} "
          f"name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}", flush=True)

    gdir = "/content/gallery" if os.path.isdir("/content") else "./data/exp_gallery"
    paths = {"target_titanic": download(TARGET_URL, os.path.join(gdir, "titanic.webp"))}
    for key, url in DISTRACTORS.items():
        paths[key] = download(url, os.path.join(gdir, f"{key}.jpg"))
    print(f"[data] gallery: {len(paths)} images in {gdir}", flush=True)

    images = {}
    for key, p in paths.items():
        with Image.open(p) as im:
            images[key] = im.convert("RGB").copy()

    registry = {"xclip": XclipWrapper, "siglip2": Siglip2Wrapper, "qwen": QwenWrapper}
    models = []
    for key in ("xclip", "siglip2", "qwen"):
        if key not in wanted:
            continue
        cls = registry[key]
        try:
            print(f"[load] {key} starting ...", flush=True)
            models.append(cls(device=device))
            print(f"[load] {key} ready.", flush=True)
        except Exception as e:
            print(f"[{cls.__name__}] SKIPPED: {type(e).__name__}: {e}", flush=True)

    if not models:
        print("No models loaded, aborting.", flush=True)
        sys.exit(2)

    queries = [("POS", q) for q in POS_QUERIES] + [("NEG", q) for q in NEG_QUERIES]
    report = {"gallery": list(paths.keys()), "results": {}}
    for m in models:
        try:
            print(f"\n===== {m.name} =====", flush=True)
            img_vecs = {k: l2norm(m.embed_image(im)) for k, im in images.items()}
            rows = []
            for kind, q in queries:
                qv = l2norm(m.embed_text(q))
                sims = {k: cosine(qv, v) for k, v in img_vecs.items()}
                ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
                rank = next(i for i, (k, _) in enumerate(ranked, 1) if k == "target_titanic")
                rows.append({"query": q, "kind": kind, "rank_target": rank, "sims": sims})
                tag = "POS" if kind == "POS" else "neg"
                print(f"[{tag}] rank={rank}/9 sim_target={sims['target_titanic']:.3f} | {q[:70]}", flush=True)
                detail = ", ".join(f"{k}={v:.3f}" for k, v in ranked)
                print(f"      {detail}", flush=True)
            pos = [r["sims"]["target_titanic"] for r in rows if r["kind"] == "POS"]
            neg = [r["sims"]["target_titanic"] for r in rows if r["kind"] == "NEG"]
            import numpy as np
            margin = float(np.mean(pos) - np.mean(neg)) if pos and neg else 0.0
            print(f"[{m.name}] mean_pos={np.mean(pos):.3f} mean_neg={np.mean(neg):.3f} margin={margin:.3f}", flush=True)
            report["results"][m.name] = {"rows": rows, "margin": margin}
        except Exception as e:
            import traceback
            print(f"[{m.name}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    out = "results_embed_compare.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] wrote {os.path.abspath(out)}", flush=True)


if __name__ == "__main__":
    main()
