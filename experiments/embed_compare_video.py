"""Video embedder comparison: X-CLIP vs Qwen3-VL-2B vs SigLIP2.

Target: 15s cargo-ship sinking video from user.
Method: sample 8 uniform frames -> per-frame embed -> mean-pool to clip vector.
Also: per-frame scores vs positive query (temporal localization sanity check).
Queries from user's Gemini description + negatives.

Run on Colab: colab exec --timeout 3600 -s <sess> -f experiments/embed_compare_video.py
"""
import os
import subprocess
import sys
import urllib.parse
import urllib.request

VIDEO_URL = "https://ik.imagekit.io/annasblackhat/project/footage-engine/Gigantic%20Cargo%20Ship%20Sinks%20Near%20Shore%20While%20Crowds%20Film%20the%20Dramatic%20Moment%20.mp4"

POS_QUERIES = [
    "gigantic cargo ship sinks near shore while crowds film",
    "large container ship named Pacific Trader listing severely to starboard in rough ocean",
    "shipping containers slipping off deck into the ocean, water flooding over starboard side",
    "container vessel angled heavily into the water, red lower hull and bulbous bow exposed, "
    "containers stacked on deck tilt dangerously close to the sea surface",
]
NEG_QUERIES = [
    "sunny beach",
    "cat sleeping on sofa",
    "city street at night",
]
N_FRAMES = 8


def ensure_deps():
    try:
        import torch, transformers, sentence_transformers, PIL, cv2  # noqa: F401
        return
    except ImportError:
        pass
    print("[setup] installing deps ...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "-q", "install",
         "torch", "transformers>=4.57", "sentence-transformers",
         "pillow", "numpy", "opencv-python-headless"]
    )


def download(url, dest, timeout=300):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[data] cached {dest} ({os.path.getsize(dest)} bytes)", flush=True)
        return dest
    print(f"[data] downloading {url[:80]} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "footage-engine-exp/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = 0
        while True:
            blk = r.read(1 << 20)
            if not blk:
                break
            f.write(blk)
            total += len(blk)
        print(f"[data] saved {dest} ({total} bytes)", flush=True)
    return dest


def sample_frames(video_path, num_frames=8):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = total / fps if fps > 0 else 0.0
        print(f"[data] fps={fps:.1f} frames={total} dur={dur:.1f}s", flush=True)
        out = []
        for i in range(num_frames):
            ts = (i + 0.5) * (dur / num_frames)
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append((ts, Image.fromarray(rgb)))
            elif out:
                out.append((ts, out[-1][1]))
        return out
    finally:
        cap.release()


def l2norm(vecs):
    import numpy as np
    v = np.asarray(vecs, dtype=float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def cosine(a, b):
    import numpy as np
    return float(np.dot(np.asarray(a, float), np.asarray(b, float)))


class XclipWrapper:
    name = "xclip-base-patch32"

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[xclip] loading on {self.device} ...", flush=True)
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

    def embed_frames_mean(self, pils):
        torch = self.torch
        vecs = []
        for im in pils:
            frames = [im.convert("RGB")] * 8
            inputs = self.processor(videos=[frames], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = self._feat(self.model.get_video_features(**inputs))
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            vecs.append(feat.squeeze(0).cpu().tolist())
        import numpy as np
        return (l2norm(np.mean(np.asarray(vecs), axis=0))).tolist()

    def embed_image(self, pil_img):
        return self.embed_frames_mean([pil_img])

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

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-base-patch16-512"
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[siglip2] loading on {self.device} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid).to(self.device)
        self.model.eval()

    def _pool(self, out):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            if hasattr(out, attr) and getattr(out, attr) is not None:
                return getattr(out, attr)
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

    def embed_frames_mean(self, pils):
        import numpy as np
        vecs = [self.embed_image(im) for im in pils]
        return (l2norm(np.mean(np.asarray(vecs), axis=0))).tolist()

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

    def __init__(self, device="cuda"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[qwen] loading on {self.device} ...", flush=True)
        self.model = SentenceTransformer(
            "Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True,
            device=self.device, model_kwargs={"dtype": "float16" if self.device == "cuda" else "float32"},
        )

    def embed_image(self, pil_img):
        out = self.model.encode([pil_img.convert("RGB")], convert_to_numpy=True,
                                normalize_embeddings=True)
        return [float(x) for x in out[0]]

    def embed_frames_mean(self, pils):
        import numpy as np
        vecs = self.model.encode([p.convert("RGB") for p in pils], convert_to_numpy=True,
                                 normalize_embeddings=True, batch_size=4)
        return (l2norm(np.mean(np.asarray(vecs), axis=0))).tolist()

    def embed_text(self, text):
        out = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                                prompt_name="query" if "query" in (self.model.prompts or {}) else None)
        return [float(x) for x in out[0]]


def main():
    ensure_deps()
    import torch
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}",
          flush=True)

    gdir = "/content/video_exp" if os.path.isdir("/content") else "./data/video_exp"
    vpath = download(VIDEO_URL, os.path.join(gdir, "cargo_ship_sinking.mp4"))
    stamped = sample_frames(vpath, N_FRAMES)
    print(f"[data] sampled {len(stamped)} frames", flush=True)
    pils = [im for _, im in stamped]

    models = []
    for cls in (XclipWrapper, Siglip2Wrapper, QwenWrapper):
        try:
            print(f"[load] {cls.__name__} starting ...", flush=True)
            models.append(cls(device=device))
            print(f"[load] {cls.__name__} ready.", flush=True)
        except Exception as e:
            print(f"[{cls.__name__}] SKIPPED: {type(e).__name__}: {e}", flush=True)
    if not models:
        sys.exit(2)

    queries = [("POS", q) for q in POS_QUERIES] + [("NEG", q) for q in NEG_QUERIES]
    for m in models:
        try:
            print(f"\n===== {m.name} (clip = mean of {len(pils)} frames) =====", flush=True)
            clip_vec = l2norm(m.embed_frames_mean(pils))
            frame_vecs = [l2norm(m.embed_image(im)) for im in pils]
            for kind, q in queries:
                qv = l2norm(m.embed_text(q))
                s = cosine(qv, clip_vec)
                tag = "POS" if kind == "POS" else "neg"
                print(f"[{tag}] clip_sim={s:.3f} | {q[:75]}", flush=True)
            # temporal check on the most descriptive positive query
            qv = l2norm(m.embed_text(POS_QUERIES[1]))
            per = [cosine(qv, fv) for fv in frame_vecs]
            ts = [t for t, _ in stamped]
            print(f"[{m.name}] per-frame vs Q2: " +
                  ", ".join(f"{t:.1f}s={s:.3f}" for t, s in zip(ts, per)), flush=True)
            pos = [cosine(l2norm(m.embed_text(q)), clip_vec) for q in POS_QUERIES]
            neg = [cosine(l2norm(m.embed_text(q)), clip_vec) for q in NEG_QUERIES]
            print(f"[{m.name}] mean_pos={np.mean(pos):.3f} mean_neg={np.mean(neg):.3f} "
                  f"margin={np.mean(pos) - np.mean(neg):.3f}", flush=True)
        except Exception as e:
            import traceback
            print(f"[{m.name}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
