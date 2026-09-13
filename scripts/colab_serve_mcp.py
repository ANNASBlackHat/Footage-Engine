"""Serve the footage-engine MCP from a Colab GPU through your own frp tunnel.

Run ONCE per Colab VM (paste into a cell or exec the file). It installs deps,
clones the repo, downloads frpc, starts the MCP (streamable-http) and the
tunnel. Your public URL stays the same every time because the subdomain is
fixed in DNS + nginx — unlike ngrok's random URLs.

On Colab:
    !FRP_AUTH_TOKEN=... FRPS_HOST=... MCP_SUBDOMAIN=footage-mcp.example.com \\
     GIT_URL=https://github.com/ANNASBlackHat/Footage-Engine.git \\
     python colab_serve_mcp.py
(or upload this file to the VM and set the env vars first)

Required env:
    FRP_AUTH_TOKEN   shared secret (same value as `token` in frps.ini)
    FRPS_HOST        your VPS public IP or hostname (frps bindPort 7000)
    MCP_SUBDOMAIN    stable public hostname, e.g. footage-mcp.example.com
Optional env:
    GIT_URL          repo to clone (default below; token URL for private repos)
    GIT_BRANCH       default: main
    EMBEDDING_BACKEND default: qwen (xclip also works, lighter/faster)
    MCP_PORT         local MCP port, default: 8000
    FRP_VERSION      default: 0.61.0

Keep the cell/kernel alive while serving; frpc auto-reconnects on blips.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

FRP_VERSION = os.environ.get("FRP_VERSION", "0.69.1")
GIT_URL = os.environ.get("GIT_URL", "https://github.com/ANNASBlackHat/Footage-Engine.git")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")
BACKEND = os.environ.get("EMBEDDING_BACKEND", "qwen")
MCP_PORT = os.environ.get("MCP_PORT", "8000")


def need(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    if not val:
        sys.exit(f"[serve] ERROR: set {name} env var first.")
    return val


# Secrets the MCP server itself needs (Colab: left panel -> Secrets (key icon),
# enable "Notebook access" for each). Same values as your local `.env`.
REQUIRED_SECRETS = [
    "DATABASE_URL",
    "VECTOR_STORE",
    "ZILLIZ_URI",
    "ZILLIZ_TOKEN",
    "ZILLIZ_COLLECTION_NAME",
    "QWEN_ZILLIZ_URI",
    "QWEN_ZILLIZ_TOKEN",
    "QWEN_ZILLIZ_COLLECTION_NAME",
    "IMAGEKIT_PUBLIC_KEY",
    "IMAGEKIT_PRIVATE_KEY",
    "IMAGEKIT_URL_ENDPOINT",
]


def check_secrets() -> None:
    missing = [k for k in REQUIRED_SECRETS if not os.environ.get(k)]
    if missing:
        sys.exit("[serve] ERROR: missing secrets (add them under Colab Secrets "
                 "with Notebook access enabled): " + ", ".join(missing))


def sh(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd))


def pip_install(pkgs: list[str]) -> None:
    print(f"[serve] pip install {' '.join(pkgs)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *pkgs])


def main() -> int:
    token = need("FRP_AUTH_TOKEN")
    frps_host = need("FRPS_HOST")
    subdomain = need("MCP_SUBDOMAIN")
    check_secrets()
    workdir = "/content/serve"
    os.makedirs(workdir, exist_ok=True)

    # 1. Dependencies (torch first: biggest download, keeps output flowing)
    pip_install(["torch"])
    pip_install(["transformers>=4.57", "sentence-transformers", "pillow", "numpy"])
    pip_install(["sqlalchemy", "pydantic", "pydantic-settings", "python-dotenv",
                 "pymilvus", "psycopg2-binary", "imagekitio", "requests",
                 "opencv-python-headless", "scenedetect", "mcp"])

    # 2. Repo (needs `git push origin main` first so the clone has latest code)
    repo_dir = os.path.join(workdir, "Footage-Engine")
    if not os.path.isdir(os.path.join(repo_dir, "footage_engine")):
        sh("git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_URL, repo_dir)

    # 3. frpc binary
    frpc = os.path.join(workdir, "frpc")
    if not os.path.exists(frpc):
        tgz = os.path.join(workdir, "frp.tgz")
        url = (f"https://github.com/fatedier/frp/releases/download/v{FRP_VERSION}/"
               f"frp_{FRP_VERSION}_linux_amd64.tar.gz")
        print(f"[serve] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(workdir)
        extracted = os.path.join(workdir, f"frp_{FRP_VERSION}_linux_amd64", "frpc")
        shutil.move(extracted, frpc)
        os.chmod(frpc, 0o755)

    # 4. frpc config (YAML — INI is deprecated; token injected, never committed)
    frpc_cfg = os.path.join(workdir, "frpc.yaml")
    with open(frpc_cfg, "w") as f:
        f.write("serverAddr: %s\nserverPort: 7000\n"
                "auth:\n  token: %s\n"
                "proxies:\n  - name: mcp\n    type: http\n"
                "    localPort: %d\n    customDomains: [%s]\n"
                % (frps_host, token, int(MCP_PORT), subdomain))
    print(f"[serve] frpc config: server={frps_host}:7000 domain={subdomain} "
          f"localPort={MCP_PORT}", flush=True)

    env = dict(os.environ, PYTHONPATH=repo_dir, EMBEDDING_BACKEND=BACKEND)
    mcp_log = open(os.path.join(workdir, "mcp.log"), "a")
    frp_log = open(os.path.join(workdir, "frpc.log"), "a")
    mcp = subprocess.Popen(
        [sys.executable, "-m", "footage_engine.mcp.server",
         "--transport", "streamable-http", "--host", "0.0.0.0", "--port", MCP_PORT,
         "--backend", BACKEND],
        cwd=repo_dir, env=env, stdout=mcp_log, stderr=subprocess.STDOUT)
    time.sleep(15)  # let the model load before exposing the tunnel
    if mcp.poll() is not None:
        sys.exit("[serve] ERROR: MCP exited during startup, see "
                 f"{workdir}/mcp.log")
    frpc_proc = subprocess.Popen(
        [frpc, "-c", frpc_cfg],
        stdout=frp_log, stderr=subprocess.STDOUT)
    time.sleep(5)  # fail fast on config/auth errors instead of serving half-dead
    if frpc_proc.poll() is not None:
        with open(os.path.join(workdir, "frpc.log")) as f:
            tail = "".join(f.readlines()[-15:])
        mcp.terminate()
        sys.exit(f"[serve] ERROR: frpc exited immediately. frpc.log tail:\n{tail}")

    print(f"[serve] MCP (pid {mcp.pid}) + frpc (pid {frpc_proc.pid}) running.", flush=True)
    print(f"[serve] STABLE URL: https://{subdomain}/mcp", flush=True)
    print("[serve] blocking to keep the tunnel alive (stop the cell to shut down).", flush=True)
    try:
        while True:
            time.sleep(60)
            for name, proc in (("mcp", mcp), ("frpc", frpc_proc)):
                if proc.poll() is not None:
                    print(f"[serve] {name} exited ({proc.returncode}), check {workdir}/{name}.log",
                          flush=True)
                    return 1
    except KeyboardInterrupt:
        pass
    finally:
        mcp.terminate()
        frpc_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
