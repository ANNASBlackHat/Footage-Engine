# Remote MCP with a stable URL (self-hosted frp)

Colab VMs are ephemeral and ngrok URLs change every run. This setup gives you
**one fixed public URL** (`https://footage-mcp.<yourdomain>`) backed by your
own VPS: Colab runs the GPU + MCP, your VPS only forwards traffic.

```
Colab GPU (ephemeral)                    Your VPS (stable)
┌──────────────────────┐                ┌──────────────────────────┐
│ MCP :8000            │──frpc tunnel──▶│ frps :7000               │
│ (streamable-http)    │  (auto-        │ nginx :443 ──▶ :18080    │
└──────────────────────┘   reconnect)   └──────────────────────────┘
                                                    │ fixed subdomain
                                                    ▼
                                          https://footage-mcp.example.com/mcp
```

## One-time VPS setup (~15 min)

1. **DNS:** A record `footage-mcp` → your VPS IP.
2. **frps:** download the `linux_amd64` release from
   `github.com/fatedier/frp`, then:
   ```bash
   cp frps /usr/local/bin/ && mkdir -p /etc/frp
   cp deploy/frps.toml /etc/frp/frps.toml
   # edit /etc/frp/frps.toml — set both REPLACE_ME_TOKEN spots
   cp deploy/frps.service /etc/systemd/system/frps.service
   systemctl daemon-reload && systemctl enable --now frps
   ufw allow 7000/tcp && ufw allow 80,443/tcp
   ```
3. **nginx:** copy `deploy/footage-mcp.nginx.conf` to
   `/etc/nginx/sites-available/footage-mcp`, replace the domain, then:
   ```bash
   ln -s /etc/nginx/sites-available/footage-mcp /etc/nginx/sites-enabled/
   certbot --nginx -d footage-mcp.example.com
   ```
   Recommended: enable the basic-auth lines (the MCP endpoint itself has no
   login — anyone with the URL could otherwise query it).

## Each Colab run

```bash
!FRP_AUTH_TOKEN=<same token as frps.ini> FRPS_HOST=<vps ip> \
 MCP_SUBDOMAIN=footage-mcp.example.com \
 python <(curl -s https://raw.githubusercontent.com/ANNASBlackHat/Footage-Engine/main/scripts/colab_serve_mcp.py)
```

First run takes ~10 min (torch + Qwen 2B download, cached afterwards).
Your agents then use the fixed URL forever:

```json
{ "mcpServers": { "footage-engine": {
    "url": "https://footage-mcp.example.com/mcp" } } }
```

With HTTP basic auth enabled, clients append
`"headers": { "Authorization": "Basic <base64 user:pass>" }`.

## Notes

* `EMBEDDING_BACKEND=qwen` (default in the script) needs the T4; use
  `EMBEDDING_BACKEND=xclip` for a lighter CPU-able server.
* Secrets (`FRP_AUTH_TOKEN`, DB/Zilliz keys) travel as env vars into the
  ephemeral VM only — nothing secret is committed to the repo.
* If the VM dies, re-run the same command: same URL, frpc reconnects itself
  on network blips.
