# Assistant Deployment Guide

## Deployment Target: Leonel Group Server

| Property | Value |
|----------|-------|
| IP Address | 138.68.59.201 |
| Domain | leonelgroup.co.ke |
| Subdomain | assistant.leonelgroup.co.ke |
| OS | Ubuntu 24.04 |
| SSH | `ssh root@138.68.59.201` |

## Services Deployed

| Service | Port | Container | Description |
|---------|------|-----------|-------------|
| Backend (Flask API) | 3001 | assistant-backend | Python Flask API with Ollama integration |
| Frontend (Next.js) | 3002 | assistant-frontend | Next.js web interface |

## Architecture

```
                    ┌─────────────────┐
                    │   Cloudflare    │
                    │  (SSL Proxy)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Nginx       │
                    │  (Reverse Proxy)│
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ assistant.      │ │ api.assistant.  │ │   Ollama        │
│ leonelgroup.    │ │ leonelgroup.    │ │  (localhost:    │
│ co.ke → :3002   │ │ co.ke → :3001   │ │   11434)        │
│ (Frontend)      │ │ (Backend API)   │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Quick Deploy Commands

```bash
# SSH into server
ssh root@138.68.59.201

# Navigate to app directory
cd /opt/assistant

# Pull latest and rebuild
git pull origin main
docker compose -f docker-compose.server.yml down
docker compose -f docker-compose.server.yml up -d --build

# Check status
docker ps | grep assistant
docker logs -f assistant-backend
```

## Initial Setup (One-time)

### 1. Add DNS Records in Cloudflare

Go to Cloudflare Dashboard → leonelgroup.co.ke → DNS and add:

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | assistant | 138.68.59.201 | Proxied (orange) |

### 2. Clone Repository

```bash
ssh root@138.68.59.201
cd /opt
git clone https://github.com/JT-tanui/Hm-office.git assistant
cd assistant
```

### 3. Build and Run Containers

```bash
docker compose -f docker-compose.server.yml up -d --build
```

### 4. Configure Nginx

```bash
cat > /etc/nginx/sites-available/assistant << 'EOF'
# Assistant Frontend
server {
    listen 80;
    listen [::]:80;
    server_name assistant.leonelgroup.co.ke;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name assistant.leonelgroup.co.ke;

    # Cloudflare Origin Certificate (wildcard)
    ssl_certificate /etc/ssl/cloudflare/origin.crt;
    ssl_certificate_key /etc/ssl/cloudflare/origin.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Cloudflare IP ranges
    set_real_ip_from 103.21.244.0/22;
    set_real_ip_from 103.22.200.0/22;
    set_real_ip_from 103.31.4.0/22;
    set_real_ip_from 104.16.0.0/13;
    set_real_ip_from 104.24.0.0/14;
    set_real_ip_from 108.162.192.0/18;
    set_real_ip_from 131.0.72.0/22;
    set_real_ip_from 141.101.64.0/18;
    set_real_ip_from 162.158.0.0/15;
    set_real_ip_from 172.64.0.0/13;
    set_real_ip_from 173.245.48.0/20;
    set_real_ip_from 188.114.96.0/20;
    set_real_ip_from 190.93.240.0/20;
    set_real_ip_from 197.234.240.0/22;
    set_real_ip_from 198.41.128.0/17;
    real_ip_header CF-Connecting-IP;

    # API routes - proxy to backend
    location /api/ {
        proxy_pass http://127.0.0.1:3001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Frontend - Next.js
    location / {
        proxy_pass http://127.0.0.1:3002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/assistant /etc/nginx/sites-enabled/

# Test and reload
nginx -t && systemctl reload nginx
```

## Environment Variables

Create `.env` file in `/opt/assistant/`:

```bash
cat > /opt/assistant/.env << 'EOF'
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Frontend Configuration  
NEXT_PUBLIC_API_URL=https://assistant.leonelgroup.co.ke/api
EOF
```

## Port Allocation

| Port | Service | Status |
|------|---------|--------|
| 3000 | Sanvellas (main site) | In use |
| 3001 | Assistant Backend | **Assigned** |
| 3002 | Assistant Frontend | **Assigned** |
| 3003-3005 | Available | - |
| 11434 | Ollama | In use |

## Useful Commands

```bash
# View running containers
docker ps | grep assistant

# View backend logs
docker logs -f assistant-backend

# View frontend logs
docker logs -f assistant-frontend

# Restart services
docker compose -f docker-compose.server.yml restart

# Rebuild and restart
docker compose -f docker-compose.server.yml up -d --build

# Check Ollama status
curl http://localhost:11434/api/tags

# Test backend API
curl http://localhost:3001/health

# Check nginx status
systemctl status nginx
```

## Troubleshooting

### 502 Bad Gateway
```bash
# Check if containers are running
docker ps | grep assistant

# Check container logs
docker logs assistant-backend
docker logs assistant-frontend

# Verify ports
ss -tlnp | grep -E "3001|3002"
```

### Ollama Connection Issues
```bash
# Check Ollama is running
systemctl status ollama

# Test Ollama directly
curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"Hi"}'

# Restart Ollama
systemctl restart ollama
```

### Low Memory
```bash
# Check memory usage
free -h
docker stats --no-stream

# Clean up unused Docker resources
docker system prune -a
```

## GitHub Actions Auto-Deploy

The repository includes a GitHub Actions workflow that automatically deploys on push to `main`. See `.github/workflows/deploy.yml`.

Required GitHub Secrets:
- `DROPLET_HOST`: 138.68.59.201
- `DROPLET_USER`: root
- `DROPLET_SSH_KEY`: SSH private key

## Maintenance

### Update Deployment
```bash
cd /opt/assistant
git pull origin main
docker compose -f docker-compose.server.yml up -d --build
```

### Backup Database
```bash
docker cp assistant-backend:/app/conversations.db /root/backups/conversations-$(date +%Y%m%d).db
```

### Check Disk Space
```bash
df -h
docker system df
```
