FROM node:20 AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

# Build-time env for Vite (overridden via docker-compose build args)
ARG VITE_API_BASE
ARG VITE_API_PORT
ARG VITE_CHAT_PATH
ARG VITE_INGESTIONS_BASE
ARG VITE_DEFAULT_INGESTION_RUN
ENV VITE_API_BASE=${VITE_API_BASE}
ENV VITE_API_PORT=${VITE_API_PORT}
ENV VITE_CHAT_PATH=${VITE_CHAT_PATH}
ENV VITE_INGESTIONS_BASE=${VITE_INGESTIONS_BASE}
ENV VITE_DEFAULT_INGESTION_RUN=${VITE_DEFAULT_INGESTION_RUN}

COPY . .

RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config template for SPA + API proxy + ingestions
# NOTE:
# - default: proxy to compose-internal api service (prod 분리 기본값)
# - override per-container via env: NGINX_API_PROXY_PASS
ENV NGINX_API_PROXY_PASS=http://api:8000/api/
RUN mkdir -p /etc/nginx/templates && cat > /etc/nginx/templates/default.conf.template << 'EOF'
server {
    listen 9097;
    root /usr/share/nginx/html;
    index index.html;

    # Prevent stale SPA shell from browser cache.
    location ~* \.html$ {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires "0";
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy upstream (injected by envsubst from NGINX_API_PROXY_PASS)
    location /api/ {
        proxy_pass ${NGINX_API_PROXY_PASS};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        # Stream SSE responses (do not buffer)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # Ingestion data
    location /data/ingestions/ {
        alias /data/ingestions/;
        autoindex on;
    }
}
EOF

EXPOSE 9097

CMD ["nginx", "-g", "daemon off;"]
