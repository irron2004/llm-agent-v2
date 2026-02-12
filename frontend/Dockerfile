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
ARG VITE_BUILD_TARGET
ENV VITE_API_BASE=${VITE_API_BASE}
ENV VITE_API_PORT=${VITE_API_PORT}
ENV VITE_CHAT_PATH=${VITE_CHAT_PATH}
ENV VITE_INGESTIONS_BASE=${VITE_INGESTIONS_BASE}
ENV VITE_DEFAULT_INGESTION_RUN=${VITE_DEFAULT_INGESTION_RUN}
ENV VITE_BUILD_TARGET=${VITE_BUILD_TARGET}

COPY . .

RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# API proxy target (default: api container on internal port 8000)
# Override via docker-compose build args for dev/prod separation
ARG API_PROXY_TARGET=http://api:8000

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config for SPA + API proxy + ingestions
# NOTE: Using << EOF (no quotes) to enable variable substitution
# resolver 127.0.0.11 is Docker's embedded DNS server for container name resolution
RUN cat > /etc/nginx/conf.d/default.conf << EOF
resolver 127.0.0.11 valid=30s ipv6=off;

server {
    listen 9097;
    root /usr/share/nginx/html;
    index index.html;
    # Allow larger payloads for batch answer runs
    client_max_body_size 50m;

    # SPA fallback
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # API proxy (target set via build arg)
    # Use variable to enable runtime DNS resolution
    location /api/ {
        set \$upstream ${API_PROXY_TARGET};
        proxy_pass \$upstream;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
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
