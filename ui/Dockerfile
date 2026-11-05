# syntax=docker/dockerfile:1
FROM nginxinc/nginx-unprivileged:1.27-alpine

# Set non-root user explicitly (UID 101 is nginx in unprivileged image)
USER 101

# Copy custom non-root nginx configuration
COPY --chown=101:101 nginx.conf /etc/nginx/nginx.conf

# Copy static chat UI files
COPY --chown=101:101 index.html /usr/share/nginx/html/index.html

EXPOSE 8080

# Exec form CMD (section 4.7)
CMD ["nginx", "-g", "daemon off;"]
