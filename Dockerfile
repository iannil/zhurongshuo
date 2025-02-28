FROM hugomods/hugo:nightly AS builder

WORKDIR /src
COPY . .

RUN hugo --minify --gc

FROM caddy:2.9.1-alpine
# COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=builder /src/docs /usr/share/caddy
EXPOSE 80 443

# CMD ["caddy", "file-server", "--access-log", "--listen", ":80"]