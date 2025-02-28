FROM klakegg/hugo:0.107.0-extended AS builder

WORKDIR /src
COPY . .

RUN hugo --minify --gc

FROM caddy:2.7.5-alpine
# COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=builder /src/public /usr/share/caddy
EXPOSE 80 443

# CMD ["caddy", "file-server", "--access-log", "--listen", ":80"]