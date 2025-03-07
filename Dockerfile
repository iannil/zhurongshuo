FROM hugomods/hugo:nightly AS builder

WORKDIR /src
COPY . .

RUN hugo --minify --gc

FROM caddy:2.9.1-alpine
WORKDIR /srv
COPY --from=builder /src/docs ./docs
COPY Caddyfile /etc/caddy/Caddyfile

RUN chown -R caddy:caddy /srv/docs

EXPOSE 80
USER caddy