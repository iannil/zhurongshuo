#!/bin/bash
git pull && \
hugo && \
git add ./ && \
git commit -m "$(date +'%Y%m%d%H%M%S') on wsl" && \
git push
