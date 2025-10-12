#!/bin/bash
# Auto-detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_NAME="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        OS_NAME="wsl"
    else
        OS_NAME="linux"
    fi
else
    OS_NAME="unknown"
fi

git pull && \
hugo && \
git add ./ && \
git commit -m "$(date +'%Y%m%d%H%M%S') on $OS_NAME" && \
git push
