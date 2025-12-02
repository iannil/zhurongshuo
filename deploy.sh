#!/bin/bash

##############################################################################
# Deploy Script
# 部署脚本 - 协调构建和发布流程
# 注意：R2 同步由 scripts/sync-gallery.sh 调用 scripts/r2-sync.sh 处理
##############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 从环境变量或 .env 文件读取配置
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

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

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# 检查脚本是否存在
check_script() {
    local script="$1"
    if [ ! -f "$script" ]; then
        log_error "脚本不存在: $script"
        return 1
    fi
    return 0
}

##############################################################################
# 主部署流程
##############################################################################

main() {
    log_step "开始部署"
    echo ""

    # Step 1: Git Pull
    log_info "[1/6] 拉取最新代码..."
    if git pull; then
        log_success "✓ 代码拉取成功"
    else
        log_error "✗ 代码拉取失败"
        exit 1
    fi
    echo ""

    # Step 2: 同步 Gallery 图片并生成 Markdown
    # 注意：此步骤内部会调用 scripts/r2-sync.sh 同步图片到 R2
    log_info "[2/6] 同步 Gallery 图片并生成页面..."
    if check_script "scripts/sync-gallery.sh" && bash scripts/sync-gallery.sh; then
        log_success "✓ Gallery 同步完成"
    else
        log_error "✗ Gallery 同步失败"
        exit 1
    fi
    echo ""

    # Step 3: 部署 Cloudflare Worker
    log_info "[3/6] 部署 Image Resizer Worker..."
    if check_script "scripts/deploy-worker.sh" && bash scripts/deploy-worker.sh; then
        log_success "✓ Worker 部署成功"
    else
        log_warn "⚠ Worker 部署失败，继续构建站点"
    fi
    echo ""

    # Step 4: 导出内容到归档
    log_info "[4/6] 导出内容到归档..."
    if check_script "scripts/export.sh" && bash scripts/export.sh; then
        log_success "✓ 内容导出成功"
    else
        log_warn "⚠ 内容导出失败，继续构建站点"
    fi
    echo ""

    # Step 5: Hugo Build
    log_info "[5/6] 构建站点..."
    if hugo; then
        log_success "✓ 站点构建成功"
    else
        log_error "✗ 站点构建失败"
        exit 1
    fi
    echo ""

    # Step 6: Git Add & Commit & Push
    log_info "[6/6] 提交并推送更改..."
    git add .
    if git commit -m "$(date +'%Y%m%d%H%M%S') on $OS_NAME"; then
        log_success "✓ 更改已提交"
    else
        log_warn "无新更改需要提交"
    fi
    echo ""
    if git push; then
        log_success "✓ 推送成功"
    else
        log_error "✗ 推送失败"
        exit 1
    fi
    echo ""

    log_step "部署完成！"
    log_success "站点已成功部署"
    echo ""
}

# 执行主函数
main "$@"
