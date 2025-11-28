#!/bin/bash

##############################################################################
# Cloudflare Worker Deployment Script
# 部署 image-resizer Worker 到 Cloudflare
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
    export $(grep -v '^#' .env | xargs 2>/dev/null) || true
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

##############################################################################
# 检查依赖
##############################################################################

check_dependencies() {
    log_step "检查依赖"

    # 检查 wrangler CLI
    if ! command -v wrangler &> /dev/null; then
        log_error "wrangler CLI 未安装"
        log_info "请运行: npm install -g wrangler"
        exit 1
    fi
    log_success "✓ wrangler CLI 已安装"

    # 检查环境变量
    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
        log_error "缺少 CLOUDFLARE_ACCOUNT_ID 环境变量"
        log_info "请在 .env 文件中设置 CLOUDFLARE_ACCOUNT_ID"
        exit 1
    fi
    log_success "✓ Cloudflare Account ID 已配置"

    if [ -z "$CLOUDFLARE_WORKER_API_TOKEN" ]; then
        log_warn "缺少 CLOUDFLARE_WORKER_API_TOKEN，将使用 wrangler login"
    else
        log_success "✓ Cloudflare Worker API Token 已配置"
    fi

    echo ""
}

##############################################################################
# 部署 Worker
##############################################################################

deploy_worker() {
    log_step "部署 Image Resizer Worker"

    # Set API token for wrangler
    export CLOUDFLARE_API_TOKEN="${CLOUDFLARE_WORKER_API_TOKEN}"

    # 部署到生产环境
    log_info "正在部署 Worker..."

    if wrangler deploy --env production; then
        log_success "✓ Worker 部署成功"
    else
        log_error "✗ Worker 部署失败"
        exit 1
    fi

    echo ""
}

##############################################################################
# 验证部署
##############################################################################

verify_deployment() {
    log_step "验证部署"

    log_info "测试 Worker 端点..."

    # 测试原图访问（不带参数）
    local test_url="https://r2.zhurongshuo.com/favicon.ico"
    log_info "测试 URL: $test_url"

    if curl -s -o /dev/null -w "%{http_code}" "$test_url" | grep -q "200"; then
        log_success "✓ 原图访问正常"
    else
        log_warn "⚠ 原图访问测试失败（可能文件不存在）"
    fi

    # 测试缩略图访问（带参数）
    local test_url_thumb="https://r2.zhurongshuo.com/favicon.ico?w=100&q=75"
    log_info "测试 URL: $test_url_thumb"

    local response_headers=$(curl -s -I "$test_url_thumb")
    if echo "$response_headers" | grep -q "200"; then
        log_success "✓ 缩略图参数解析正常"

        # 检查处理状态
        if echo "$response_headers" | grep -qi "X-Image-Processing"; then
            local processing_status=$(echo "$response_headers" | grep -i "X-Image-Processing" | cut -d: -f2 | tr -d ' \r')
            log_info "图片处理状态: $processing_status"

            if [ "$processing_status" = "resized" ]; then
                log_success "✓ 图片缩放功能正常工作"
            elif [ "$processing_status" = "original-fallback" ]; then
                log_warn "⚠ 图片缩放功能未启用，返回原图"
                log_info "提示: 需要启用 Cloudflare Image Resizing 服务"
                log_info "访问: https://dash.cloudflare.com/?to=/:account/images/image-resizing"
            else
                log_info "当前返回原图（未请求缩放）"
            fi
        fi
    else
        log_warn "⚠ 缩略图访问测试失败（可能文件不存在）"
    fi

    echo ""
}

##############################################################################
# 显示使用说明
##############################################################################

show_usage_info() {
    log_step "部署成功！"

    echo -e "${GREEN}Worker 已成功部署到 Cloudflare${NC}"
    echo ""
    echo -e "${BLUE}使用方法：${NC}"
    echo "1. 原图访问:"
    echo "   https://r2.zhurongshuo.com/images/gallery/photo.jpg"
    echo ""
    echo "2. 缩略图访问 (600px 宽):"
    echo "   https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=600&q=75"
    echo ""
    echo "3. 自定义尺寸:"
    echo "   https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=800&h=600&q=85"
    echo ""
    echo -e "${YELLOW}注意事项：${NC}"
    echo "• 如需真正的图片缩放功能，请启用 Cloudflare Image Resizing ($5/月)"
    echo "• 当前配置会优雅降级：未启用时返回原图，启用后自动缩放"
    echo "• 所有响应都有良好的缓存策略，提升性能"
    echo ""
}

##############################################################################
# 主函数
##############################################################################

main() {
    echo ""
    check_dependencies
    deploy_worker
    verify_deployment
    show_usage_info
}

# 执行主函数
main "$@"
