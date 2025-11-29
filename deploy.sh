#!/bin/bash

##############################################################################
# Deploy Script with R2 Sync
# 部署脚本，支持自动同步静态资源到 Cloudflare R2
##############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
BUCKET_NAME="zhurongshuo"
SYNC_IMAGES=true  # 是否同步图片到 R2
SKIP_R2_ON_ERROR=true  # R2 同步失败时是否继续部署

# 从环境变量或 .env 文件读取配置
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs 2>/dev/null) || true
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

##############################################################################
# R2 同步函数（集成自 scripts/r2-sync.sh）
##############################################################################

# 检查 R2 依赖
check_r2_dependencies() {
    if ! command -v wrangler &> /dev/null; then
        log_warn "wrangler CLI 未安装，跳过 R2 同步"
        return 1
    fi

    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ] || [ -z "$CLOUDFLARE_R2_API_TOKEN" ]; then
        log_warn "缺少 Cloudflare 凭证，跳过 R2 同步"
        return 1
    fi

    return 0
}

# 检查文件是否已存在于 R2
file_exists_in_r2() {
    local r2_path="$1"

    # 使用 wrangler r2 object head 检查对象是否存在（不下载内容，更高效）
    if wrangler r2 object head "$BUCKET_NAME/$r2_path" &> /dev/null; then
        return 0
    else
        # head 命令失败时，尝试使用 get 方法验证
        wrangler r2 object get "$BUCKET_NAME/$r2_path" --file=/dev/null &> /dev/null
        return $?
    fi
}

# 上传文件到 R2
upload_to_r2() {
    local file="$1"
    local r2_path="$2"

    wrangler r2 object put "$BUCKET_NAME/$r2_path" --file="$file" --remote &> /dev/null
    return $?
}

# 同步图片到 R2（增量同步）
sync_images_to_r2() {
    log_step "同步图片到 R2"

    if [ "$SYNC_IMAGES" != "true" ]; then
        log_info "R2 同步已禁用，跳过"
        return 0
    fi

    if ! check_r2_dependencies; then
        if [ "$SKIP_R2_ON_ERROR" = "true" ]; then
            log_warn "R2 同步跳过，继续部署"
            return 0
        else
            return 1
        fi
    fi

    log_info "扫描待同步文件..."

    local total=0
    local uploaded=0
    local skipped=0
    local failed=0

    # 查找所有图片文件
    while IFS= read -r -d '' file; do
        [ -z "$file" ] && continue

        # 计算 R2 路径：static/images/gallery/photo.jpg -> images/gallery/photo.jpg
        local r2_path="${file#static/}"

        total=$((total + 1))

        # 检查文件是否已存在于 R2
        if file_exists_in_r2 "$r2_path"; then
            skipped=$((skipped + 1))
            continue
        fi

        # 上传文件
        log_info "上传: $r2_path"
        if upload_to_r2 "$file" "$r2_path"; then
            log_success "✓ $r2_path"
            uploaded=$((uploaded + 1))
        else
            log_error "✗ $r2_path"
            failed=$((failed + 1))
        fi
    done < <(find static -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.webp" \) -print0 2>/dev/null)

    echo ""
    log_info "R2 同步统计："
    log_info "  总计: $total 个文件"
    log_success "  上传: $uploaded 个"
    log_info "  跳过: $skipped 个（已存在）"

    if [ $failed -gt 0 ]; then
        log_error "  失败: $failed 个"
        if [ "$SKIP_R2_ON_ERROR" != "true" ]; then
            return 1
        fi
    fi

    if [ $uploaded -eq 0 ] && [ $total -gt 0 ]; then
        log_info "所有文件已同步到 R2"
    fi

    echo ""
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

    # Step 2: 同步图片到 R2（在构建之前）
    log_info "[2/7] 同步静态资源..."
    if sync_images_to_r2; then
        log_success "✓ R2 同步完成"
    else
        log_error "✗ R2 同步失败"
        exit 1
    fi

    # Step 3: 部署 Cloudflare Worker
    log_info "[3/7] 部署 Image Resizer Worker..."
    if bash scripts/deploy-worker.sh; then
        log_success "✓ Worker 部署成功"
    else
        log_warn "⚠ Worker 部署失败，继续构建站点"
    fi
    echo ""

    # Step 4: 导出内容到归档
    log_info "[4/7] 导出内容到归档..."
    if bash scripts/export.sh; then
        log_success "✓ 内容导出成功"
    else
        log_warn "⚠ 内容导出失败，继续构建站点"
    fi
    echo ""

    # Step 5: Hugo Build
    log_info "[5/7] 构建站点..."
    if hugo; then
        log_success "✓ 站点构建成功"
    else
        log_error "✗ 站点构建失败"
        exit 1
    fi
    echo ""

    # Step 6: Git Add & Commit
    log_info "[6/7] 提交更改..."
    git add ./
    if git commit -m "$(date +'%Y%m%d%H%M%S') on $OS_NAME"; then
        log_success "✓ 更改已提交"
    else
        log_warn "无新更改需要提交"
    fi
    echo ""

    # Step 7: Git Push
    log_info "[7/7] 推送到远程仓库..."
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
