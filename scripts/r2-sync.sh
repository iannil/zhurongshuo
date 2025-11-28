#!/bin/bash

##############################################################################
# R2 同步工具
# 用途：上传单个或多个文件到 Cloudflare R2
# 作者：Claude Code
# 使用：./scripts/r2-sync.sh [OPTIONS] <file_or_directory>
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
BUCKET_NAME="zhurongshuo"
DRY_RUN=false
INCREMENTAL=false
VERBOSE=false

# 从环境变量或 .env 文件读取配置
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 帮助信息
show_help() {
    cat << EOF
使用方法: $(basename "$0") [OPTIONS] <file_or_directory>

上传文件到 Cloudflare R2，支持单文件或整个目录

选项:
    -h, --help              显示帮助信息
    -d, --dry-run           预览模式，不实际上传
    -i, --incremental       增量同步（通过查询 R2 在线状态跳过已存在的文件）
    -v, --verbose           详细输出（包括 R2 查询过程）
    --bucket NAME           指定 bucket 名称（默认：$BUCKET_NAME）

环境变量:
    CLOUDFLARE_ACCOUNT_ID   Cloudflare 账户 ID
    CLOUDFLARE_R2_API_TOKEN    Cloudflare API Token（需要 R2 读写权限）

增量同步说明:
    使用 --incremental 选项时，脚本会通过 wrangler r2 object head
    命令查询 R2 在线状态，只上传不存在的文件，已存在的文件会被跳过。
    这确保了同步的准确性，避免重复上传。

示例:
    # 上传单个文件
    $(basename "$0") static/images/gallery/photo.jpg

    # 上传整个目录
    $(basename "$0") static/images/gallery/

    # 增量同步（只上传 R2 中不存在的文件）
    $(basename "$0") --incremental static/images/gallery/

    # 增量同步 + 详细输出（查看 R2 查询过程）
    $(basename "$0") -i -v static/images/gallery/

    # 预览模式（查看将要上传的文件）
    $(basename "$0") --dry-run static/images/

EOF
}

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

# 检查依赖
check_dependencies() {
    if ! command -v wrangler &> /dev/null; then
        log_error "wrangler CLI 未安装"
        log_info "请运行: npm install -g wrangler"
        exit 1
    fi

    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ] || [ -z "$CLOUDFLARE_R2_API_TOKEN" ]; then
        log_error "缺少 Cloudflare 凭证"
        log_info "请设置环境变量："
        log_info "  export CLOUDFLARE_ACCOUNT_ID=your_account_id"
        log_info "  export CLOUDFLARE_R2_API_TOKEN=your_api_token"
        exit 1
    fi
}

# 上传单个文件到 R2
upload_file() {
    local local_path="$1"
    local r2_path="$2"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] 将上传: $local_path -> $r2_path"
        return 0
    fi

    if [ "$VERBOSE" = true ]; then
        log_info "上传: $local_path -> $r2_path"
    fi

    # 使用 wrangler 上传文件
    # 注意：这里使用 R2 的 S3 兼容 API，--remote 确保上传到远程 R2
    wrangler r2 object put "$BUCKET_NAME/$r2_path" --file="$local_path" --remote 2>&1 | \
        if [ "$VERBOSE" = true ]; then cat; else grep -v "^$"; fi

    if [ $? -eq 0 ]; then
        log_success "✓ $r2_path"
        return 0
    else
        log_error "✗ $r2_path"
        return 1
    fi
}

# 检查文件是否已存在于 R2
# 注意：始终查询 R2 在线状态，不依赖本地缓存
file_exists_in_r2() {
    local r2_path="$1"

    # 使用 wrangler r2 object head 检查对象是否存在（不下载内容，更高效）
    # 如果 head 命令不可用，回退到 get 方法
    if wrangler r2 object head "$BUCKET_NAME/$r2_path" &> /dev/null; then
        return 0
    else
        # head 命令失败时，尝试使用 get 方法验证
        # 注意：有些 wrangler 版本可能不支持 head 命令
        wrangler r2 object get "$BUCKET_NAME/$r2_path" --file=/dev/null &> /dev/null
        return $?
    fi
}

# 处理单个文件
process_file() {
    local file="$1"
    local base_dir="$2"

    # 计算 R2 路径
    # static/images/gallery/photo.jpg -> images/gallery/photo.jpg
    # 移除 "static/" 前缀，保留后面的路径结构
    local r2_path="${file#static/}"

    # 如果是增量模式，检查文件是否已存在（查询 R2 在线状态）
    if [ "$INCREMENTAL" = true ]; then
        if [ "$VERBOSE" = true ]; then
            log_info "检查 R2 在线状态: $r2_path"
        fi

        if file_exists_in_r2 "$r2_path"; then
            log_info "跳过（R2 已存在）: $r2_path"
            return 0
        else
            if [ "$VERBOSE" = true ]; then
                log_info "文件不存在，准备上传: $r2_path"
            fi
        fi
    fi

    upload_file "$file" "$r2_path"
}

# 处理目录
process_directory() {
    local dir="$1"

    log_info "扫描目录: $dir"

    local count=0
    local success=0
    local failed=0

    # 查找所有图片文件
    while IFS= read -r -d '' file; do
        count=$((count + 1))
        if process_file "$file" "$dir"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done < <(find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.webp" \) -print0)

    echo ""
    log_info "处理完成："
    log_info "  总计: $count 个文件"
    log_success "  成功: $success"
    if [ $failed -gt 0 ]; then
        log_error "  失败: $failed"
    fi
}

# 主函数
main() {
    # 解析参数
    local target=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -i|--incremental)
                INCREMENTAL=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --bucket)
                BUCKET_NAME="$2"
                shift 2
                ;;
            -*)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
            *)
                target="$1"
                shift
                ;;
        esac
    done

    # 检查参数
    if [ -z "$target" ]; then
        log_error "缺少文件或目录参数"
        show_help
        exit 1
    fi

    if [ ! -e "$target" ]; then
        log_error "文件或目录不存在: $target"
        exit 1
    fi

    # 检查依赖
    check_dependencies

    # 显示配置
    log_info "========================================="
    log_info "R2 同步工具"
    log_info "========================================="
    log_info "Bucket: $BUCKET_NAME"
    log_info "目标: $target"
    [ "$DRY_RUN" = true ] && log_warn "模式: 预览（不实际上传）"
    [ "$INCREMENTAL" = true ] && log_info "模式: 增量同步"
    log_info "========================================="
    echo ""

    # 处理文件或目录
    if [ -f "$target" ]; then
        # 单个文件
        local dir=$(dirname "$target")
        process_file "$target" "$dir"
    elif [ -d "$target" ]; then
        # 目录
        process_directory "$target"
    else
        log_error "不支持的目标类型"
        exit 1
    fi

    echo ""
    log_success "同步完成！"
}

main "$@"
