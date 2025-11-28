#!/bin/bash

##############################################################################
# R2 批量迁移脚本
# 用途：将现有图片从 static/images/gallery/ 批量迁移到 Cloudflare R2
# 作者：Claude Code
# 使用：./scripts/migrate-to-r2.sh [OPTIONS]
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认配置
SOURCE_DIR="static/images/gallery"
BUCKET_NAME="zhurongshuo"
DRY_RUN=false
BACKUP=true
LOG_FILE="logs/r2-migration-$(date +%Y%m%d_%H%M%S).log"

# 从环境变量或 .env 文件读取配置
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 创建日志目录
mkdir -p logs

# 帮助信息
show_help() {
    cat << EOF
使用方法: $(basename "$0") [OPTIONS]

批量迁移 static/images/gallery/ 中的所有图片到 Cloudflare R2

选项:
    -h, --help              显示帮助信息
    -d, --dry-run           预览模式，不实际上传
    --no-backup             不创建备份
    --source DIR            指定源目录（默认：$SOURCE_DIR）
    --bucket NAME           指定 bucket 名称（默认：$BUCKET_NAME）

环境变量:
    CLOUDFLARE_ACCOUNT_ID   Cloudflare 账户 ID
    CLOUDFLARE_API_TOKEN    Cloudflare API Token（需要 R2 写权限）

示例:
    # 预览迁移
    $(basename "$0") --dry-run

    # 执行迁移
    $(basename "$0")

    # 迁移不备份
    $(basename "$0") --no-backup

EOF
}

# 日志函数
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

# 检查依赖
check_dependencies() {
    log_step "检查依赖..."

    if ! command -v wrangler &> /dev/null; then
        log_error "wrangler CLI 未安装"
        log_info "请运行: npm install -g wrangler"
        exit 1
    fi

    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ] || [ -z "$CLOUDFLARE_API_TOKEN" ]; then
        log_error "缺少 Cloudflare 凭证"
        log_info "请设置环境变量："
        log_info "  export CLOUDFLARE_ACCOUNT_ID=your_account_id"
        log_info "  export CLOUDFLARE_API_TOKEN=your_api_token"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 扫描待迁移文件
scan_files() {
    log_step "扫描待迁移文件..."

    if [ ! -d "$SOURCE_DIR" ]; then
        log_error "源目录不存在: $SOURCE_DIR"
        exit 1
    fi

    # 查找所有图片文件
    local files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.webp" \) -print0)

    local count=${#files[@]}
    local total_size=$(du -sh "$SOURCE_DIR" 2>/dev/null | cut -f1)

    log_info "找到 $count 个图片文件"
    log_info "总大小: $total_size"

    if [ $count -eq 0 ]; then
        log_warn "没有找到待迁移文件"
        exit 0
    fi

    # 显示文件列表样例
    log_info "文件列表样例（前 5 个）："
    local i=0
    for file in "${files[@]}"; do
        if [ $i -ge 5 ]; then break; fi
        local size=$(du -h "$file" | cut -f1)
        log_info "  - $(basename "$file") ($size)"
        i=$((i + 1))
    done

    echo "${files[@]}"
}

# 创建备份
create_backup() {
    if [ "$BACKUP" = false ] || [ "$DRY_RUN" = true ]; then
        return 0
    fi

    log_step "创建备份..."

    local backup_dir="backups/images-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    cp -r "$SOURCE_DIR" "$backup_dir/"
    log_success "备份创建完成: $backup_dir"
}

# 上传文件到 R2
upload_to_r2() {
    local file="$1"
    local total="$2"
    local current="$3"

    # 计算 R2 路径
    # static/images/gallery/photo.jpg -> images/gallery/photo.jpg
    local r2_path="${file#static/}"

    local percentage=$((current * 100 / total))
    log_info "[$current/$total] ($percentage%) 上传: $(basename "$file")"

    if [ "$DRY_RUN" = true ]; then
        log_info "  [DRY-RUN] $file -> $r2_path"
        return 0
    fi

    # 使用 wrangler 上传
    if wrangler r2 object put "$BUCKET_NAME/$r2_path" --file="$file" >> "$LOG_FILE" 2>&1; then
        log_success "  ✓ $r2_path"
        return 0
    else
        log_error "  ✗ 上传失败: $r2_path"
        return 1
    fi
}

# 验证上传
verify_upload() {
    local file="$1"
    local r2_path="${file#static/}"

    if [ "$DRY_RUN" = true ]; then
        return 0
    fi

    # 获取本地文件 MD5
    local local_md5=$(md5 -q "$file" 2>/dev/null || md5sum "$file" | cut -d' ' -f1)

    # 检查 R2 文件是否存在
    if wrangler r2 object get "$BUCKET_NAME/$r2_path" --file=/dev/null >> "$LOG_FILE" 2>&1; then
        return 0
    else
        log_error "验证失败: $r2_path 不存在"
        return 1
    fi
}

# 批量迁移
migrate_all() {
    local files=($1)
    local total=${#files[@]}
    local success=0
    local failed=0
    local current=0

    log_step "开始批量迁移..."
    log_info "总计: $total 个文件"
    echo ""

    for file in "${files[@]}"; do
        current=$((current + 1))

        if upload_to_r2 "$file" "$total" "$current"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo ""
    log_step "迁移统计："
    log_info "  总计: $total"
    log_success "  成功: $success"
    if [ $failed -gt 0 ]; then
        log_error "  失败: $failed"
    fi

    return $failed
}

# 生成迁移报告
generate_report() {
    local success=$1
    local failed=$2
    local total=$((success + failed))

    log_step "生成迁移报告..."

    cat << EOF >> "$LOG_FILE"

========================================
迁移报告
========================================
时间: $(date '+%Y-%m-%d %H:%M:%S')
源目录: $SOURCE_DIR
目标 Bucket: $BUCKET_NAME
========================================
总文件数: $total
成功: $success
失败: $failed
成功率: $(echo "scale=2; $success * 100 / $total" | bc)%
========================================

EOF

    log_info "详细日志已保存到: $LOG_FILE"
}

# 显示下一步操作
show_next_steps() {
    echo ""
    log_step "下一步操作："
    echo ""
    echo "1. 验证图片访问："
    echo "   打开浏览器访问: https://r2.zhurongshuo.com/images/gallery/[图片名]"
    echo ""
    echo "2. 部署 Cloudflare Worker："
    echo "   cd /Users/iannil/Code/zhurongshuo"
    echo "   wrangler deploy"
    echo ""
    echo "3. 更新 Hugo 模板（使用 R2 URL）"
    echo ""
    echo "4. 本地测试构建："
    echo "   hugo server"
    echo ""
    echo "5. 确认无误后，清理本地图片："
    echo "   git rm -r static/images/gallery/"
    echo ""
}

# 主函数
main() {
    # 解析参数
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
            --no-backup)
                BACKUP=false
                shift
                ;;
            --source)
                SOURCE_DIR="$2"
                shift 2
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
                shift
                ;;
        esac
    done

    # 显示标题
    echo ""
    log "========================================="
    log "R2 批量迁移脚本"
    log "========================================="
    log "源目录: $SOURCE_DIR"
    log "目标 Bucket: $BUCKET_NAME"
    [ "$DRY_RUN" = true ] && log_warn "模式: 预览（不实际上传）"
    [ "$BACKUP" = true ] && log_info "备份: 启用"
    log "日志文件: $LOG_FILE"
    log "========================================="
    echo ""

    # 检查依赖
    check_dependencies

    # 扫描文件
    local files=$(scan_files)
    echo ""

    # 确认执行
    if [ "$DRY_RUN" = false ]; then
        read -p "确认开始迁移？(y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_warn "迁移已取消"
            exit 0
        fi
        echo ""
    fi

    # 创建备份
    create_backup
    echo ""

    # 执行迁移
    migrate_all "$files"
    local failed=$?

    # 生成报告
    generate_report "$((${#files[@]} - failed))" "$failed"

    # 显示下一步
    if [ "$DRY_RUN" = false ] && [ $failed -eq 0 ]; then
        show_next_steps
    fi

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "迁移成功完成！"
        exit 0
    else
        log_error "迁移完成，但有 $failed 个文件失败"
        exit 1
    fi
}

main "$@"
