#!/bin/bash

##############################################################################
# Gallery Sync Script
# 自动同步 static/images/gallery 下的新增图片到 R2，并生成对应的 markdown 页面
##############################################################################

# 不使用 set -e，因为需要处理上传失败的情况

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
BUCKET_NAME="zhurongshuo"
GALLERY_DIR="static/images/gallery"
CONTENT_DIR="content/gallery"
R2_SYNC_RECORD=".r2-sync-record"

# 从环境变量或 .env 文件读取配置
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs 2>/dev/null) || true
fi

# wrangler 需要 CLOUDFLARE_API_TOKEN，如果没有设置但有 CLOUDFLARE_R2_API_TOKEN，则使用它
if [ -z "$CLOUDFLARE_API_TOKEN" ] && [ -n "$CLOUDFLARE_R2_API_TOKEN" ]; then
    export CLOUDFLARE_API_TOKEN="$CLOUDFLARE_R2_API_TOKEN"
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

# 检查依赖
check_dependencies() {
    if ! command -v wrangler &> /dev/null; then
        log_error "wrangler CLI 未安装"
        log_info "安装方法: npm install -g wrangler"
        return 1
    fi

    # wrangler 需要 CLOUDFLARE_API_TOKEN 环境变量
    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ] || [ -z "$CLOUDFLARE_API_TOKEN" ]; then
        log_error "缺少 Cloudflare 凭证"
        log_info "请在 .env 文件中配置 CLOUDFLARE_ACCOUNT_ID 和 CLOUDFLARE_API_TOKEN"
        return 1
    fi

    if [ ! -d "$GALLERY_DIR" ]; then
        log_error "图片目录不存在: $GALLERY_DIR"
        return 1
    fi

    if [ ! -d "$CONTENT_DIR" ]; then
        log_warn "内容目录不存在，创建: $CONTENT_DIR"
        mkdir -p "$CONTENT_DIR"
    fi

    return 0
}

# 初始化同步记录文件
init_sync_record() {
    if [ ! -f "$R2_SYNC_RECORD" ]; then
        log_info "创建同步记录文件: $R2_SYNC_RECORD"
        touch "$R2_SYNC_RECORD"
    fi
}

# 检查文件是否已同步
is_synced() {
    local file="$1"
    grep -qF "$file" "$R2_SYNC_RECORD" 2>/dev/null
    return $?
}

# 标记文件已同步
mark_synced() {
    local file="$1"
    echo "$file" >> "$R2_SYNC_RECORD"
}

# 上传文件到 R2
upload_to_r2() {
    local file="$1"
    local r2_path="$2"

    # 移除开头的斜杠（如果有）
    r2_path="${r2_path#/}"

    env CLOUDFLARE_API_TOKEN="$CLOUDFLARE_API_TOKEN" \
        CLOUDFLARE_ACCOUNT_ID="$CLOUDFLARE_ACCOUNT_ID" \
        wrangler r2 object put "$BUCKET_NAME/$r2_path" --file="$file" --remote &> /dev/null
    return $?
}

# 生成 markdown 文件
generate_markdown() {
    local image_file="$1"
    local image_name=$(basename "$image_file")
    local name_without_ext="${image_name%.*}"
    local md_file="$CONTENT_DIR/${name_without_ext}.md"

    # 如果 markdown 文件已存在，跳过
    if [ -f "$md_file" ]; then
        log_info "  Markdown 已存在，跳过: $name_without_ext.md"
        return 0
    fi

    # 生成当前时间戳
    local timestamp=$(date +"%Y-%m-%dT%H:%M:%S+08:00")

    # 创建 markdown 文件
    cat > "$md_file" << EOF
---
title: "$name_without_ext"
date: $timestamp
draft: false
type: "gallery"
featured_image: "/images/gallery/$image_name"
description: ""
tags: []
---
EOF

    log_success "  ✓ 生成 Markdown: $name_without_ext.md"
    return 0
}

# 同步图片到 R2 并生成 markdown
sync_gallery() {
    log_step "同步 Gallery 图片到 R2"

    if ! check_dependencies; then
        return 1
    fi

    init_sync_record

    log_info "扫描图片目录: $GALLERY_DIR"
    echo ""

    local total=0
    local uploaded=0
    local skipped=0
    local failed=0
    local new_markdown=0

    # 查找所有图片文件
    while IFS= read -r -d '' file; do
        [ -z "$file" ] && continue

        # 跳过 .gitkeep 等特殊文件
        if [[ "$file" == *".gitkeep"* ]] || [[ "$file" == *"node_modules"* ]]; then
            continue
        fi

        total=$((total + 1))

        # 计算 R2 路径：static/images/gallery/photo.jpg -> images/gallery/photo.jpg
        local r2_path="${file#static/}"
        local filename=$(basename "$file")

        # 检查是否已同步
        if is_synced "$file"; then
            skipped=$((skipped + 1))
            log_info "跳过（已同步）: $filename"
            continue
        fi

        # 上传文件到 R2
        log_info "上传到 R2: $filename"
        if upload_to_r2 "$file" "$r2_path"; then
            log_success "  ✓ R2 上传成功: $r2_path"
            mark_synced "$file"
            uploaded=$((uploaded + 1))

            # 生成对应的 markdown 文件
            if generate_markdown "$file"; then
                new_markdown=$((new_markdown + 1))
            fi
        else
            log_error "  ✗ R2 上传失败: $r2_path"
            failed=$((failed + 1))
        fi

        echo ""
    done < <(find "$GALLERY_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.webp" \) -print0 2>/dev/null)

    # 统计信息
    log_step "同步完成"
    log_info "统计信息："
    log_info "  总计图片: $total 个"
    log_success "  新上传: $uploaded 个"
    log_info "  跳过: $skipped 个（已同步）"
    log_success "  生成 Markdown: $new_markdown 个"

    if [ $failed -gt 0 ]; then
        log_error "  失败: $failed 个"
        return 1
    fi

    if [ $uploaded -eq 0 ]; then
        log_info "所有图片已同步"
    fi

    echo ""
    return 0
}

# 主函数
main() {
    sync_gallery
}

# 执行主函数
main "$@"
