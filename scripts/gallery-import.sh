#!/bin/bash

##############################################################################
# Gallery Import Script
# 批量导入图片到图库，自动生成 markdown 文件
# 支持从 EXIF 提取日期，自动优化图片
##############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GALLERY_CONTENT="${PROJECT_ROOT}/content/gallery"
GALLERY_IMAGES="${PROJECT_ROOT}/static/images/gallery"

# 默认参数
SOURCE_DIR=""
OPTIMIZE=true
USE_EXIF=true
DRY_RUN=false
DATE_STRUCTURE=true  # 是否使用年月目录结构
TITLE_PREFIX=""      # 标题前缀

# 帮助信息
show_help() {
    cat << EOF
用法: $0 <源目录> [选项]

批量导入图片到图库

参数:
  源目录                导入图片的源目录路径

选项:
  --no-optimize         不优化图片
  --no-exif             不从 EXIF 读取日期
  --flat                不使用年月目录结构
  --title-prefix TEXT   标题前缀
  -n, --dry-run         演习模式，不实际创建文件
  -h, --help            显示此帮助信息

示例:
  $0 ~/Pictures/photos                    # 导入照片目录
  $0 ~/Downloads/artwork --no-optimize    # 导入但不优化
  $0 ~/Pictures --title-prefix "旅行"      # 添加标题前缀

EOF
}

# 解析命令行参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

SOURCE_DIR="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-optimize)
            OPTIMIZE=false
            shift
            ;;
        --no-exif)
            USE_EXIF=false
            shift
            ;;
        --flat)
            DATE_STRUCTURE=false
            shift
            ;;
        --title-prefix)
            TITLE_PREFIX="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 验证源目录
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}错误: 源目录不存在: $SOURCE_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  图库批量导入工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "配置:"
echo -e "  源目录: ${SOURCE_DIR}"
echo -e "  目标目录: ${GALLERY_CONTENT}"
echo -e "  图片目录: ${GALLERY_IMAGES}"
echo -e "  优化图片: ${OPTIMIZE}"
echo -e "  使用 EXIF: ${USE_EXIF}"
echo -e "  目录结构: $([ "$DATE_STRUCTURE" = true ] && echo "按年月分组" || echo "扁平结构")"
[ -n "$TITLE_PREFIX" ] && echo -e "  标题前缀: ${TITLE_PREFIX}"
echo -e "  演习模式: ${DRY_RUN}"
echo ""

##############################################################################
# 辅助函数
##############################################################################

# 生成 slug（URL 友好的文件名）
generate_slug() {
    local filename="$1"

    # 移除扩展名
    slug="${filename%.*}"

    # 转换为小写
    slug=$(echo "$slug" | tr '[:upper:]' '[:lower:]')

    # 移除特殊字符，保留字母、数字、中文、连字符
    slug=$(echo "$slug" | sed 's/[^a-z0-9\u4e00-\u9fa5_-]/-/g')

    # 移除多余的连字符
    slug=$(echo "$slug" | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//')

    echo "$slug"
}

# 从图片 EXIF 提取日期
get_exif_date() {
    local image_file="$1"

    if ! command -v exiftool &> /dev/null; then
        return 1
    fi

    local date_str=$(exiftool -DateTimeOriginal -CreateDate -d "%Y-%m-%dT%H:%M:%S+08:00" "$image_file" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+08:00' | head -n1)

    if [ -n "$date_str" ]; then
        echo "$date_str"
        return 0
    fi

    return 1
}

# 从文件修改时间获取日期
get_file_date() {
    local file="$1"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f "%Sm" -t "%Y-%m-%dT%H:%M:%S+08:00" "$file"
    else
        date -r "$file" "+%Y-%m-%dT%H:%M:%S+08:00"
    fi
}

# 提取年月
extract_year_month() {
    local date_str="$1"

    year=$(echo "$date_str" | grep -oE '^[0-9]{4}')
    month=$(echo "$date_str" | grep -oE '-[0-9]{2}-' | tr -d '-')

    echo "$year/$month"
}

# 创建 markdown 文件
create_markdown() {
    local image_file="$1"
    local slug="$2"
    local date="$3"
    local title="$4"

    # 构建目标路径
    local md_dir="$GALLERY_CONTENT"
    if [ "$DATE_STRUCTURE" = true ]; then
        local year_month=$(extract_year_month "$date")
        md_dir="$GALLERY_CONTENT/$year_month"
    fi

    local md_file="$md_dir/${slug}.md"

    # 检查文件是否已存在
    if [ -f "$md_file" ]; then
        echo -e "${YELLOW}  ! 文件已存在: $md_file (跳过)${NC}"
        return 1
    fi

    # 创建目录
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$md_dir"
    fi

    # 生成 frontmatter
    local content="---
title: \"${title}\"
date: ${date}
draft: false
type: \"gallery\"
featured_image: \"/images/gallery/$(basename "$image_file")\"
description: \"\"
tags: []
---
"

    # 写入文件
    if [ "$DRY_RUN" = false ]; then
        echo "$content" > "$md_file"
    fi

    echo "$md_file"
}

##############################################################################
# 检查依赖工具
##############################################################################

echo -e "${YELLOW}[1/4] 检查依赖工具...${NC}"

if [ "$USE_EXIF" = true ] && ! command -v exiftool &> /dev/null; then
    echo -e "${YELLOW}  ! exiftool 未安装，将使用文件修改时间${NC}"
    USE_EXIF=false
fi

if [ "$OPTIMIZE" = true ] && ! command -v convert &> /dev/null; then
    echo -e "${YELLOW}  ! ImageMagick 未安装，将跳过图片优化${NC}"
    OPTIMIZE=false
fi

echo -e "${GREEN}  ✓ 依赖检查完成${NC}"
echo ""

##############################################################################
# 扫描源目录
##############################################################################

echo -e "${YELLOW}[2/4] 扫描源目录...${NC}"

# 查找所有图片文件
IMAGE_FILES=$(find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null || true)

IMAGE_COUNT=$(echo "$IMAGE_FILES" | grep -c . || echo 0)

if [ $IMAGE_COUNT -eq 0 ]; then
    echo -e "${RED}未找到任何图片文件${NC}"
    exit 0
fi

echo -e "${GREEN}  ✓ 找到 ${IMAGE_COUNT} 个图片文件${NC}"
echo ""

##############################################################################
# 导入图片
##############################################################################

echo -e "${YELLOW}[3/4] 导入图片...${NC}"

IMPORTED=0
SKIPPED=0
FAILED=0

while IFS= read -r source_file; do
    [ -z "$source_file" ] && continue

    filename=$(basename "$source_file")
    echo -e "${BLUE}处理: $filename${NC}"

    # 生成 slug
    slug=$(generate_slug "$filename")

    # 获取日期
    if [ "$USE_EXIF" = true ]; then
        date=$(get_exif_date "$source_file")
        if [ -z "$date" ]; then
            date=$(get_file_date "$source_file")
            echo -e "${YELLOW}  ! EXIF 无日期，使用文件时间${NC}"
        fi
    else
        date=$(get_file_date "$source_file")
    fi

    # 生成标题
    title="$filename"
    if [ -n "$TITLE_PREFIX" ]; then
        title="${TITLE_PREFIX} - ${filename}"
    fi

    # 目标图片路径
    target_image="$GALLERY_IMAGES/$filename"

    # 复制图片
    if [ "$DRY_RUN" = false ]; then
        if [ ! -f "$target_image" ]; then
            cp "$source_file" "$target_image"
            echo -e "${GREEN}  ✓ 复制图片到: $target_image${NC}"

            # 优化图片
            if [ "$OPTIMIZE" = true ]; then
                original_size=$(stat -f%z "$target_image" 2>/dev/null || echo 0)

                # 根据格式优化
                if [[ "$filename" =~ \.(jpg|jpeg|JPG|JPEG)$ ]]; then
                    convert "$target_image" -strip -quality 85 -sampling-factor 4:2:0 "$target_image.tmp" 2>/dev/null
                else
                    convert "$target_image" -strip -define png:compression-level=9 "$target_image.tmp" 2>/dev/null
                fi

                new_size=$(stat -f%z "$target_image.tmp" 2>/dev/null || echo 0)

                if [ $new_size -lt $original_size ]; then
                    mv "$target_image.tmp" "$target_image"
                    echo -e "${GREEN}  ✓ 优化: $(numfmt --to=iec $original_size) → $(numfmt --to=iec $new_size)${NC}"
                else
                    rm "$target_image.tmp"
                fi
            fi
        else
            echo -e "${YELLOW}  ! 图片已存在，跳过复制${NC}"
        fi
    fi

    # 创建 markdown
    md_file=$(create_markdown "$target_image" "$slug" "$date" "$title")

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ 创建: $md_file${NC}"
        IMPORTED=$((IMPORTED + 1))
    else
        SKIPPED=$((SKIPPED + 1))
    fi

    echo ""

done <<< "$IMAGE_FILES"

##############################################################################
# 总结
##############################################################################

echo -e "${BLUE}========================================${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}演习模式完成${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "预计导入: ${IMPORTED} 个"
    echo -e "预计跳过: ${SKIPPED} 个"
    echo ""
    echo -e "如果确认无误，请去掉 --dry-run 参数重新运行"
else
    echo -e "${GREEN}✓ 导入完成!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "统计信息:"
    echo -e "  • 成功导入: ${IMPORTED} 个"
    echo -e "  • 跳过: ${SKIPPED} 个"
    echo -e "  • 失败: ${FAILED} 个"
    echo ""
    echo -e "${YELLOW}后续步骤:${NC}"
    echo -e "  1. 运行验证: scripts/gallery-meta.sh validate"
    echo -e "  2. 预览站点: hugo server"
    echo -e "  3. 提交更改: git add . && git commit -m 'feat: 导入新图库内容'"
fi
echo ""
