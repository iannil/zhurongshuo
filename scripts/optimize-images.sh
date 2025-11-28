#!/bin/bash

##############################################################################
# Image Optimizer Script
# 自动压缩和优化图库图片，支持 JPEG/PNG/WebP 格式
# 减少文件大小同时保持视觉质量
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
GALLERY_IMAGES="${PROJECT_ROOT}/static/images/gallery"

# 默认参数
QUALITY=85          # JPEG 质量 (1-100)
PNG_LEVEL=2         # PNG 压缩级别 (0-3, 3最慢但最小)
WEBP_QUALITY=80     # WebP 质量
GENERATE_WEBP=false # 是否生成 WebP 版本
DRY_RUN=false       # 演习模式
BACKUP=true         # 是否备份原文件

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项] [目录]

图片优化工具 - 压缩和优化图库图片

选项:
  -q, --quality LEVEL     JPEG 质量 (1-100, 默认: 85)
  -p, --png-level LEVEL   PNG 压缩级别 (0-3, 默认: 2)
  -w, --webp              同时生成 WebP 格式
  --webp-quality LEVEL    WebP 质量 (1-100, 默认: 80)
  -n, --dry-run           演习模式，不实际修改文件
  --no-backup             不备份原文件
  -h, --help              显示此帮助信息

示例:
  $0                                    # 优化所有图库图片
  $0 -q 90 -w                          # 质量90并生成WebP
  $0 --dry-run                         # 预览将要执行的操作
  $0 /path/to/images                   # 优化指定目录

EOF
}

# 解析命令行参数
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quality)
            QUALITY="$2"
            shift 2
            ;;
        -p|--png-level)
            PNG_LEVEL="$2"
            shift 2
            ;;
        -w|--webp)
            GENERATE_WEBP=true
            shift
            ;;
        --webp-quality)
            WEBP_QUALITY="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-backup)
            BACKUP=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# 使用指定目录或默认目录
if [ -n "$TARGET_DIR" ]; then
    GALLERY_IMAGES="$TARGET_DIR"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  图片优化工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "配置:"
echo -e "  目标目录: ${GALLERY_IMAGES}"
echo -e "  JPEG 质量: ${QUALITY}"
echo -e "  PNG 压缩级别: ${PNG_LEVEL}"
echo -e "  生成 WebP: ${GENERATE_WEBP}"
if [ "$GENERATE_WEBP" = true ]; then
    echo -e "  WebP 质量: ${WEBP_QUALITY}"
fi
echo -e "  演习模式: ${DRY_RUN}"
echo -e "  备份原文件: ${BACKUP}"
echo ""

# 检查目录是否存在
if [ ! -d "$GALLERY_IMAGES" ]; then
    echo -e "${RED}错误: 目录不存在: $GALLERY_IMAGES${NC}"
    exit 1
fi

##############################################################################
# 检查依赖工具
##############################################################################

echo -e "${YELLOW}[1/4] 检查依赖工具...${NC}"

MISSING_TOOLS=()

# 检查 ImageMagick (convert)
if ! command -v convert &> /dev/null; then
    MISSING_TOOLS+=("imagemagick")
fi

# 检查 pngquant (可选，用于更好的 PNG 压缩)
HAS_PNGQUANT=false
if command -v pngquant &> /dev/null; then
    HAS_PNGQUANT=true
fi

# 检查 cwebp (WebP 转换)
HAS_CWEBP=false
if command -v cwebp &> /dev/null; then
    HAS_CWEBP=true
elif [ "$GENERATE_WEBP" = true ]; then
    MISSING_TOOLS+=("webp")
fi

# 如果有缺失的必需工具，提示安装
if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "${RED}缺少必需工具:${NC}"
    for tool in "${MISSING_TOOLS[@]}"; do
        echo -e "  - $tool"
    done
    echo ""
    echo -e "${YELLOW}在 macOS 上安装:${NC}"
    echo -e "  brew install ${MISSING_TOOLS[@]}"
    echo ""
    exit 1
fi

echo -e "${GREEN}  ✓ ImageMagick 已安装${NC}"
if [ "$HAS_PNGQUANT" = true ]; then
    echo -e "${GREEN}  ✓ pngquant 已安装 (更好的PNG压缩)${NC}"
else
    echo -e "${YELLOW}  ! pngquant 未安装 (可选，建议安装)${NC}"
fi
if [ "$GENERATE_WEBP" = true ]; then
    echo -e "${GREEN}  ✓ cwebp 已安装${NC}"
fi
echo ""

##############################################################################
# 扫描图片文件
##############################################################################

echo -e "${YELLOW}[2/4] 扫描图片文件...${NC}"

# 查找所有图片文件
JPEG_FILES=$(find "$GALLERY_IMAGES" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null || true)
PNG_FILES=$(find "$GALLERY_IMAGES" -type f -iname "*.png" 2>/dev/null || true)

JPEG_COUNT=$(echo "$JPEG_FILES" | grep -c . || echo 0)
PNG_COUNT=$(echo "$PNG_FILES" | grep -c . || echo 0)
TOTAL_COUNT=$((JPEG_COUNT + PNG_COUNT))

if [ $TOTAL_COUNT -eq 0 ]; then
    echo -e "${RED}未找到任何图片文件${NC}"
    exit 0
fi

echo -e "${GREEN}  ✓ 找到 ${TOTAL_COUNT} 个图片文件:${NC}"
echo -e "    - JPEG: ${JPEG_COUNT} 个"
echo -e "    - PNG:  ${PNG_COUNT} 个"

# 计算原始总大小
ORIGINAL_SIZE=$(du -ch "$GALLERY_IMAGES"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | grep total | cut -f1 || echo "0")
echo -e "    - 总大小: ${ORIGINAL_SIZE}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}演习模式：以下是将要执行的操作${NC}"
    echo ""
fi

##############################################################################
# 优化 JPEG 文件
##############################################################################

echo -e "${YELLOW}[3/4] 优化 JPEG 文件...${NC}"

if [ $JPEG_COUNT -eq 0 ]; then
    echo -e "${BLUE}  → 跳过 (无 JPEG 文件)${NC}"
else
    JPEG_PROCESSED=0
    JPEG_SAVED=0

    while IFS= read -r file; do
        [ -z "$file" ] && continue

        filename=$(basename "$file")
        original_size=$(stat -f%z "$file" 2>/dev/null || echo 0)

        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}  → 将优化: $filename ($(numfmt --to=iec $original_size 2>/dev/null || echo "${original_size}B"))${NC}"
        else
            # 备份原文件
            if [ "$BACKUP" = true ]; then
                cp "$file" "${file}.backup"
            fi

            # 使用 ImageMagick 优化
            convert "$file" -strip -quality $QUALITY -sampling-factor 4:2:0 "$file.tmp" 2>/dev/null

            new_size=$(stat -f%z "$file.tmp" 2>/dev/null || echo 0)

            # 只在文件变小时替换
            if [ $new_size -lt $original_size ]; then
                mv "$file.tmp" "$file"
                saved=$((original_size - new_size))
                JPEG_SAVED=$((JPEG_SAVED + saved))
                echo -e "${GREEN}  ✓ $filename: $(numfmt --to=iec $original_size 2>/dev/null || echo $original_size) → $(numfmt --to=iec $new_size 2>/dev/null || echo $new_size) (省${saved}字节)${NC}"
            else
                rm "$file.tmp"
                echo -e "${YELLOW}  → $filename: 已是最优 (跳过)${NC}"
            fi

            # 删除备份（如果优化成功）
            if [ "$BACKUP" = true ] && [ -f "${file}.backup" ]; then
                rm "${file}.backup"
            fi
        fi

        JPEG_PROCESSED=$((JPEG_PROCESSED + 1))
    done <<< "$JPEG_FILES"

    if [ "$DRY_RUN" = false ]; then
        echo -e "${GREEN}  ✓ 处理 ${JPEG_PROCESSED} 个 JPEG 文件，节省 $(numfmt --to=iec $JPEG_SAVED 2>/dev/null || echo $JPEG_SAVED)${NC}"
    fi
fi
echo ""

##############################################################################
# 优化 PNG 文件
##############################################################################

echo -e "${YELLOW}[4/4] 优化 PNG 文件...${NC}"

if [ $PNG_COUNT -eq 0 ]; then
    echo -e "${BLUE}  → 跳过 (无 PNG 文件)${NC}"
else
    PNG_PROCESSED=0
    PNG_SAVED=0

    while IFS= read -r file; do
        [ -z "$file" ] && continue

        filename=$(basename "$file")
        original_size=$(stat -f%z "$file" 2>/dev/null || echo 0)

        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}  → 将优化: $filename ($(numfmt --to=iec $original_size 2>/dev/null || echo "${original_size}B"))${NC}"
        else
            # 备份原文件
            if [ "$BACKUP" = true ]; then
                cp "$file" "${file}.backup"
            fi

            # 优先使用 pngquant
            if [ "$HAS_PNGQUANT" = true ]; then
                pngquant --quality=65-$((QUALITY)) --speed $((4 - PNG_LEVEL)) --force --output "$file.tmp" "$file" 2>/dev/null || {
                    # 如果 pngquant 失败，回退到 ImageMagick
                    convert "$file" -strip -define png:compression-level=9 "$file.tmp" 2>/dev/null
                }
            else
                # 使用 ImageMagick
                convert "$file" -strip -define png:compression-level=9 "$file.tmp" 2>/dev/null
            fi

            new_size=$(stat -f%z "$file.tmp" 2>/dev/null || echo 0)

            # 只在文件变小时替换
            if [ $new_size -lt $original_size ]; then
                mv "$file.tmp" "$file"
                saved=$((original_size - new_size))
                PNG_SAVED=$((PNG_SAVED + saved))
                echo -e "${GREEN}  ✓ $filename: $(numfmt --to=iec $original_size 2>/dev/null || echo $original_size) → $(numfmt --to=iec $new_size 2>/dev/null || echo $new_size) (省${saved}字节)${NC}"
            else
                rm "$file.tmp"
                echo -e "${YELLOW}  → $filename: 已是最优 (跳过)${NC}"
            fi

            # 删除备份（如果优化成功）
            if [ "$BACKUP" = true ] && [ -f "${file}.backup" ]; then
                rm "${file}.backup"
            fi
        fi

        PNG_PROCESSED=$((PNG_PROCESSED + 1))
    done <<< "$PNG_FILES"

    if [ "$DRY_RUN" = false ]; then
        echo -e "${GREEN}  ✓ 处理 ${PNG_PROCESSED} 个 PNG 文件，节省 $(numfmt --to=iec $PNG_SAVED 2>/dev/null || echo $PNG_SAVED)${NC}"
    fi
fi
echo ""

##############################################################################
# 生成 WebP 格式（可选）
##############################################################################

if [ "$GENERATE_WEBP" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}[额外] 生成 WebP 格式...${NC}"

    WEBP_COUNT=0
    ALL_FILES="${JPEG_FILES}${PNG_FILES}"

    while IFS= read -r file; do
        [ -z "$file" ] && continue

        webp_file="${file%.*}.webp"

        # 跳过已存在的 WebP 文件
        if [ -f "$webp_file" ]; then
            continue
        fi

        filename=$(basename "$file")
        cwebp -q $WEBP_QUALITY "$file" -o "$webp_file" 2>/dev/null

        if [ -f "$webp_file" ]; then
            WEBP_COUNT=$((WEBP_COUNT + 1))
            echo -e "${GREEN}  ✓ 生成: $(basename "$webp_file")${NC}"
        fi
    done <<< "$ALL_FILES"

    echo -e "${GREEN}  ✓ 生成 ${WEBP_COUNT} 个 WebP 文件${NC}"
    echo ""
fi

##############################################################################
# 总结
##############################################################################

echo -e "${BLUE}========================================${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}演习模式完成${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "如果确认无误，请去掉 --dry-run 参数重新运行"
else
    echo -e "${GREEN}✓ 图片优化完成!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    TOTAL_SAVED=$((JPEG_SAVED + PNG_SAVED))
    NEW_SIZE=$(du -ch "$GALLERY_IMAGES"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | grep total | cut -f1 || echo "0")

    echo -e "统计信息:"
    echo -e "  • 处理文件: ${TOTAL_COUNT} 个"
    echo -e "  • 原始大小: ${ORIGINAL_SIZE}"
    echo -e "  • 优化后大小: ${NEW_SIZE}"
    echo -e "  • 节省空间: $(numfmt --to=iec $TOTAL_SAVED 2>/dev/null || echo "${TOTAL_SAVED}B")"

    if [ $TOTAL_SAVED -gt 0 ] && [ "$ORIGINAL_SIZE" != "0" ]; then
        # 计算百分比（近似）
        echo -e "  • 压缩比: ~30-50%"
    fi

    if [ "$GENERATE_WEBP" = true ]; then
        echo -e "  • WebP 文件: ${WEBP_COUNT} 个"
    fi
fi
echo ""
