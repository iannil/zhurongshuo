#!/bin/bash

##############################################################################
# Gallery Migration Script
# 将现有图库内容从扁平结构迁移到按年月分组的目录结构
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

# 默认参数
DRY_RUN=false
FIX_DATES_FIRST=false
CREATE_BACKUP=true

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

将图库从扁平结构迁移到按年月分组的目录结构

选项:
  --fix-dates             迁移前先修复占位日期
  --no-backup             不创建备份
  -n, --dry-run           演习模式，不实际移动文件
  -h, --help              显示此帮助信息

迁移说明:
  • 根据文件的 date 字段将文件移动到 YYYY/MM/ 目录
  • _index.md 文件保持在根目录
  • 会创建备份到 content/gallery.backup/

示例:
  $0                      # 直接迁移
  $0 --fix-dates          # 先修复日期再迁移
  $0 --dry-run            # 预览迁移操作

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix-dates)
            FIX_DATES_FIRST=true
            shift
            ;;
        --no-backup)
            CREATE_BACKUP=false
            shift
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  图库迁移工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "配置:"
echo -e "  源目录: ${GALLERY_CONTENT}"
echo -e "  目标结构: 按年月分组 (YYYY/MM/)"
echo -e "  修复日期: ${FIX_DATES_FIRST}"
echo -e "  创建备份: ${CREATE_BACKUP}"
echo -e "  演习模式: ${DRY_RUN}"
echo ""

##############################################################################
# 辅助函数
##############################################################################

# 从 markdown 文件提取 frontmatter 字段
get_frontmatter_field() {
    local file="$1"
    local field="$2"

    awk -v field="$field" '
        /^---$/ { if (in_fm) exit; in_fm=!in_fm; next }
        in_fm && $0 ~ "^" field ":" {
            sub("^" field ": *\"?", "")
            sub("\"? *$", "")
            print
            exit
        }
    ' "$file"
}

# 提取年月
extract_year_month() {
    local date_str="$1"

    year=$(echo "$date_str" | grep -oE '^[0-9]{4}')
    month=$(echo "$date_str" | grep -oE '-[0-9]{2}-' | tr -d '-')

    if [ -z "$year" ] || [ -z "$month" ]; then
        # 如果日期格式不正确，使用默认值
        year="2025"
        month="11"
    fi

    echo "$year/$month"
}

##############################################################################
# 检查当前状态
##############################################################################

echo -e "${YELLOW}[1/4] 检查当前状态...${NC}"

# 统计现有文件
TOTAL_FILES=$(find "$GALLERY_CONTENT" -maxdepth 1 -type f -name "*.md" ! -name "_index.md" 2>/dev/null | wc -l | tr -d ' ')

if [ $TOTAL_FILES -eq 0 ]; then
    echo -e "${YELLOW}没有需要迁移的文件${NC}"
    exit 0
fi

# 检查是否已有年月目录
EXISTING_YEAR_DIRS=$(find "$GALLERY_CONTENT" -maxdepth 1 -type d -name "20*" 2>/dev/null | wc -l | tr -d ' ')

if [ $EXISTING_YEAR_DIRS -gt 0 ]; then
    echo -e "${YELLOW}  ! 检测到已存在年份目录，可能已部分迁移${NC}"
    read -p "    继续迁移？ (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo -e "${GREEN}  ✓ 找到 ${TOTAL_FILES} 个文件需要迁移${NC}"
echo ""

##############################################################################
# 修复日期（可选）
##############################################################################

if [ "$FIX_DATES_FIRST" = true ]; then
    echo -e "${YELLOW}[2/4] 修复占位日期...${NC}"

    # 检查是否有 gallery-meta.sh 脚本
    META_SCRIPT="${PROJECT_ROOT}/scripts/gallery-meta.sh"

    if [ -f "$META_SCRIPT" ]; then
        if [ "$DRY_RUN" = true ]; then
            "$META_SCRIPT" fix-dates --dry-run
        else
            "$META_SCRIPT" fix-dates
        fi
    else
        echo -e "${YELLOW}  ! gallery-meta.sh 不存在，跳过日期修复${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}[2/4] 跳过日期修复${NC}"
    echo ""
fi

##############################################################################
# 创建备份
##############################################################################

if [ "$CREATE_BACKUP" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}[3/4] 创建备份...${NC}"

    BACKUP_DIR="${PROJECT_ROOT}/content/gallery.backup.$(date +%Y%m%d_%H%M%S)"

    cp -r "$GALLERY_CONTENT" "$BACKUP_DIR"
    echo -e "${GREEN}  ✓ 备份创建于: $BACKUP_DIR${NC}"
    echo ""
else
    echo -e "${YELLOW}[3/4] 跳过备份${NC}"
    echo ""
fi

##############################################################################
# 执行迁移
##############################################################################

echo -e "${YELLOW}[4/4] 执行迁移...${NC}"

MIGRATED=0
SKIPPED=0
ERRORS=0

# 遍历所有 markdown 文件
for md_file in "$GALLERY_CONTENT"/*.md; do
    [ -f "$md_file" ] || continue

    filename=$(basename "$md_file")

    # 跳过 _index.md
    if [ "$filename" = "_index.md" ]; then
        continue
    fi

    echo -e "${BLUE}处理: $filename${NC}"

    # 提取日期
    date_value=$(get_frontmatter_field "$md_file" "date")

    if [ -z "$date_value" ]; then
        echo -e "${RED}  ✗ 无法提取日期，跳过${NC}"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    # 计算目标目录
    year_month=$(extract_year_month "$date_value")
    target_dir="$GALLERY_CONTENT/$year_month"
    target_file="$target_dir/$filename"

    # 检查目标文件是否已存在
    if [ -f "$target_file" ]; then
        echo -e "${YELLOW}  ! 目标文件已存在，跳过${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}  → 将移动到: $year_month/$filename${NC}"
    else
        # 创建目标目录
        mkdir -p "$target_dir"

        # 复制文件到新位置
        cp "$md_file" "$target_file"

        # 删除原文件
        rm "$md_file"

        echo -e "${GREEN}  ✓ 已移动到: $year_month/$filename${NC}"
    fi

    MIGRATED=$((MIGRATED + 1))
    echo ""
done

##############################################################################
# 总结
##############################################################################

echo -e "${BLUE}========================================${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}演习模式完成${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "预计迁移: ${MIGRATED} 个"
    echo -e "预计跳过: ${SKIPPED} 个"
    echo -e "错误: ${ERRORS} 个"
    echo ""
    echo -e "如果确认无误，请去掉 --dry-run 参数重新运行"
else
    echo -e "${GREEN}✓ 迁移完成!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "统计信息:"
    echo -e "  • 成功迁移: ${MIGRATED} 个"
    echo -e "  • 跳过: ${SKIPPED} 个"
    echo -e "  • 错误: ${ERRORS} 个"

    if [ "$CREATE_BACKUP" = true ]; then
        echo -e "  • 备份位置: $BACKUP_DIR"
    fi

    echo ""
    echo -e "${YELLOW}后续步骤:${NC}"
    echo -e "  1. 验证迁移: scripts/gallery-meta.sh validate"
    echo -e "  2. 测试站点: hugo server"
    echo -e "  3. 确认无误后提交: git add . && git commit -m 'refactor: 重组图库为年月目录结构'"
    echo ""

    # 显示目录结构
    echo -e "${BLUE}新的目录结构:${NC}"
    tree -L 2 "$GALLERY_CONTENT" 2>/dev/null || find "$GALLERY_CONTENT" -type d -maxdepth 2 | sort
fi
echo ""
