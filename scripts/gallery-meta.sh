#!/bin/bash

##############################################################################
# Gallery Metadata Management Tool
# 管理图库 markdown 文件的元数据
# 支持从 EXIF 提取日期、批量更新、验证完整性
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
ACTION=""
DRY_RUN=false
FIX_DATES=false
DATE_SOURCE="file"  # 默认使用文件时间（exif对AI生成图片无效）

# 帮助信息
show_help() {
    cat << EOF
用法: $0 <action> [选项]

图库元数据管理工具

操作:
  validate                验证所有 gallery 文件的完整性
  fix-dates               修复占位日期（从 EXIF 或文件时间）
  report                  生成详细报告
  update-date FILE DATE   更新指定文件的日期

选项:
  -n, --dry-run           演习模式，不实际修改文件
  --source SOURCE         日期来源: exif, file, manual, keep (默认: file)
                          file - 使用文件修改时间（推荐，适用于AI生成图片）
                          exif - 使用图片EXIF数据（仅对真实照片有效）
                          keep - 保持当前markdown中的日期不变
                          manual - 手动输入每个文件的日期
  -h, --help              显示此帮助信息

示例:
  $0 validate                      # 验证所有文件
  $0 fix-dates                     # 使用文件时间修复日期（默认）
  $0 fix-dates --source exif       # 使用EXIF修复日期（真实照片）
  $0 fix-dates --source file       # 从文件时间修复日期
  $0 fix-dates --source keep       # 保持现有日期不变
  $0 report                        # 生成报告
  $0 update-date img-0002.md "2023-05-15T14:30:00+08:00"

EOF
}

# 解析命令行参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

ACTION="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --source)
            DATE_SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # 保存额外参数
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  图库元数据管理工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查目录是否存在
if [ ! -d "$GALLERY_CONTENT" ]; then
    echo -e "${RED}错误: 目录不存在: $GALLERY_CONTENT${NC}"
    exit 1
fi

##############################################################################
# 辅助函数
##############################################################################

# 从 markdown 文件提取 frontmatter 字段
get_frontmatter_field() {
    local file="$1"
    local field="$2"

    # 提取 frontmatter 部分并查找字段
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

# 更新 frontmatter 字段
update_frontmatter_field() {
    local file="$1"
    local field="$2"
    local value="$3"

    # 使用 sed 更新字段
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|^${field}:.*|${field}: \"${value}\"|" "$file"
    else
        # Linux
        sed -i "s|^${field}:.*|${field}: \"${value}\"|" "$file"
    fi
}

# 从图片 EXIF 提取日期
get_exif_date() {
    local image_file="$1"

    # 检查是否安装了 exiftool
    if ! command -v exiftool &> /dev/null; then
        return 1
    fi

    # 提取 DateTimeOriginal，添加超时保护
    local output=$(timeout 3 exiftool -s -s -s -DateTimeOriginal "$image_file" 2>/dev/null || true)

    if [ -n "$output" ]; then
        # 转换格式: "2024:03:21 21:10:27" -> "2024-03-21T21:10:27+08:00"
        local date_str=$(echo "$output" | sed 's/:/-/; s/:/-/; s/ /T/' | sed 's/$/+08:00/')
        echo "$date_str"
        return 0
    fi

    return 1
}

# 从文件修改时间获取日期
get_file_date() {
    local file="$1"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        stat -f "%Sm" -t "%Y-%m-%dT%H:%M:%S+08:00" "$file"
    else
        # Linux
        date -r "$file" "+%Y-%m-%dT%H:%M:%S+08:00"
    fi
}

##############################################################################
# 验证功能
##############################################################################

validate_gallery() {
    echo -e "${YELLOW}验证图库文件...${NC}"
    echo ""

    local total=0
    local valid=0
    local missing_images=0
    local placeholder_dates=0
    local empty_descriptions=0
    local empty_tags=0
    local issues=()

    # 遍历所有 markdown 文件
    for md_file in "$GALLERY_CONTENT"/*.md; do
        [ -f "$md_file" ] || continue
        [ "$(basename "$md_file")" = "_index.md" ] && continue

        total=$((total + 1))
        filename=$(basename "$md_file")
        has_issue=false

        # 检查 featured_image
        featured_image=$(get_frontmatter_field "$md_file" "featured_image")
        if [ -z "$featured_image" ]; then
            issues+=("${RED}✗${NC} $filename: 缺少 featured_image")
            missing_images=$((missing_images + 1))
            has_issue=true
        else
            # 移除开头的 /
            image_path="${featured_image#/}"
            full_image_path="${PROJECT_ROOT}/static/${image_path}"

            if [ ! -f "$full_image_path" ]; then
                issues+=("${RED}✗${NC} $filename: 图片文件不存在: $featured_image")
                missing_images=$((missing_images + 1))
                has_issue=true
            fi
        fi

        # 检查日期是否为占位符
        date_value=$(get_frontmatter_field "$md_file" "date")
        if [[ "$date_value" =~ T00:00:00 ]]; then
            issues+=("${YELLOW}!${NC} $filename: 使用占位日期: $date_value")
            placeholder_dates=$((placeholder_dates + 1))
            has_issue=true
        fi

        # 检查 description
        description=$(get_frontmatter_field "$md_file" "description")
        if [ -z "$description" ] || [ "$description" = '""' ]; then
            empty_descriptions=$((empty_descriptions + 1))
        fi

        # 检查 tags
        tags=$(get_frontmatter_field "$md_file" "tags")
        if [ -z "$tags" ] || [ "$tags" = "[]" ]; then
            empty_tags=$((empty_tags + 1))
        fi

        if [ "$has_issue" = false ]; then
            valid=$((valid + 1))
        fi
    done

    # 输出问题
    if [ ${#issues[@]} -gt 0 ]; then
        echo -e "${YELLOW}发现的问题:${NC}"
        printf '%s\n' "${issues[@]}"
        echo ""
    fi

    # 输出统计
    echo -e "${BLUE}========================================${NC}"
    echo -e "验证结果:"
    echo -e "  • 总文件数: $total"
    echo -e "  • 完全正常: $valid"
    echo -e "  • 缺失图片: $missing_images"
    echo -e "  • 占位日期: $placeholder_dates"
    echo -e "  • 空描述: $empty_descriptions"
    echo -e "  • 空标签: $empty_tags"
    echo ""

    if [ $missing_images -eq 0 ] && [ $placeholder_dates -eq 0 ]; then
        echo -e "${GREEN}✓ 所有文件验证通过!${NC}"
    else
        echo -e "${YELLOW}建议运行 'fix-dates' 修复占位日期${NC}"
    fi
}

##############################################################################
# 修复日期功能
##############################################################################

fix_dates() {
    echo -e "${YELLOW}修复占位日期...${NC}"
    echo -e "日期来源: ${DATE_SOURCE}"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[演习模式]${NC}"
    fi
    echo ""

    # 检查 exiftool（如果需要）
    if [ "$DATE_SOURCE" = "exif" ]; then
        if ! command -v exiftool &> /dev/null; then
            echo -e "${RED}错误: exiftool 未安装${NC}"
            echo -e "${YELLOW}安装方法:${NC}"
            echo -e "  brew install exiftool"
            echo ""
            echo -e "或者使用文件时间作为日期来源:"
            echo -e "  $0 fix-dates --source file"
            exit 1
        fi
    fi

    local total=0
    local fixed=0
    local skipped=0

    # 遍历所有 markdown 文件
    for md_file in "$GALLERY_CONTENT"/*.md; do
        [ -f "$md_file" ] || continue
        [ "$(basename "$md_file")" = "_index.md" ] && continue

        filename=$(basename "$md_file")

        # 检查日期是否为占位符
        date_value=$(get_frontmatter_field "$md_file" "date")
        if [[ ! "$date_value" =~ T00:00:00 ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        total=$((total + 1))

        # 获取图片路径
        featured_image=$(get_frontmatter_field "$md_file" "featured_image")
        image_path="${featured_image#/}"
        full_image_path="${PROJECT_ROOT}/static/${image_path}"

        new_date=""

        # 根据来源获取日期
        case "$DATE_SOURCE" in
            keep)
                # 保持现有日期，不修改
                echo -e "${BLUE}  → $filename: 保持现有日期 $date_value${NC}"
                continue
                ;;
            exif)
                if [ -f "$full_image_path" ]; then
                    new_date=$(get_exif_date "$full_image_path")
                    if [ -z "$new_date" ]; then
                        # EXIF 失败，回退到文件时间
                        new_date=$(get_file_date "$md_file")
                        echo -e "${YELLOW}  ! $filename: EXIF无日期，使用文件时间${NC}"
                    fi
                else
                    # 图片不存在，使用 markdown 文件时间
                    new_date=$(get_file_date "$md_file")
                    echo -e "${YELLOW}  ! $filename: 图片不存在，使用markdown文件时间${NC}"
                fi
                ;;
            file)
                new_date=$(get_file_date "$md_file")
                ;;
            manual)
                echo -e "${BLUE}  ? $filename 当前日期: $date_value${NC}"
                read -p "    输入新日期 (格式: YYYY-MM-DDTHH:MM:SS+08:00, 回车跳过): " new_date
                [ -z "$new_date" ] && continue
                ;;
        esac

        # 更新日期
        if [ -n "$new_date" ]; then
            if [ "$DRY_RUN" = true ]; then
                echo -e "${BLUE}  → $filename: $date_value → $new_date${NC}"
            else
                update_frontmatter_field "$md_file" "date" "$new_date"
                echo -e "${GREEN}  ✓ $filename: 已更新为 $new_date${NC}"
            fi
            fixed=$((fixed + 1))
        fi
    done

    echo ""
    echo -e "${BLUE}========================================${NC}"

    if [ "$DATE_SOURCE" = "keep" ]; then
        echo -e "统计结果:"
        echo -e "  • 占位日期文件: $total"
        echo -e "  • 保持不变: $total"
        echo -e "  • 正常日期文件: $skipped"
        echo ""
        echo -e "${BLUE}使用 --source keep 保持所有日期不变${NC}"
    else
        echo -e "修复结果:"
        echo -e "  • 需要修复: $total"
        echo -e "  • 已修复: $fixed"
        echo -e "  • 已跳过: $skipped"
        echo ""

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}这是演习模式，如确认无误请去掉 --dry-run 参数${NC}"
        else
            echo -e "${GREEN}✓ 日期修复完成!${NC}"
        fi
    fi
}

##############################################################################
# 生成报告功能
##############################################################################

generate_report() {
    echo -e "${YELLOW}生成图库报告...${NC}"
    echo ""

    local total=0
    local series_files=0
    local photo_files=0
    local art_files=0

    # 使用临时文件代替关联数组（兼容 bash 3.2）
    local year_stats=$(mktemp)
    local series_stats=$(mktemp)

    for md_file in "$GALLERY_CONTENT"/*.md; do
        [ -f "$md_file" ] || continue
        [ "$(basename "$md_file")" = "_index.md" ] && continue

        total=$((total + 1))
        filename=$(basename "$md_file" .md)

        # 分类统计
        if [[ "$filename" =~ ^img- ]] || [[ "$filename" =~ ^IMG ]]; then
            photo_files=$((photo_files + 1))
        elif [[ "$filename" =~ -[0-9]+$ ]]; then
            series_files=$((series_files + 1))
        else
            art_files=$((art_files + 1))
        fi

        # 按年份统计
        date_value=$(get_frontmatter_field "$md_file" "date")
        year=$(echo "$date_value" | grep -oE '^[0-9]{4}')
        if [ -n "$year" ]; then
            echo "$year" >> "$year_stats"
        fi

        # 识别系列
        if [[ "$filename" =~ ^(.+)-[0-9]+$ ]]; then
            series_name="${BASH_REMATCH[1]}"
            echo "$series_name" >> "$series_stats"
        fi
    done

    # 输出报告
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}图库统计报告${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    echo -e "${GREEN}总览:${NC}"
    echo -e "  • 总文件数: $total"
    echo -e "  • 照片文件: $photo_files"
    echo -e "  • 系列作品: $series_files"
    echo -e "  • 其他作品: $art_files"
    echo ""

    # 按年份分布
    if [ -s "$year_stats" ]; then
        echo -e "${GREEN}按年份分布:${NC}"
        sort "$year_stats" | uniq -c | sort -k2 | while read count year; do
            echo -e "  • $year: $count 个"
        done
        echo ""
    fi

    # 系列作品
    if [ -s "$series_stats" ]; then
        echo -e "${GREEN}系列作品:${NC}"
        sort "$series_stats" | uniq -c | sort -k2 | while read count series_name; do
            echo -e "  • $series_name: $count 个"

            # 检查系列连续性
            missing=""
            for i in $(seq 1 $count); do
                if [ ! -f "$GALLERY_CONTENT/${series_name}-${i}.md" ]; then
                    missing="$missing $i"
                fi
            done

            if [ -n "$missing" ]; then
                echo -e "    ${YELLOW}缺失编号:$missing${NC}"
            fi
        done
        echo ""
    fi

    # 特殊文件
    echo -e "${GREEN}特殊文件:${NC}"
    local has_special=false
    for md_file in "$GALLERY_CONTENT"/*.md; do
        filename=$(basename "$md_file" .md)

        # 检查是否为哈希文件名
        if [[ "$filename" =~ ^[0-9a-f]{32}$ ]]; then
            title=$(get_frontmatter_field "$md_file" "title")
            echo -e "  • ${YELLOW}哈希文件名:${NC} $filename (标题: $title)"
            has_special=true
        fi
    done

    if [ "$has_special" = false ]; then
        echo -e "  无"
    fi
    echo ""

    # 清理临时文件
    rm -f "$year_stats" "$series_stats"
}

##############################################################################
# 更新单个文件日期
##############################################################################

update_single_date() {
    local file="$1"
    local new_date="$2"

    if [ -z "$file" ] || [ -z "$new_date" ]; then
        echo -e "${RED}错误: 缺少参数${NC}"
        echo "用法: $0 update-date <文件名> <新日期>"
        exit 1
    fi

    # 构建完整路径
    if [[ "$file" != /* ]]; then
        file="$GALLERY_CONTENT/$file"
    fi

    if [ ! -f "$file" ]; then
        echo -e "${RED}错误: 文件不存在: $file${NC}"
        exit 1
    fi

    # 验证日期格式
    if [[ ! "$new_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+08:00$ ]]; then
        echo -e "${RED}错误: 日期格式不正确${NC}"
        echo "正确格式: YYYY-MM-DDTHH:MM:SS+08:00"
        echo "示例: 2023-05-15T14:30:00+08:00"
        exit 1
    fi

    old_date=$(get_frontmatter_field "$file" "date")
    update_frontmatter_field "$file" "date" "$new_date"

    echo -e "${GREEN}✓ 日期已更新:${NC}"
    echo -e "  文件: $(basename "$file")"
    echo -e "  旧日期: $old_date"
    echo -e "  新日期: $new_date"
}

##############################################################################
# 主逻辑
##############################################################################

case "$ACTION" in
    validate)
        validate_gallery
        ;;
    fix-dates)
        fix_dates
        ;;
    report)
        generate_report
        ;;
    update-date)
        update_single_date "${EXTRA_ARGS[0]}" "${EXTRA_ARGS[1]}"
        ;;
    *)
        echo -e "${RED}错误: 未知操作 '$ACTION'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
