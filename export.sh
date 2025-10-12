#!/bin/bash

# 脚本功能：
# 1. 递归查找指定目录下的所有 .md 文件。
# 2. 从每个文件中提取 YAML front-matter 中的 'date' 字段。
# 3. 提取 Markdown 内容的最后一个段落。
# 4. 清理段落内容（移除换行、特定符号等）。
# 5. 将结果按日期降序排序。
# 6. 将排序后的数据（日期, 段落）写入一个带BOM的UTF-8编码的CSV文件。
#
# 使用方法:
# 将此脚本保存为 process_posts.sh, 然后在终端中运行:
# chmod +x process_posts.sh
# ./process_posts.sh

# --- 配置 ---
# 源目录，包含Markdown文章
DOCS_DIR="./content/posts"
# 输出CSV文件的前缀和后缀
CSV_PREFIX="祝融说_副本"
CSV_SUFFIX=".csv"
# 最终的CSV文件名，格式如: 祝融说_副本20231225.csv
CSV_FILE="./archive/${CSV_PREFIX}$(date +%Y%m%d)${CSV_SUFFIX}"

# --- 主逻辑 ---

# 检查源目录是否存在
if [ ! -d "$DOCS_DIR" ]; then
    echo "错误: 目录 '$DOCS_DIR' 不存在。" >&2
    exit 1
fi

echo "正在从 '$DOCS_DIR' 目录中查找 .md 文件..."

# 使用 find 命令查找所有 .md 文件，然后通过一个循环来处理它们。
# 使用 process substitution 和一个临时变量来存储中间结果，这样更安全、高效。
# 中间格式为： "日期\t段落内容" (使用制表符分隔，便于后续处理)
processed_data=$(
    find "$DOCS_DIR" -type f -name "*.md" | while IFS= read -r file; do
        # 1. 提取 'date' 字段
        # 使用 grep 查找以 "date:" 开头的行，-m 1 确保只匹配第一个
        # 使用 sed 移除 "date: " 前缀和可能存在的引号
        date_val=$(grep -m 1 '^date:' "$file" | sed -E "s/^date: *['\"]?//; s/['\"]?$//")

        # 2. 提取最后一个段落
        # awk 是处理这种基于分隔符的文本块的完美工具。
        # RS='---' 将 "---" 设置为记录分隔符，这样文件就被分成了三块（或更多）。
        # 'END{print}' 会打印最后一个块，即正文内容。
        #
        # 3. 清理段落内容 (与PHP脚本的str_replace保持一致)
        # tr -d '\n\r'      : 移除所有换行符和回车符
        # sed -e '...'      : 执行多个替换操作
        #   s/> //g         : 移除 "> " (引用)
        #   s/\*//g         : 移除 "*" (列表项)
        #   s/<!--more-->//g: 移除 "<!--more-->"
        #   s/^[[:space:]]*//; s/[[:space:]]*$// : 移除开头和结尾的空白 (trim)
        last_paragraph=$(awk 'BEGIN{RS="---"} END{print}' "$file" | \
                         tr -d '\n\r' | \
                         sed -e 's/> //g' -e 's/\*//g' -e 's/<!--more-->//g' -e 's/^[[:space:]]*//; s/[[:space:]]*$//')

        # 如果成功提取了日期和内容，则输出
        if [ -n "$date_val" ] && [ -n "$last_paragraph" ]; then
            # 使用制表符作为临时分隔符，因为它在普通文本中不常见
            echo -e "$date_val\t$last_paragraph"
        fi
    done
)

# 检查是否处理了任何数据
if [ -z "$processed_data" ]; then
    echo "没有找到符合条件的文件或数据为空..."
    exit 0
fi

echo "数据提取完成，正在排序并生成CSV文件..."

# 4. 排序和写入CSV
# sort -r : 按整行进行反向（降序）排序。由于日期是ISO 8601格式，字符串排序等同于时间排序。
# awk       : 将制表符分隔的数据转换为RFC 4180兼容的CSV格式。
#   BEGIN{...} : 设置输入分隔符(FS)为制表符，输出分隔符(OFS)为逗号，定义引号变量Q。
#   {...}      : 对每一行进行处理。
#     f2 = $2  : 将第二列（段落）赋值给变量f2。
#     gsub(Q, Q Q, f2) : 将f2中所有的双引号替换为两个双引号，这是CSV转义规则。
#     if ($2 ~ /"|,/) : 如果原始段落内容包含双引号或逗号...
#     f2 = Q f2 Q    : ...则用双引号将整个字段包裹起来。
#     print $1, f2   : 打印第一列（日期）和处理后的第二列。

{
    # 写入UTF-8 BOM头，以确保Excel等软件能正确打开中文文件
    printf '\xEF\xBB\xBF'
    # 将已排序的数据流通过管道送入awk进行最终的CSV格式化
    echo "$processed_data" | sort -r | awk '
    BEGIN {
        FS="\t"
        OFS=","
        Q="\""
    }
    {
        # 第1个字段（日期）通常不需要处理
        f1 = $1

        # 第2个字段（段落）需要进行CSV转义
        f2 = $2
        # 1. 将字段内的所有双引号替换为两个双引号
        gsub(Q, Q Q, f2)
        # 2. 如果原始字段包含逗号或双引号，则整个字段用双引号包裹
        if ($2 ~ /"|,/) {
            f2 = Q f2 Q
        }
        print f1, f2
    }'
} > "$CSV_FILE" # 将所有输出重定向到最终的CSV文件

# 检查文件是否成功创建且非空
if [ -s "$CSV_FILE" ]; then
    echo "文件写入成功: $CSV_FILE"
else
    echo "文件写入失败..." >&2
    exit 1
fi

exit 0
