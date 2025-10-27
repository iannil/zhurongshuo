#!/bin/bash
# 文件名: find-todos.sh

echo "=== 查找所有 TODO 标记 ==="
echo ""

cd /Users/iannil/Code/zhurongshuo/content/books/

grep -r -n "TODO\|FIXME\|待完成\|未完成" . --include="*.md" | while IFS=: read -r file line content; do
    echo "文件: $file"
    echo "行号: $line"
    echo "内容: $content"
    echo "---"
done

echo ""
echo "总计:"
grep -r "TODO\|FIXME\|待完成\|未完成" . --include="*.md" | wc -l
