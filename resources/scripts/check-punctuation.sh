#!/bin/bash
# 文件名: check-punctuation.sh

echo "=== 标点符号问题统计 ==="
echo ""

cd /Users/iannil/Code/zhurongshuo/content/books/

echo "1. 全角括号使用情况:"
grep -r "（\|）" . --include="*.md" | wc -l

echo "2. 中英文逗号混用（估算）:"
grep -r -P "[\u4e00-\u9fa5]," . --include="*.md" 2>/dev/null | wc -l

echo "3. 多余空格:"
grep -r "  " . --include="*.md" | wc -l

echo "4. 冒号后多余空格:"
grep -r -E "：\s+|:\s{2,}" . --include="*.md" | wc -l
