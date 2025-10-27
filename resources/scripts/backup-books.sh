#!/bin/bash
# 文件名: backup-books.sh

BACKUP_DIR="/Users/iannil/Code/zhurongshuo/content/books_backup_$(date +%Y%m%d_%H%M%S)"

echo "=== 创建备份 ==="
echo "备份目录: $BACKUP_DIR"

cp -r /Users/iannil/Code/zhurongshuo/content/books "$BACKUP_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 备份成功！"
    echo "备份位置: $BACKUP_DIR"
else
    echo "❌ 备份失败！"
    exit 1
fi
