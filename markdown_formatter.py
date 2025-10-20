#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown 格式标准化工具
用于标准化 content/books 文件夹下的 Markdown 文件
"""

import os
import re
from pathlib import Path
from typing import List


class MarkdownFormatter:
    def __init__(self):
        self.front_matter_pattern = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)

    def format_file(self, file_path: str) -> bool:
        """格式化单个 Markdown 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取 front matter 和内容
            match = self.front_matter_pattern.match(content)
            if match:
                front_matter = match.group(0)
                body = content[match.end():]
            else:
                front_matter = ""
                body = content

            # 格式化内容
            formatted_body = self.format_content(body)

            # 重新组合
            formatted_content = front_matter + formatted_body

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)

            return True
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return False

    def format_content(self, content: str) -> str:
        """格式化 Markdown 内容"""
        lines = content.split('\n')
        formatted_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # 1. 清理行尾空格
            line = line.rstrip()

            # 2. 处理标题
            if line.startswith('#'):
                # 确保标题前有空行（除非是第一行）
                if formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')

                # 调整标题层级：将 #### 作为文档第一级标题的改为 ##
                # 同时保持相对层级
                formatted_lines.append(self.normalize_heading(line))

                # 确保标题后有空行
                if i + 1 < len(lines) and lines[i + 1].strip() != '':
                    formatted_lines.append('')

            # 3. 处理列表
            elif self.is_list_item(line):
                # 确保列表前有空行
                if formatted_lines and formatted_lines[-1].strip() != '' and not self.is_list_item(formatted_lines[-1]):
                    formatted_lines.append('')

                formatted_lines.append(line)

                # 检查列表后是否需要空行
                if i + 1 < len(lines) and lines[i + 1].strip() != '' and not self.is_list_item(lines[i + 1]):
                    # 需要在列表后添加空行的标记
                    pass

            # 4. 处理引用
            elif line.startswith('>'):
                # 确保引用前有空行
                if formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')

                formatted_lines.append(line)

                # 确保引用后有空行
                if i + 1 < len(lines) and lines[i + 1].strip() != '' and not lines[i + 1].startswith('>'):
                    formatted_lines.append('')

            # 5. 处理普通段落
            else:
                # 处理空行：确保段落间只有一个空行
                if line.strip() == '':
                    if not formatted_lines or formatted_lines[-1].strip() != '':
                        formatted_lines.append('')
                else:
                    formatted_lines.append(line)

            i += 1

        # 清理结尾多余空行
        while formatted_lines and formatted_lines[-1].strip() == '':
            formatted_lines.pop()

        # 确保文件以一个换行符结尾
        result = '\n'.join(formatted_lines)
        if result and not result.endswith('\n'):
            result += '\n'

        return result

    def normalize_heading(self, line: str) -> str:
        """标准化标题层级"""
        # 提取标题级别和内容
        match = re.match(r'^(#+)\s+(.*)', line)
        if not match:
            return line

        hashes = match.group(1)
        content = match.group(2)
        level = len(hashes)

        # 如果是 #### 开头，可能需要调整
        # 这里我们保持原有层级，因为无法判断全局上下文
        # 仅确保格式正确：# 和文字之间有一个空格
        return f"{hashes} {content}"

    def is_list_item(self, line: str) -> bool:
        """判断是否为列表项"""
        stripped = line.lstrip()
        # 无序列表
        if stripped.startswith(('- ', '* ', '+ ')):
            return True
        # 有序列表
        if re.match(r'^\d+\.\s', stripped):
            return True
        return False


def main():
    """主函数"""
    # 设置根目录
    books_dir = Path('/Users/iannil/Code/zhurongshuo/content/books')

    if not books_dir.exists():
        print(f"目录不存在: {books_dir}")
        return

    # 获取所有 Markdown 文件
    md_files = list(books_dir.rglob('*.md'))
    total = len(md_files)

    print(f"找到 {total} 个 Markdown 文件")

    formatter = MarkdownFormatter()
    success_count = 0

    for i, file_path in enumerate(md_files, 1):
        print(f"处理 [{i}/{total}]: {file_path.relative_to(books_dir)}")
        if formatter.format_file(str(file_path)):
            success_count += 1

    print(f"\n完成！成功处理 {success_count}/{total} 个文件")


if __name__ == '__main__':
    main()
