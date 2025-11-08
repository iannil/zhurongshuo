# 仓库开发指南 (GEMINI.md)

## 1. 项目概览

本项目是一个基于 [Hugo](https://gohugo.io/) 静态网站生成器构建的个人博客，名为“祝融说”(zhurongshuo.com)。

- **核心内容**: 博客主要探讨中国哲学，特别是《道德经》的解读以及原创的哲学思想体系。
- **技术栈**: Hugo (扩展版), Docker, Caddy, GitHub Actions。
- **主题**: 使用名为 "zozo" 的主题。
- **部署地址**: [https://zhurongshuo.com/](https://zhurongshuo.com/)

## 2. 项目结构

```
.
├── content/         # Markdown 源文件
│   ├── posts/       # 主要博文 (按年月组织)
│   ├── core/        # 核心哲学框架
│   ├── kachuai/dao/ # 《道德经》解读
│   └── ...          # 其他分类
├── layouts/         # 自定义模板 (覆盖主题)
├── static/          # 静态资源 (图片, CSS, JS)
├── themes/zozo/     # 主题源码
├── docs/            # Hugo 生成的静态站点 (用于部署)
├── archetypes/      # 内容模板
├── config.toml      # Hugo 主配置文件
├── Dockerfile       # 生产环境 Docker 配置
├── docker-compose.yml # Docker Compose 配置
├── deploy.sh        # 自动化部署脚本
└── .github/workflows/hugo.yml # GitHub Actions CI/CD 配置
```

- **内容核心**: 所有文章内容均存放于 `content/` 目录。
- **布局定制**: 优先在根目录的 `layouts/` 中覆写或扩展主题功能，避免直接修改 `themes/zozo/`。
- **静态资源**: 图片、PDF 等共享资源应存放于 `static/assets/`，并通过绝对路径 (`/assets/...`) 引用。
- **构建输出**: Hugo 会将生成的网站发布到 `docs/` 目录，请勿手动编辑此目录。

## 3. 开发与构建命令

### 本地开发

```bash
# 启动本地开发服务器 (实时刷新)
# 访问: http://localhost:1313
hugo server -E -F --watch
```

### 站点构建

```bash
# 标准构建 (输出到 docs/ 目录)
hugo

# 生产构建 (压缩并清理)
hugo --minify --gc
```

### Docker 环境

```bash
# 启动容器化的开发环境 (带热重载)
docker-compose up dev

# 启动生产环境容器
docker-compose up -d prod

# 查看生产环境日志
docker-compose logs -f prod
```

## 4. 内容工作流

### 创建新文章

使用 Hugo 命令快速创建符合规范的草稿文件：

```bash
hugo new posts/2025/11/example.md
```

### Front Matter 规范

所有文章必须包含以下 YAML Front Matter 字段，并确保时区为 `+08:00`：

```yaml
---
title: "文章标题"
date: 2025-11-01T10:00:00+08:00
hidden: false      # 是否在列表页隐藏
draft: false       # 是否为草稿
tags: ["标签1", "标签2"]
keywords: ["关键词1", "关键词2"]
description: "文章摘要 (可选)"
slug: "url-slug"   # 自定义 URL 片段
---
```

### 内容导出

项目包含一个工具脚本，可将所有文章的日期和最后一段内容导出为 CSV 文件：

```bash
./export.sh
```

## 5. 部署流程

项目支持自动化和手动两种部署方式。

### 自动化部署 (GitHub Actions)

- **触发**: 当 `master` 分支有新的 `push` 时自动触发。
- **流程**:
  1. 使用 `hugo:nightly` 镜像构建站点。
  2. 将 `docs/` 目录下的构建产物部署到 GitHub Pages。

### 手动部署

执行项目根目录的部署脚本，它会自动完成拉取、构建、提交和推送的全过程：

```bash
./deploy.sh
```

**注意**: 执行此脚本前，请确保本地没有未提交的冲突。

## 6. 代码风格与提交规范

### 代码风格

- **Markdown**: 使用 UTF-8 编码。
- **模板**: Go 模板和短代码使用两个空格缩进。
- **命名**: 文件名使用小写连字符，如 `2025-11-01-my-post.md`。发布后避免修改 `slug`。

### 提交规范

- **格式**: 遵循仓库现有的时间戳风格: `YYYYMMDDHHMMSS on [os]`。
  - 示例: `20251101153000 on macos`
- **原子性**: 保持单次提交聚焦于一个独立的功能或修复。
- **合并请求 (PR)**:
  - 清晰描述改动范围和目的。
  - 如果涉及 UI 变动，请附上截图。
  - 明确说明是否已在本地更新 `docs/` 目录。

## 7. 关键技术细节

- **Hugo 版本**: CI/CD 使用 `v0.138.0` (扩展版)，本地建议使用兼容版本。
- **Web 服务器**: 生产环境使用 Caddy 2.9.1，并配置了 HSTS 和安全头。
- **数学公式**: 已启用 MathJax 支持。
- **时区**: 所有日期和时间戳均使用中国标准时间 (CST, `+08:00`)。