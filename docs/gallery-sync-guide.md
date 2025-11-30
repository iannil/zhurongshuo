# Gallery 图片同步指南

## 概述

本项目提供了自动化脚本，用于将 `static/images/gallery` 目录下的新增图片：

1. 增量上传到 Cloudflare R2 存储
2. 自动生成对应的 markdown 页面到 `content/gallery` 目录

## 使用方法

### 1. 添加新图片

将图片文件（支持 jpg, jpeg, png, gif, webp）添加到 `static/images/gallery` 目录：

```bash
cp your-image.png static/images/gallery/
```

### 2. 执行部署脚本

运行部署脚本，自动完成同步：

```bash
./deploy.sh
```

部署脚本会自动执行以下步骤：

1. 拉取最新代码
2. **同步 Gallery 图片并生成 Markdown**（新增功能）
3. 同步其他静态资源到 R2
4. 部署 Worker
5. 导出内容归档
6. 构建 Hugo 站点
7. 提交并推送更改

### 3. 单独运行同步脚本

如果只想同步图片而不执行完整部署：

```bash
bash scripts/sync-gallery.sh
```

## 工作原理

### 增量同步

脚本使用 `.r2-sync-record` 文件记录已同步的图片，避免重复上传：

- 首次运行：上传所有图片
- 后续运行：仅上传新增图片

### Markdown 生成

对于每个上传的图片，脚本会自动生成对应的 markdown 文件：

```yaml
---
title: "图片名称"
date: 2025-11-30T15:48:27+08:00
draft: false
type: "gallery"
featured_image: "/images/gallery/图片名称.png"
description: ""
tags: []
---
```

如果 markdown 文件已存在，脚本会跳过生成步骤。

## 配置要求

### 环境变量

脚本需要以下环境变量（在 `.env` 文件中配置）：

```bash
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_R2_API_TOKEN=your_r2_api_token
# 或者
CLOUDFLARE_API_TOKEN=your_api_token
```

### 依赖工具

- `wrangler`：Cloudflare CLI 工具
  ```bash
  npm install -g wrangler
  ```

## 输出示例

```
========================================
同步 Gallery 图片到 R2
========================================
[INFO] 扫描图片目录: static/images/gallery

[INFO] 上传到 R2: 新作品.png
[SUCCESS]   ✓ R2 上传成功: images/gallery/新作品.png
[SUCCESS]   ✓ 生成 Markdown: 新作品.md

[INFO] 跳过（已同步）: 旧作品.png

========================================
同步完成
========================================
[INFO] 统计信息：
[INFO]   总计图片: 30 个
[SUCCESS]   新上传: 1 个
[INFO]   跳过: 29 个（已同步）
[SUCCESS]   生成 Markdown: 1 个
```

## 文件结构

```
.
├── static/images/gallery/       # 图片源文件
│   ├── 作品1.png
│   └── 作品2.png
├── content/gallery/              # 自动生成的 markdown
│   ├── 作品1.md
│   └── 作品2.md
├── .r2-sync-record              # 同步记录（自动生成）
├── scripts/
│   └── sync-gallery.sh          # 同步脚本
└── deploy.sh                     # 部署脚本
```

## 故障排查

### 上传失败

如果遇到上传失败：

1. 检查环境变量配置
   ```bash
   source .env
   echo $CLOUDFLARE_ACCOUNT_ID
   echo $CLOUDFLARE_R2_API_TOKEN
   ```

2. 测试 wrangler 连接
   ```bash
   wrangler r2 bucket list
   ```

3. 查看详细错误日志
   编辑 `scripts/sync-gallery.sh`，临时移除上传函数中的 `&> /dev/null`

### 同步记录重置

如需重新上传所有图片：

```bash
rm .r2-sync-record
bash scripts/sync-gallery.sh
```

## 注意事项

- 图片文件名将直接用作 markdown 标题和 URL
- 支持中文文件名
- 已存在的 markdown 文件不会被覆盖
- R2 中已存在的文件会被记录为已同步（通过本地记录文件）
