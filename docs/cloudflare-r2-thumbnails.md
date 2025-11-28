# Cloudflare R2 图片缩略图配置说明

## 概述

本项目使用 Cloudflare R2 作为图片存储，并通过 Cloudflare Worker 实现图片缩略图功能，以优化 Gallery 页面的加载速度。

## 工作原理

1. **Hugo 模板** (`layouts/gallery/list.html`) 自动为图片添加缩略图参数
   - 缩略图: `?w=600&q=75` (宽度600px，质量75%)
   - 原图: 无参数

2. **Cloudflare Worker** (`workers/image-resizer.js`) 拦截请求并处理图片
   - 检测到缩略图参数时尝试缩放图片
   - 支持 WebP 自动转换（根据浏览器 Accept header）
   - 未启用 Image Resizing 时优雅降级返回原图

3. **缓存策略**
   - CDN 缓存: `max-age=31536000` (1年)
   - 浏览器缓存: `immutable`

## 部署步骤

### 1. 配置环境变量

在 `.env` 文件中配置（已配置好）:

```bash
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_WORKER_API_TOKEN=your_worker_api_token
CLOUDFLARE_R2_API_TOKEN=your_r2_api_token
```

### 2. 部署 Worker

```bash
# 方法 1: 使用部署脚本（推荐）
./scripts/deploy-worker.sh

# 方法 2: 使用 wrangler 直接部署
wrangler deploy --env production
```

### 3. 验证部署

部署脚本会自动测试 Worker 端点，你也可以手动验证：

```bash
# 测试原图
curl -I https://r2.zhurongshuo.com/images/gallery/photo.jpg

# 测试缩略图
curl -I https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=600&q=75
```

查看响应头中的 `X-Image-Processing` 字段：
- `resized`: 图片已成功缩放 ✅
- `original-fallback`: 返回原图（未启用 Image Resizing）
- `original`: 原图请求（未带参数）

## Cloudflare Image Resizing

### 当前状态

Worker 已配置支持 Cloudflare Image Resizing，但需要：

1. **订阅服务** ($5/月) 或使用付费计划
   - Pro 计划及以上自动包含
   - Free 计划可单独购买 Image Resizing 附加服务

2. **启用方式**
   - 访问: https://dash.cloudflare.com/?to=/:account/images/image-resizing
   - 或在 Cloudflare Dashboard → Images → Image Resizing

### 功能对比

| 功能 | 未启用 Image Resizing | 已启用 Image Resizing |
|------|---------------------|---------------------|
| 图片访问 | ✅ 正常 | ✅ 正常 |
| 缩略图参数 | ✅ 识别但返回原图 | ✅ 自动缩放 |
| WebP 转换 | ❌ 不支持 | ✅ 自动转换 |
| 带宽节省 | ❌ 无节省 | ✅ 30-70% 节省 |
| 加载速度 | 🐌 较慢 | ⚡ 更快 |

### 替代方案

如果不想订阅 Image Resizing，可以：

1. **预生成缩略图**（推荐免费方案）
   - 在本地或 CI/CD 中使用 ImageMagick/Sharp 生成缩略图
   - 上传时保存多个尺寸（如 `photo.jpg`, `photo_thumb.jpg`）
   - 修改 Hugo 模板使用预生成的缩略图

2. **使用其他 CDN**
   - imgix, Cloudinary 等提供免费额度
   - 但需要迁移存储

## Gallery 模板配置

当前 `layouts/gallery/list.html` 的缩略图配置（第 33 行）:

```go
{{- $thumbnailImage = printf "%s%s?w=600&q=75" $cdnURL .Params.featured_image -}}
```

参数说明:
- `w=600`: 宽度 600px（适合 Gallery grid 显示）
- `q=75`: JPEG 质量 75%（平衡质量与大小）

可以根据需要调整这些参数：
- 更小缩略图: `w=400&q=70`
- 更高质量: `w=800&q=85`
- 固定尺寸: `w=600&h=400&fit=cover`

## 监控与调试

查看 Worker 日志:

```bash
wrangler tail --env production
```

检查响应头:

```bash
curl -I "https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=600&q=75"
```

关键响应头：
- `X-Image-Processing`: 处理状态
- `X-Image-Width`: 请求的宽度
- `X-Image-Quality`: 图片质量
- `Cache-Control`: 缓存策略
- `Content-Type`: 图片格式

## 性能优化建议

1. ✅ **已实现**: 缩略图参数配置
2. ✅ **已实现**: 延迟加载 (`loading="lazy"`)
3. ✅ **已实现**: 长期缓存策略
4. ✅ **已实现**: WebP 支持（需启用 Image Resizing）
5. 🔄 **可选**: 启用 Cloudflare Image Resizing 获得最佳性能

## 成本分析

| 方案 | 存储成本 | 流量成本 | 处理成本 | 总成本估算 |
|------|---------|---------|---------|-----------|
| R2 + 原图 | $0.015/GB | $0 | $0 | 最低 |
| R2 + Image Resizing | $0.015/GB | $0 | $5/月 | 中等 |
| R2 + 预生成缩略图 | $0.03/GB | $0 | $0 | 较低 |

对于小型站点（<10GB 图片），预生成缩略图最经济。
对于大型站点或频繁更新图片，Image Resizing 更方便。

## 故障排查

### Worker 未生效

1. 检查 Worker 路由配置: `wrangler.toml` 中的 `route`
2. 验证 DNS 记录: `r2.zhurongshuo.com` 应为 CNAME 或 A 记录
3. 查看 Worker 日志: `wrangler tail`

### 图片 404

1. 确认图片已上传到 R2: `wrangler r2 object list zhurongshuo`
2. 检查路径大小写是否匹配
3. 验证 R2 bucket 绑定配置

### 缩略图不工作

1. 检查响应头 `X-Image-Processing` 值
2. 如果是 `original-fallback`: 需启用 Image Resizing
3. 如果是 `original`: 检查 URL 参数是否正确

## 相关文件

- Worker 代码: `workers/image-resizer.js`
- Worker 配置: `wrangler.toml`
- Gallery 模板: `layouts/gallery/list.html`
- 部署脚本: `scripts/deploy-worker.sh`
- 主部署脚本: `deploy.sh`
- 环境配置: `.env`
