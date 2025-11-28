# Cloudflare Image Resizing Worker

这个 Worker 为 R2 存储的图片提供动态调整大小和优化功能。

## 功能特性

- **动态调整大小**: 通过 URL 参数指定宽度和高度
- **质量控制**: 自定义图片压缩质量
- **自动格式转换**: 根据浏览器支持自动转换为 WebP 或 AVIF
- **智能缓存**: CDN 和浏览器双重缓存
- **优雅降级**: 如果调整大小失败，自动回退到原始图片

## 使用方法

### 基本 URL 格式

```
https://r2.zhurongshuo.com/<path-to-image>?<parameters>
```

### 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `w` | 宽度（像素） | - | `w=800` |
| `h` | 高度（像素） | - | `h=600` |
| `q` | 质量（1-100） | 85 | `q=75` |
| `fit` | 缩放模式 | scale-down | `fit=cover` |

### 缩放模式 (fit)

- `scale-down`: 缩小到指定尺寸，不放大
- `contain`: 等比缩放，完整显示
- `cover`: 等比缩放，填充整个区域
- `crop`: 裁剪到指定尺寸
- `pad`: 添加边距以适应尺寸

### 使用示例

#### 1. 原始图片
```
https://r2.zhurongshuo.com/images/gallery/photo.jpg
```

#### 2. 调整宽度为 800px
```
https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=800
```

#### 3. 调整宽度和高度
```
https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=800&h=600
```

#### 4. 调整质量
```
https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=800&q=75
```

#### 5. 组合参数
```
https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=600&h=400&q=80&fit=cover
```

#### 6. 缩略图
```
https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=200&h=200&fit=cover&q=75
```

## 浏览器支持

Worker 会自动检测浏览器的 `Accept` 请求头，并返回最优格式：

- **AVIF**: 支持 AVIF 的现代浏览器（Chrome 85+, Firefox 93+）
- **WebP**: 支持 WebP 的浏览器（大多数现代浏览器）
- **原格式**: 其他浏览器保持原始格式（JPEG, PNG 等）

## 缓存策略

- **CDN 缓存**: Cloudflare CDN 自动缓存处理后的图片
- **浏览器缓存**: `Cache-Control: public, max-age=31536000, immutable`（1年）
- **ETag**: 支持 ETag 进行条件请求

## 响应头

Worker 会添加以下自定义响应头用于调试：

```
X-Image-Processing: resized | original | fallback-to-original
X-Image-Width-Requested: 800
X-Image-Height-Requested: 600
X-Image-Quality-Requested: 85
X-Image-Fit: scale-down
X-Image-Format: webp | avif | auto
CF-Cache-Status: HIT | MISS | EXPIRED
CF-Image-Format: webp | avif | jpeg
```

## 在网站中使用

### HTML

```html
<!-- 响应式图片 -->
<img
  src="https://r2.zhurongshuo.com/images/photo.jpg?w=800"
  srcset="
    https://r2.zhurongshuo.com/images/photo.jpg?w=400 400w,
    https://r2.zhurongshuo.com/images/photo.jpg?w=800 800w,
    https://r2.zhurongshuo.com/images/photo.jpg?w=1200 1200w
  "
  sizes="(max-width: 600px) 100vw, 800px"
  alt="Photo"
>

<!-- 缩略图 -->
<img
  src="https://r2.zhurongshuo.com/images/photo.jpg?w=200&h=200&fit=cover"
  alt="Thumbnail"
>
```

### Markdown

```markdown
![Photo](https://r2.zhurongshuo.com/images/photo.jpg?w=800)
```

### CSS

```css
.hero {
  background-image: url('https://r2.zhurongshuo.com/images/hero.jpg?w=1920&q=85');
}

@media (max-width: 768px) {
  .hero {
    background-image: url('https://r2.zhurongshuo.com/images/hero.jpg?w=768&q=80');
  }
}
```

## 部署

```bash
# 部署到生产环境
export CLOUDFLARE_API_TOKEN=your-api-token
wrangler deploy --env production
```

## 配置要求

1. **Cloudflare Image Resizing 订阅**
   - 访问 [Cloudflare Dashboard](https://dash.cloudflare.com/?to=/:account/images/image-resizing)
   - 启用 Image Resizing（需要付费计划或 $5/月 附加服务）

2. **R2 Bucket 绑定**
   - 在 `wrangler.toml` 中配置 R2 bucket 绑定
   - 确保 bucket 名称正确

3. **Route 配置**
   - 配置自定义域名指向 Worker
   - 设置 DNS CNAME 记录

## 性能优化建议

1. **使用合适的质量参数**
   - 照片: `q=85` (默认)
   - 缩略图: `q=75-80`
   - 背景图: `q=70-75`

2. **合理设置尺寸**
   - 移动设备: 400-800px
   - 平板: 800-1200px
   - 桌面: 1200-1920px

3. **利用浏览器缓存**
   - 图片 URL 不变时，浏览器会自动使用缓存
   - 修改图片后更改文件名或添加版本参数

4. **使用响应式图片**
   - 使用 `srcset` 和 `sizes` 属性
   - 让浏览器根据设备选择最合适的尺寸

## 故障排查

### 图片显示为原始尺寸

可能原因：
1. Image Resizing 服务未启用
2. Worker 配置错误
3. 查看响应头 `X-Image-Processing: fallback-to-original` 和 `X-Image-Error`

### 图片加载缓慢

建议：
1. 检查 `CF-Cache-Status` 响应头
2. 首次访问时会较慢（需要处理并缓存）
3. 后续访问应该很快（从 CDN 缓存）

### 图片质量不理想

调整：
1. 提高质量参数 `q=90`
2. 增加图片尺寸
3. 尝试不同的 `fit` 模式

## 成本估算

Cloudflare Image Resizing 定价：
- **免费计划**: 不包含
- **Pro 计划**: $20/月，包含基础 Image Resizing
- **Business/Enterprise**: 包含高级 Image Resizing
- **附加服务**: $5/月（添加到任何计划）

计费方式：
- 按处理的唯一图片数量计费
- CDN 缓存的图片不重复计费
- 建议设置合理的缓存策略

## 更多资源

- [Cloudflare Image Resizing 文档](https://developers.cloudflare.com/images/image-resizing/)
- [Workers 文档](https://developers.cloudflare.com/workers/)
- [R2 文档](https://developers.cloudflare.com/r2/)
