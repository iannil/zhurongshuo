# Gallery 使用说明

这个目录用于存放摄影作品页面的媒体文件。

## 目录结构

```
static/
├── images/gallery/  # 存放照片
├── videos/gallery/  # 存放视频
└── audio/gallery/   # 存放音频
```

## 如何添加新的作品

### 1. 添加图片作品

在 `content/gallery/` 目录下创建新的 `.md` 文件，例如 `my-photo.md`：

**使用本地文件：**

```markdown
---
title: "作品标题"
date: 2025-11-27T00:00:00+08:00
draft: false
type: "gallery"
featured_image: "/images/gallery/your-image.jpg"
description: "作品简短描述"
tags: ["标签1", "标签2"]
---

这里是作品的详细描述内容。

可以包含拍摄参数、创作故事等信息。
```

然后将图片文件放到 `static/images/gallery/` 目录下。

**使用外部URL：**

```markdown
---
title: "作品标题"
date: 2025-11-27T00:00:00+08:00
draft: false
type: "gallery"
featured_image: "https://example.com/path/to/image.jpg"
description: "作品简短描述"
tags: ["标签1", "标签2"]
---

图片托管在CDN或其他服务上，直接填写完整URL即可。
```

### 2. 添加视频作品

**使用本地文件：**

```markdown
---
title: "视频标题"
date: 2025-11-27T00:00:00+08:00
draft: false
type: "gallery"
video: "/videos/gallery/your-video.mp4"
description: "视频简短描述"
tags: ["标签1", "标签2"]
---

视频的详细描述。
```

将视频文件放到 `static/videos/gallery/` 目录下。

**使用外部URL：**

```markdown
---
title: "视频标题"
date: 2025-11-27T00:00:00+08:00
draft: false
type: "gallery"
video: "https://cdn.example.com/videos/your-video.mp4"
description: "视频简短描述"
tags: ["标签1", "标签2"]
---

视频托管在视频服务或CDN上。
```

### 3. 添加音频作品

**使用本地文件：**

```markdown
---
title: "音频标题"
date: 2025-11-27T00:00:00+08:00
draft: false
type: "gallery"
audio: "/audio/gallery/your-audio.mp3"
description: "音频简短描述"
tags: ["标签1", "标签2"]
---

音频的详细描述。
```

将音频文件放到 `static/audio/gallery/` 目录下。

**使用外部URL：**

```markdown
---
title: "音频标题"
date: 2025-11-27T00:00:00+08:00
draft: false
type: "gallery"
audio: "https://cdn.example.com/audio/your-audio.mp3"
description: "音频简短描述"
tags: ["标签1", "标签2"]
---

音频托管在云存储或CDN上。
```

## 注意事项

1. 所有作品文件必须设置 `type: "gallery"` 才能正确显示
2. **媒体文件支持两种方式**：
   - **本地文件**：以 `/` 开头的相对路径（如 `/images/gallery/photo.jpg`）
   - **外部URL**：完整的HTTP/HTTPS地址（如 `https://cdn.example.com/image.jpg`）
3. 本地图片建议压缩优化后再上传，建议宽度不超过 2000px
4. 视频建议使用 MP4 格式，音频建议使用 MP3 格式
5. 文件名建议使用英文和数字，避免使用中文和特殊字符
6. 使用外部URL时，请确保URL可访问且支持跨域（CORS）

## 示例文件

目录中已经包含了18个示例文件，展示了流式布局效果：

**图片示例（15个）：**
- `sample-photo-1.md` - 城市夜景（竖图 800×1200）
- `sample-photo-2.md` - 自然风光（竖图 900×1400）
- `sample-photo-url.md` - 外部URL示例（横图）
- `sample-photo-3.md` - 夕阳余晖（竖图 800×1200）
- `sample-photo-4.md` - 海边的宁静（竖图 800×1300）
- `sample-photo-5.md` - 森林深处（竖图 900×1350）
- `sample-photo-6.md` - 雪山之巅（竖图 900×1350）
- `sample-photo-7.md` - 城市街道（竖图 800×1000）
- `sample-photo-8.md` - 星空璀璨（竖图 1200×1600）
- `sample-photo-9.md` - 田园风光（竖图 800×1100）
- `sample-photo-10.md` - 现代建筑（竖图 900×1200）
- `sample-photo-11.md` - 秋日色彩（方图 1200×900）
- `sample-photo-12.md` - 沙漠之美（竖图 800×1100）
- `sample-photo-13.md` - 瀑布奔流（竖图 1000×1500）
- `sample-photo-14.md` - 湖光山色（竖图 850×1300）
- `sample-photo-15.md` - 花的世界（竖图 850×1200）

**视频示例（2个）：**
- `sample-video-1.md` - 延时摄影
- `sample-video-2.md` - 自然风光

**音频示例（1个）：**
- `sample-audio-1.md` - 自然之声

这些示例大部分使用竖图（高度大于宽度），能够很好地展示瀑布流布局的错落有致效果。

实际使用时，请将示例文件替换为你自己的作品。
