# Cloudflare 免费套餐优化指南

本指南针对祝融说网站（zhurongshuo.com）使用 Cloudflare 作为 CDN 代理 GitHub Pages 的优化配置。

## 📊 免费套餐限制

- **Page Rules**: 3 条规则
- **Cache Rules**: 替代 Page Rules（推荐）
- **Transform Rules**: 10 条规则
- **Rate Limiting**: 1 条规则
- **带宽**: 无限制
- **请求数**: 100,000/天

## 🎯 优化策略

### 1. 缓存规则配置（Cache Rules）

#### 规则 1: 静态资源长期缓存

- **匹配条件**: 文件扩展名为 `.css`, `.js`, `.woff`, `.woff2`, `.ttf`, `.ico`
- **缓存设置**:
  - Edge Cache TTL: 1 年 (31536000 秒)
  - Browser Cache TTL: 1 个月 (2592000 秒)
  - Cache Level: Cache Everything

```text
URI Path 匹配:
/css/*
/js/*
/fonts/*
*.css
*.js
*.woff*
*.ttf
*.ico
```

#### 规则 2: HTML 内容短期缓存

- **匹配条件**: 文件扩展名为 `.html` 或根路径
- **缓存设置**:
  - Edge Cache TTL: 2 小时 (7200 秒)
  - Browser Cache TTL: 30 分钟 (1800 秒)
  - Cache Level: Standard

```text
URI Path 匹配:
/*.html
/
/posts/*
/about/*
/start/*
/advanced/*
```

#### 规则 3: XML 文件和媒体缓存

- **匹配条件**: 文件扩展名为 `.xml`, `.jpg`, `.png`, `.gif`, `.webp`
- **缓存设置**:
  - Edge Cache TTL: 1 天 (86400 秒)
  - Browser Cache TTL: 4 小时 (14400 秒)

```text
URI Path 匹配:
/sitemap.xml
/index.xml
*.jpg
*.jpeg
*.png
*.gif
*.webp
*.svg
```

### 2. 性能优化设置

在 Cloudflare Dashboard > Speed > Optimization 中启用:

- ✅ **Auto Minify**: HTML, CSS, JavaScript
- ✅ **Brotli 压缩**: 更好的压缩比
- ✅ **Early Hints**: 预加载关键资源
- ✅ **HTTP/3 (QUIC)**: 更快的连接
- ✅ **0-RTT Connection Resumption**: 减少握手时间

### 3. 安全头配置（Transform Rules）

#### 规则 1: 添加安全头

```text
Expression: (http.request.uri.path matches ".*")
Headers:
- Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' www.googletagmanager.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'
```

#### 规则 2: 缓存控制头

```text
Expression: (http.request.uri.path contains "/css/" or http.request.uri.path contains "/js/" or http.request.uri.path contains "/fonts/")
Headers:
- Cache-Control: public, max-age=31536000, immutable
```

### 4. Bot Protection 和安全

在 Security > Bots 中配置:

- ✅ **Bot Fight Mode**: 自动阻止恶意机器人
- ✅ **Super Bot Fight Mode**: 如果需要更强保护（付费功能）

在 Security > Scrape Shield 中配置:

- ✅ **Hotlink Protection**: 防止图片盗链
- ✅ **Email Address Obfuscation**: 混淆邮箱地址

### 5. Rate Limiting 配置（1 条免费规则）

保护最重要的页面或接口:

```text
Rule Name: 主页保护
Expression: (http.request.uri.path eq "/")
Characteristics: IP
Period: 1 minute
Requests: 60
Action: Challenge
```

### 6. Page Rules 配置（如果不使用 Cache Rules）

如果继续使用传统 Page Rules：

#### 规则 1: 静态资源

- **URL**: `zhurongshuo.com/css/*`, `zhurongshuo.com/js/*`, `zhurongshuo.com/fonts/*`
- **设置**: Cache Level = Cache Everything, Edge Cache TTL = 1 year

#### 规则 2: HTML 内容

- **URL**: `zhurongshuo.com/*`
- **设置**: Cache Level = Standard, Browser Cache TTL = 2 hours

#### 规则 3: XML 文件

- **URL**: `zhurongshuo.com/*.xml`
- **设置**: Cache Level = Cache Everything, Edge Cache TTL = 1 day

## 🔧 DNS 配置优化

在 DNS 设置中：

- 主域名记录设为 "橙色云朵"（Proxied）
- TTL 设置为 Auto（Cloudflare 自动管理）

## 📈 监控和维护

### Analytics 监控

- 定期检查 Analytics > Traffic 了解流量模式
- 监控 Analytics > Performance 查看缓存命中率
- 关注 Analytics > Security 检查威胁阻止情况

### 缓存清理

需要更新内容时，使用 Caching > Configuration > Purge Cache:

- **清理单个文件**: 输入具体文件 URL
- **清理所有内容**: 谨慎使用，会影响性能
- **清理标签**: 如果使用了 Cache-Tag（需要代码配置）

### 性能测试

- 使用 [GTmetrix](https://gtmetrix.com) 测试页面性能
- 使用 [WebPageTest](https://www.webpagetest.org) 测试全球加载速度
- 监控 Core Web Vitals 指标

## 🚨 注意事项

1. **免费套餐限制**: 小心不要超过规则数量限制
2. **缓存策略**: HTML 缓存时间不要太长，避免内容更新不及时
3. **安全头**: CSP 规则要根据实际需求调整，避免功能异常
4. **监控流量**: 虽然带宽无限，但要注意请求数量限制

## 🔄 部署后验证

部署这些配置后，验证以下项目：

1. **缓存命中率**: 在 Analytics 中查看缓存性能
2. **页面加载速度**: 使用在线工具测试
3. **安全头**: 使用 [Security Headers](https://securityheaders.com) 检查
4. **CDN 分发**: 确认全球访问速度

## 🛠 故障排除

如果遇到问题：

1. **缓存问题**: 临时暂停 Cloudflare（设为 DNS Only）测试
2. **样式丢失**: 检查 CSS/JS 文件的缓存规则
3. **功能异常**: 逐个禁用 Transform Rules 排查
4. **速度变慢**: 检查 Origin Server（GitHub Pages）状态

---

*更新于: 2025-11-19*
*适用于: Cloudflare 免费套餐*
