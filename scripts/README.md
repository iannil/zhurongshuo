# CSS 优化脚本

自动提取项目中实际使用的 Remixicon 图标和 Animate.css 动画，生成精简版 CSS 文件。

## 功能

- 🔍 **自动扫描**: 扫描所有 HTML 模板，提取实际使用的图标和动画
- 📦 **大幅减小体积**:
  - Remixicon: 108KB → ~1KB (减少 99%)
  - Animate.css: 70KB → ~1KB (减少 98%)
- 🎨 **保持完整功能**: 包含所有必需的样式和字体引用
- 🔄 **可重复运行**: 随时重新生成以同步最新使用情况

## 使用方法

```bash
# 进入项目目录
cd /path/to/zhurongshuo

# 运行优化脚本
./scripts/optimize-css.sh

# 重新构建站点
hugo

# 本地预览
hugo server
```

## 工作原理

1. **扫描阶段**:
   - 遍历 `themes/zozo/layouts/` 和 `layouts/` 目录
   - 使用正则表达式提取 `ri-*` 和 `animate__*` 类名
   - 去重并统计数量

2. **生成阶段**:
   - **Remixicon**: 从原始 CSS 提取 @font-face、基础样式和使用到的图标定义
   - **Animate.css**: 手动生成对应动画的 keyframes 和类定义

3. **输出文件**:
   - `themes/zozo/static/css/remixicon-custom.css`
   - `themes/zozo/static/css/animate-custom.css`

## 当前使用情况

### Remixicon 图标 (5个)

- ri-arrow-up-s-line - 回到顶部按钮
- ri-map-pin-time-line - 日期图标
- ri-menu-line - 菜单图标
- ri-rss-fill - RSS订阅图标
- ri-stack-line - 标签图标

### Animate.css 动画 (1个)

- fadeInDown - 淡入向下动画

## 添加新动画支持

如果项目中添加了新的动画，需要在脚本中添加对应的 case 分支：

```bash
# 编辑 scripts/optimize-css.sh 的第 163-231 行
case $anim in
    fadeInDown)
        # ... 现有代码 ...
        ;;
    newAnimation)
        cat >> "$ANIMATE_OUTPUT" << 'EOF'
@keyframes newAnimation {
  /* 动画定义 */
}
.animate__newAnimation {
  animation-name: newAnimation;
}
EOF
        ;;
esac
```

## 脚本输出示例

```text
========================================
  CSS 优化脚本
========================================

[1/4] 扫描项目中使用的 Remixicon 图标...
  ✓ 找到 5 个图标:
    - ri-arrow-up-s-line
    - ri-map-pin-time-line
    - ri-menu-line
    - ri-rss-fill
    - ri-stack-line

[2/4] 生成 remixicon-custom.css...
  ✓ 生成成功: themes/zozo/static/css/remixicon-custom.css
    原始大小: 108K → 优化后: 1.0K

[3/4] 扫描项目中使用的 Animate.css 动画...
  ✓ 找到 1 个动画:
    - animate__fadeInDown

[4/4] 生成 animate-custom.css...
  ✓ 生成成功: themes/zozo/static/css/animate-custom.css
    原始大小: 72K → 优化后: 1.0K

========================================
✓ CSS 优化完成!
========================================

统计信息:
  • Remixicon 图标: 5 个
  • Animate.css 动画: 1 个

文件大小:
  • remixicon.css:        108K → 1.0K
  • animate.min.css:      72K → 1.0K
```

## 注意事项

1. **首次运行后**: 需要修改 `themes/zozo/layouts/partials/head.html`，将引用从完整 CSS 改为精简版
2. **定期运行**: 添加新图标或动画后应重新运行脚本
3. **版本控制**: 生成的 `-custom.css` 文件应该提交到 git
4. **Hugo 配置**: 确保 Hugo 能够复制这些新文件到 docs 目录

## 相关文件

- `scripts/optimize-css.sh` - 优化脚本
- `themes/zozo/static/css/remixicon.css` - 原始图标库
- `themes/zozo/static/css/animate.min.css` - 原始动画库
- `themes/zozo/static/css/remixicon-custom.css` - 生成的精简图标CSS
- `themes/zozo/static/css/animate-custom.css` - 生成的精简动画CSS
- `themes/zozo/layouts/partials/head.html` - 引用CSS的模板文件
