# 仓库指南

## 项目结构与模块组织

- `content/` 存放所有 Markdown 源文件；按既有栏目结构（`posts/`、`core/`、`advanced/`、`teach/` 等）归类新内容，并通过 `archetypes/default.md` 快速生成稿件。
- 展示层位于 `layouts/` 与 `themes/zozo/`；优先在 `layouts/` 中覆写模板，避免直接改动主题目录。
- `static/` 保存原始资源，Hugo 会将构建产物发布到用于 GitHub Pages 的 `docs/`；请勿手动编辑 `docs/`。

## 构建、测试与开发命令

- `hugo` — 将站点构建到 `docs/`，提交前运行以捕捉链接或 Front Matter 异常。
- `hugo --minify --gc` — 与 CI 一致的生产构建，发版前执行。
- `hugo server -D -E -F --watch` — 在 `http://localhost:1313` 预览草稿与未来文章，含实时刷新。
- `docker-compose up dev` — 启动容器化开发环境，对齐 Caddy 头信息与缓存策略。
- `./deploy.sh` — 审核通过后执行全量拉取、重建、提交与推送，避免并行手动 Git 操作。

## 代码风格与命名约定

- 内容文件使用 UTF-8 Markdown，YAML Front Matter 字段顺序为 `title`、`date`、`draft`、`hidden`、`tags`、`keywords`、`description`、`slug`，保持 +08:00 时间戳。
- 短代码与 Go 模板采用两个空格缩进，文件名使用小写连字符（如 `2025/10/0201.md`），发布后避免改动 slug。
- 修改模板或局部时保持逻辑精简，仅对不易理解的辅助函数添加简短注释。

## 测试指南

- 提交 PR 前需确保 `hugo` 无警告输出，及时修复缺失资源或翻译提示。
- 调整布局或 CSS 时，使用 `hugo server` 在多个栏目预览，并附上修改前后截图说明影响。
- 改动 Docker 或部署脚本时，运行 `docker-compose up dev` 验证启动流程与 Caddy 日志是否正常。

## 提交与合并请求规范

- 遵循仓库现有的时间戳式提交风格（`YYYYMMDDHHMMSS on macos`），必要时追加平台或范围后缀。
- 保持单次提交聚焦单一主题（内容 / 布局 / 基础设施），便于回滚。
- PR 描述需概述改动范围、列出涉及目录、关联问题并提供界面截图（如有 UI 变化）。
- 明确 `docs/` 是否已重新生成；若依赖 CI 构建，请在 PR 中说明。

## 内容工作流提示

- 使用 `hugo new posts/2025/10/example.md` 创建草稿，确保元数据完整。
- 共享图片或 PDF 存放于 `static/assets/`，并通过绝对路径引用（`/assets/...`）以匹配线上环境。
