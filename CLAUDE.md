# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hugo-based static site for a Chinese philosophy blog called "祝融说" (zhurongshuo.com). The site explores philosophical concepts, particularly around the Dao De Jing and original philosophical frameworks.

## Build & Development Commands

### Local Development

```bash
# Build the site (output to docs/ directory)
hugo

# Build with minification and garbage collection (production)
hugo --minify --gc

# Run local development server with drafts and live reload
hugo server -D -E -F --watch
# Access at http://localhost:1313
```

### Docker Development

```bash
# Start development environment with hot reload
docker-compose up dev

# Start production environment
docker-compose up -d prod

# View logs
docker-compose logs -f prod
```

### Deployment

```bash
# Automated deploy script (pull, build, commit, push)
./deploy.sh

# Linux deploy variant
./deploy-linux.sh
```

### Content Export

```bash
# Export posts to CSV (extracts dates and last paragraphs)
./export.sh
```

## Architecture

### Hugo Configuration

- **Config**: `config.toml` - Hugo site configuration
- **Base URL**: <https://zhurongshuo.com/>
- **Output Directory**: `docs/` (GitHub Pages compatible)
- **Theme**: "zozo" theme located in `themes/zozo/`
- **Language**: Chinese (zh-cn)

### Content Structure

The site uses Hugo's content organization with several distinct sections:

#### Main Content Sections

- **`content/posts/`** - Primary blog posts organized by year/month (e.g., `2025/10/0201.md`)
  - Posts include front matter with: title, date, tags, keywords, description, slug, hidden, draft
  - URL structure uses slug field for permalinks

- **`content/core/`** - Philosophical core framework document
  - Contains axioms, theorems, and corollaries about consciousness and reality
  - Central to the site's philosophical framework

- **`content/kachuai/dao/`** - Dao De Jing interpretations
  - Contains 49 numbered files (1.md through 49.md)
  - Each interprets a chapter of the Dao De Jing
  - Tagged with "卡揣", "诸子百家", "老子"

- **`content/advanced/`** - Advanced philosophical content
- **`content/teach/`** - Teaching materials
- **`content/yiguan/`** - Additional philosophical essays
- **`content/about/`** - About page

#### Archetype Template

- **`archetypes/default.md`** - Template for new content with YAML front matter

### Deployment Pipeline

1. **GitHub Actions** (`.github/workflows/hugo.yml`)
   - Triggers on push to master branch
   - Uses Hugo 0.138.0 extended version
   - Builds site and deploys to GitHub Pages
   - Output artifact from `docs/` directory

2. **Docker Production**
   - Multi-stage Dockerfile using `hugomods/hugo:nightly` and `caddy:2.9.1-alpine`
   - Caddy serves the static site with security headers
   - Configured via `Caddyfile`

### Front Matter Format

Posts use YAML front matter with these fields:

```yaml
---
title: "Post Title"
date: 2025-01-01T10:00:00+08:00
hidden: false  # Whether to hide from listings
draft: false   # Draft status
tags: ["tag1", "tag2"]
keywords: ["keyword1", "keyword2"]
description: "Optional description"
slug: "url-slug"
---
```

## Key Technical Details

- **Hugo Version**: Site uses extended Hugo (locally v0.151.0, CI uses v0.138.0)
- **Port**: Development server runs on port 1313
- **Web Server**: Caddy 2.9.1 in production with HSTS headers and security rules
- **Timezone**: All dates use +08:00 (China Standard Time)
- **Output Format**: Static HTML to `docs/` directory
- **MathJax**: Enabled for mathematical notation support

## Content Export Utility

The `export.sh` script:

- Recursively processes all `.md` files in `content/posts/`
- Extracts YAML `date` field and last paragraph from each post
- Outputs to CSV with UTF-8 BOM encoding
- Sorts by date in descending order
- Filename format: `祝融说_副本YYYYMMDD.csv`

## Navigation Structure

Configured in `config.toml` with menu items:

- 首页 (Home) - `/`
- 核心 (Core) - `/core/`
- 进阶 (Advanced) - `/advanced/`
- 归档 (Archive) - `/posts/`
- 标签 (Tags) - `/tags/`
- 关于 (About) - `/about/`
