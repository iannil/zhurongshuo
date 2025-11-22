---
title: "Theme Preview"
date: 2019-03-31T17:49:40+08:00
hidden: false
draft: false
tags: ["theme"]
keywords: []
description: ""
slug: "Theme Preview"
---

# Headings

```markdown
# H1
## H2
### H3
#### H4
##### H5
###### H6
```

<!--more-->

# H1

## H2

### H3

#### H4

##### H5

###### H6

# Paragraphs

```markdown
This is a paragraph.
I am still part of the paragraph.

New paragraph.
```

This is a paragraph.
I am still part of the paragraph.

New paragraph.

# Image

```markdown
Web Image

![Web Image](https://i.loli.net/2019/04/13/5cb1d33cf0ee6.jpg)

Local Image

![Local Image](100.jpg)

```

Web Image

![Web Image](https://i.loli.net/2019/04/13/5cb1d33cf0ee6.jpg)

Local Image

![Local Image](/100.jpg)

# Block Quotes

```markdown
> This is a block quote
```

> This is a block quote

# Code Blocks

``````markdown
```javascript
// Fenced **with** highlighting
function doIt() {
    for (var i = 1; i <= slen ; i^^) {
        setTimeout("document.z.textdisplay.value = newMake()", i*300);
        setTimeout("window.status = newMake()", i*300);
    }
}
```
``````

```javascript
function doIt() {
    for (var i = 1; i <= slen ; i^^) {
        setTimeout("document.z.textdisplay.value = newMake()", i*300);
        setTimeout("window.status = newMake()", i*300);
    }
}
```

# Tables

```markdown
| Colors        | Fruits          | Vegetable         |
| ------------- |:---------------:| -----------------:|
| Red           | *Apple*         | [Pepper](#Tables) |
| ~~Orange~~    | Oranges         | **Carrot**        |
| Green         | ~~***Pears***~~ | Spinach           |
```

| Colors        | Fruits          | Vegetable         |
| ------------- |:---------------:| -----------------:|
| Red           | *Apple*         | [Pepper](#tables) |
| ~~Orange~~    | Oranges         | **Carrot**        |
| Green         | ~~***Pears***~~ | Spinach           |

# List Types

#### Ordered List

```markdown
1. First item
2. Second item
3. Third item
```

1. First item
2. Second item
3. Third item

#### Unordered List

```markdown
- First item
- Second item
- Third item
```

- First item
- Second item
- Third item

# Math

```
$$
evidence\_{i}=\sum\_{j}W\_{ij}x\_{j}+b\_{i}
$$

$$
AveP = \int_0^1 p(r) dr
$$

When $a \ne 0$, there are two solutions to \(ax^2 + bx + c = 0\) and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$
```

$$
evidence\_{i}=\sum\_{j}W\_{ij}x\_{j}+b\_{i}
$$

$$
AveP = \int_0^1 p(r) dr
$$

When $a \ne 0$, there are two solutions to \(ax^2 + bx + c = 0\) and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

#### Emoji

This is a test for emoji.
:smile:
:see_no_evil:
:smile_cat:
:watermelon:

#### Audio

在任何 Markdown 文章中，使用以下语法添加音频：

```
{{< audio src="/audio/your-file.mp3" title="音频标题" >}}
```

参数说明：

- src（必需）：音频文件路径，相对于 static/ 目录
- title（可选）：音频标题，显示在播放器上方

示例：

{{< audio src="/audio/podcast.mp3" title="第一期播客" >}}

1. 添加音频文件：在项目根目录创建 static/audio/ 文件夹，将音频文件放入其中
2. 运行 Hugo 预览：hugo server -D 查看效果
3. 在文章中使用：在任何文章的 Markdown 内容中插入音频 shortcode
