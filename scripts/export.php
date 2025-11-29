<?php

$docs = dirname(__DIR__) . '/content/posts';
$archiveDir = dirname(__DIR__) . '/archive';
$csv = $archiveDir . '/祝融说_副本' . date('Ymd') . '.csv';
$files = getFilesByExtension($docs, 'md');
$data = [];

foreach ($files as $file) {
    $content = file_get_contents($file);

    $result = parseMarkdown($content);

    $data[] = [$result['metadata']['date'], $result['last_paragraph']];
}

$data = sortByDate($data, false);
if (empty($data)) {
    echo "没有找到符合条件的文件或数据为空...".PHP_EOL;
    exit(0);
}

if (writeCsvWithoutFunction($data, $csv, false)) {
    echo "文件写入成功: $csv".PHP_EOL;

    // 删除过期的导出文件（保留最近7天的文件）
    cleanupOldExports($archiveDir, 7);
} else {
    echo "文件写入失败...".PHP_EOL;
}

/**
 * 清理过期的导出文件
 *
 * @param string $dir 归档目录
 * @param int $keepDays 保留天数
 */
function cleanupOldExports(string $dir, int $keepDays = 7): void
{
    if (!is_dir($dir)) {
        return;
    }

    $cutoffTime = time() - ($keepDays * 86400);
    $pattern = '/^祝融说_副本\d{8}\.csv$/';

    $files = scandir($dir);
    $deletedCount = 0;

    foreach ($files as $file) {
        if ($file === '.' || $file === '..') {
            continue;
        }

        $filePath = $dir . '/' . $file;

        // 只删除匹配导出文件命名模式的CSV文件
        if (preg_match($pattern, $file) && is_file($filePath)) {
            $fileTime = filemtime($filePath);

            if ($fileTime < $cutoffTime) {
                if (unlink($filePath)) {
                    echo "已删除过期文件: $file" . PHP_EOL;
                    $deletedCount++;
                }
            }
        }
    }

    if ($deletedCount > 0) {
        echo "清理完成，共删除 $deletedCount 个过期文件" . PHP_EOL;
    }
}

/**
 * 按照日期对数组进行排序
 * 
 * @param array $items 待排序的数组，每个元素应包含 'date' 和 'content' 键
 * @param bool $ascending 是否升序排列，默认为true（升序）
 * @return array 排序后的数组
 * @throws InvalidArgumentException 如果数组元素缺少必要的键
 */
function sortByDate(array $items, bool $ascending = true): array
{
    // 验证数组结构
    foreach ($items as $item) {
        if (!isset($item[0]) || !isset($item[1])) {
            throw new InvalidArgumentException('每个数组元素长度必须为2');
        }
    }
    // 排序操作
    usort($items, function ($a, $b) use ($ascending) {
        // 将日期转换为时间戳进行比较
        $timestampA = strtotime($a[0]);
        $timestampB = strtotime($b[0]);
        
        // 根据排序方向返回比较结果
        return $ascending 
            ? $timestampA <=> $timestampB 
            : $timestampB <=> $timestampA;
    });
    return $items;
}

/**
 * 手动实现CSV文件写入（兼容RFC 4180标准）
 * @param array $data 二维数组数据
 * @param string $filePath 输出文件路径
 * @param bool $hasHeader 是否包含标题行
 * @param string $delimiter 分隔符（默认逗号）
 * @param string $enclosure 包裹符（默认双引号）
 * @return bool
 */
function writeCsvWithoutFunction(array $data, string $filePath, bool $hasHeader = false, string $delimiter = ',', string $enclosure = '"'): bool {
    // 参数有效性验证
    if (empty($data) || !is_array(current($data))) {
        trigger_error("输入数据必须是非空二维数组", E_USER_WARNING);
        return false;
    }

    // 自动创建目录
    $dir = dirname($filePath);
    if (!is_dir($dir) && !mkdir($dir, 0755, true)) {
        trigger_error("无法创建目录：{$dir}", E_USER_WARNING);
        return false;
    }

    // 打开文件句柄
    if (($handle = fopen($filePath, 'w')) === false) {
        trigger_error("无法打开文件：{$filePath}", E_USER_WARNING);
        return false;
    }

    try {
        // CSV BOM头（解决中文乱码）
        fwrite($handle, "\xEF\xBB\xBF");

        // 处理标题行
        if ($hasHeader && !empty($data)) {
            $header = array_shift($data);
            writeCsvLine($handle, $header, $delimiter, $enclosure);
        }

        // 写入数据行
        foreach ($data as $row) {
            writeCsvLine($handle, $row, $delimiter, $enclosure);
        }

        return true;
    } catch (Exception $e) {
        trigger_error("CSV写入失败：" . $e->getMessage(), E_USER_WARNING);
        return false;
    } finally {
        fclose($handle);
    }
}

/**
 * 处理单个CSV行
 */
function writeCsvLine($handle, array $row, string $delimiter, string $enclosure): void {
    $processed = [];
    
    foreach ($row as $field) {
        $field = (string)$field;
        
        // 处理需要包裹的字段
        $needsEnclosure = strpbrk($field, "{$delimiter}{$enclosure}\n\r") !== false;
        
        // 转义包裹符
        $escaped = str_replace($enclosure, $enclosure . $enclosure, $field);
        
        $processed[] = $needsEnclosure ? ($enclosure . $escaped . $enclosure) : $escaped;
    }

    $line = implode($delimiter, $processed) . "\r\n";
    
    if (fwrite($handle, $line) === false) {
        throw new Exception("写入文件失败");
    }
}

/**
 * 获取指定目录下特定后缀的文件列表
 * 
 * @param string $directory 要搜索的目录路径
 * @param string|array $extensions 要查找的文件扩展名（支持字符串或数组）
 * @return array 匹配的文件路径数组
 */
function getFilesByExtension($directory, $extensions)
{
    // 参数有效性验证
    if (!is_dir($directory) || !is_readable($directory)) {
        return [];
    }

    // 统一扩展名格式为小写数组
    $extensions = array_map(
        'strtolower',
        is_array($extensions)
            ? $extensions
            : explode(',', str_replace(' ', '', $extensions))
    );

    $foundFiles = [];

    // 创建递归目录迭代器
    $iterator = new RecursiveIteratorIterator(
        new RecursiveDirectoryIterator(
            $directory,
            FilesystemIterator::SKIP_DOTS | FilesystemIterator::UNIX_PATHS
        ),
        RecursiveIteratorIterator::CHILD_FIRST
    );

    // 遍历文件系统
    foreach ($iterator as $fileInfo) {
        if ($fileInfo->isFile()) {
            $fileExt = strtolower($fileInfo->getExtension());
            if (in_array($fileExt, $extensions)) {
                $foundFiles[] = $fileInfo->getRealPath();
            }
        }
    }

    return $foundFiles;
}

function processMatches($matches)
{
    $result = [];

    // 基础字段处理
    foreach (['title', 'date', 'draft', 'description', 'slug'] as $field) {
        $result[$field] = $matches[$field] ?? null;
    }

    // 数组字段处理
    foreach (['tags', 'keywords'] as $field) {
        try {
            $result[$field] = json_decode(str_replace('"', '\"', $matches[$field]));
        } catch (Exception $e) {
            $result[$field] = [];
        }
    }

    // 内容处理
    $result['content'] = htmlspecialchars_decode(
        trim(strip_tags($matches['content'] ?? ''))
    );

    // 布尔类型转换
    $result['draft'] = ($result['draft'] === 'true');

    return $result;
}

function parseMarkdown($content)
{
    $frontMatterPattern = '/^---\R(?<frontmatter>.*?)\R---/sm';

    $patterns = [
        'title'       => '/^title:\s*(["\'])(?<title>.*?)\1/m',
        'date'        => '/^date:\s*(?<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2})/m',
        'draft'       => '/^draft:\s*(?<draft>true|false)/mi',
        'tags'        => '/^tags:\s*\[(?<tags>[^\]]+)\]/m',
        'keywords'    => '/^keywords:\s*\[(?<keywords>[^\]]+)\]/m',
        'description' => '/^description:\s*(["\']?)(?<description>.*?)\1$/m',
        'slug'        => '/^slug:\s*(["\']?)(?<slug>[^"\'\s]+)\1$/m'
    ];

    $result = [
        'metadata' => [],
        'last_paragraph' => ''
    ];

    // 第一部分：提取front-matter
    if (preg_match($frontMatterPattern, $content, $frontMatch)) {
        foreach ($patterns as $key => $pattern) {
            if (preg_match($pattern, $frontMatch['frontmatter'], $matches)) {
                // 特殊处理数组类型字段
                switch ($key) {
                    case 'tags':
                    case 'keywords':
                        $result['metadata'][$key] = array_map(
                            'trim',
                            explode(',', $matches[$key])
                        );
                        break;
                    case 'draft':
                        $result['metadata'][$key] = strtolower($matches[$key]) === 'true';
                        break;
                    default:
                        $result['metadata'][$key] = $matches[$key];
                }
            }
        }
    }

    $content = explode('---', $content);
    $result['last_paragraph'] = str_replace(["\n", "\r", "> ", "*", "<!--more-->"], ["", "", "", "", ""], trim($content[count($content) - 1]));

    return $result;
}
