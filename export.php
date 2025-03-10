<?php

$docs = './content/posts';
$csv = './祝融说_副本'.date('Ymd').'.csv';
$files = getFilesByExtension($docs, 'md');
$data = [];

foreach ($files as $file) {
    $content = file_get_contents($file);

    $result = parseMarkdown($content);

    $data[] = [$result['metadata']['date'], $result['last_paragraph']];
}

if (writeCsvWithoutFunction($data, $csv, false)) {
    echo "文件写入成功...".PHP_EOL;
} else {
    echo "文件写入失败...".PHP_EOL;
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
    $result['last_paragraph'] = trim($content[count($content) - 1]);

    return $result;
}
