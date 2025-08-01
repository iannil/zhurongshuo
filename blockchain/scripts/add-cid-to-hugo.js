const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

class CidInjector {
  constructor() {
    this.uploadsPath = path.join(__dirname, '../uploads/upload-records.json');
    this.docsPath = path.join(__dirname, '../../docs');
    this.contentPath = path.join(__dirname, '../../content');
  }

  loadUploadRecords() {
    if (!fs.existsSync(this.uploadsPath)) {
      console.log('未找到上传记录文件');
      return [];
    }
    
    try {
      const records = JSON.parse(fs.readFileSync(this.uploadsPath, 'utf8'));
      return records.filter(record => !record.error);
    } catch (error) {
      console.error('读取上传记录失败:', error.message);
      return [];
    }
  }

  findMatchingHtmlFile(markdownPath) {
    // 将content路径转换为docs路径
    const relativePath = path.relative(this.contentPath, markdownPath);
    const htmlPath = path.join(this.docsPath, relativePath.replace('.md', '/index.html'));
    
    if (fs.existsSync(htmlPath)) {
      return htmlPath;
    }
    
    // 尝试其他可能的路径
    const possiblePaths = [
      path.join(this.docsPath, relativePath.replace('.md', '.html')),
      path.join(this.docsPath, path.dirname(relativePath), 'index.html')
    ];
    
    for (const possiblePath of possiblePaths) {
      if (fs.existsSync(possiblePath)) {
        return possiblePath;
      }
    }
    
    return null;
  }

  injectCidToHtml(htmlPath, cid, txHash, blockNumber) {
    try {
      let htmlContent = fs.readFileSync(htmlPath, 'utf8');
      
      // 创建区块链信息HTML
      const blockchainInfo = `
<!-- 区块链信息 -->
<div class="blockchain-info" style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; border-radius: 4px;">
  <h4 style="margin: 0 0 10px 0; color: #007bff;">🔗 区块链存储信息</h4>
  <div style="font-family: 'Courier New', monospace; font-size: 14px;">
    <p style="margin: 5px 0;"><strong>IPFS CID:</strong> <a href="https://ipfs.io/ipfs/${cid}" target="_blank" style="color: #007bff;">${cid}</a></p>
    <p style="margin: 5px 0;"><strong>交易哈希:</strong> <a href="https://polygonscan.com/tx/${txHash}" target="_blank" style="color: #007bff;">${txHash}</a></p>
    <p style="margin: 5px 0;"><strong>区块号:</strong> ${blockNumber}</p>
    <p style="margin: 5px 0;"><strong>存储时间:</strong> ${new Date().toLocaleString('zh-CN')}</p>
  </div>
  <div style="margin-top: 10px; font-size: 12px; color: #666;">
    <p style="margin: 0;">✅ 此内容已永久存储在去中心化网络上</p>
    <p style="margin: 0;">🔒 内容不可篡改，可验证真实性</p>
  </div>
</div>
`;

      // 查找合适的位置插入区块链信息
      // 尝试在文章内容后插入
      const contentEndPattern = /<\/article>/;
      if (contentEndPattern.test(htmlContent)) {
        htmlContent = htmlContent.replace(contentEndPattern, `${blockchainInfo}\n</article>`);
      } else {
        // 如果没有找到</article>，尝试在body结束前插入
        const bodyEndPattern = /<\/body>/;
        if (bodyEndPattern.test(htmlContent)) {
          htmlContent = htmlContent.replace(bodyEndPattern, `${blockchainInfo}\n</body>`);
        } else {
          // 最后尝试在</html>前插入
          htmlContent = htmlContent.replace(/<\/html>/, `${blockchainInfo}\n</html>`);
        }
      }
      
      fs.writeFileSync(htmlPath, htmlContent);
      console.log(`✅ 已注入CID到: ${path.relative(this.docsPath, htmlPath)}`);
      
      return true;
    } catch (error) {
      console.error(`❌ 注入CID失败: ${htmlPath}`, error.message);
      return false;
    }
  }

  updateMarkdownWithCid(markdownPath, cid, txHash, blockNumber) {
    try {
      const content = fs.readFileSync(markdownPath, 'utf8');
      const { data, content: markdownContent } = matter(content);
      
      // 添加区块链信息到front matter
      data.cid = cid;
      data.txHash = txHash;
      data.blockNumber = blockNumber;
      data.blockchainTimestamp = new Date().toISOString();
      
      // 重新生成markdown内容
      const updatedContent = matter.stringify(markdownContent, data);
      fs.writeFileSync(markdownPath, updatedContent);
      
      console.log(`✅ 已更新Markdown文件: ${path.relative(this.contentPath, markdownPath)}`);
      return true;
    } catch (error) {
      console.error(`❌ 更新Markdown失败: ${markdownPath}`, error.message);
      return false;
    }
  }

  processAllRecords() {
    console.log('=== 开始处理区块链信息注入 ===');
    
    const records = this.loadUploadRecords();
    if (records.length === 0) {
      console.log('没有找到有效的上传记录');
      return;
    }
    
    console.log(`找到 ${records.length} 条上传记录`);
    
    let successCount = 0;
    let failCount = 0;
    
    for (const record of records) {
      try {
        const markdownPath = record.filePath;
        
        if (!fs.existsSync(markdownPath)) {
          console.log(`⚠️  Markdown文件不存在: ${markdownPath}`);
          continue;
        }
        
        // 更新Markdown文件
        const markdownUpdated = this.updateMarkdownWithCid(
          markdownPath, 
          record.cid, 
          record.txHash, 
          record.blockNumber
        );
        
        // 查找并更新HTML文件
        const htmlPath = this.findMatchingHtmlFile(markdownPath);
        if (htmlPath) {
          const htmlUpdated = this.injectCidToHtml(
            htmlPath, 
            record.cid, 
            record.txHash, 
            record.blockNumber
          );
          
          if (markdownUpdated && htmlUpdated) {
            successCount++;
          } else {
            failCount++;
          }
        } else {
          console.log(`⚠️  未找到对应的HTML文件: ${markdownPath}`);
          if (markdownUpdated) {
            successCount++;
          } else {
            failCount++;
          }
        }
        
      } catch (error) {
        console.error(`❌ 处理记录失败:`, error.message);
        failCount++;
      }
    }
    
    console.log(`\n=== 处理完成 ===`);
    console.log(`✅ 成功: ${successCount} 个文件`);
    console.log(`❌ 失败: ${failCount} 个文件`);
  }

  createCidIndex() {
    console.log('=== 创建CID索引页面 ===');
    
    const records = this.loadUploadRecords();
    if (records.length === 0) {
      console.log('没有找到上传记录');
      return;
    }
    
    let indexHtml = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>区块链存储索引 - 祝融说</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .stats { display: flex; justify-content: space-around; margin-bottom: 30px; }
        .stat { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }
        .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
        .article { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 8px; }
        .article-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
        .blockchain-info { font-family: 'Courier New', monospace; font-size: 0.9em; color: #666; }
        .blockchain-info a { color: #007bff; text-decoration: none; }
        .blockchain-info a:hover { text-decoration: underline; }
        .timestamp { color: #999; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔗 区块链存储索引</h1>
        <p>所有内容已永久存储在去中心化网络上</p>
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-number">${records.length}</div>
            <div>总文章数</div>
        </div>
        <div class="stat">
            <div class="stat-number">${new Set(records.map(r => r.txHash)).size}</div>
            <div>区块链交易</div>
        </div>
        <div class="stat">
            <div class="stat-number">${new Set(records.map(r => r.cid)).size}</div>
            <div>IPFS文件</div>
        </div>
    </div>
    
    <div class="articles">
`;

    // 按时间倒序排列
    records.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    for (const record of records) {
      const date = new Date(record.timestamp).toLocaleDateString('zh-CN');
      const time = new Date(record.timestamp).toLocaleTimeString('zh-CN');
      
      indexHtml += `
        <div class="article">
            <div class="article-title">${record.title}</div>
            <div class="blockchain-info">
                <div><strong>IPFS CID:</strong> <a href="https://ipfs.io/ipfs/${record.cid}" target="_blank">${record.cid}</a></div>
                <div><strong>交易哈希:</strong> <a href="https://polygonscan.com/tx/${record.txHash}" target="_blank">${record.txHash}</a></div>
                <div><strong>区块号:</strong> ${record.blockNumber}</div>
                <div><strong>文件:</strong> ${record.fileName}</div>
            </div>
            <div class="timestamp">存储时间: ${date} ${time}</div>
        </div>
      `;
    }
    
    indexHtml += `
    </div>
    
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>🔒 所有内容已通过智能合约永久存储在区块链上</p>
        <p>🌐 内容同时存储在IPFS去中心化文件系统中</p>
        <p>✅ 数据不可篡改，可验证真实性</p>
    </div>
</body>
</html>
`;

    const indexPath = path.join(this.docsPath, 'blockchain-index.html');
    fs.writeFileSync(indexPath, indexHtml);
    console.log(`✅ 已创建区块链索引页面: ${indexPath}`);
  }
}

// 命令行接口
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'process';
  
  const injector = new CidInjector();
  
  switch (command) {
    case 'process':
      injector.processAllRecords();
      break;
      
    case 'index':
      injector.createCidIndex();
      break;
      
    case 'all':
      injector.processAllRecords();
      injector.createCidIndex();
      break;
      
    default:
      console.log(`
使用方法:
  node add-cid-to-hugo.js process    # 处理所有记录，注入CID到HTML和Markdown
  node add-cid-to-hugo.js index      # 创建区块链索引页面
  node add-cid-to-hugo.js all        # 执行所有操作
      `);
      break;
  }
}

if (require.main === module) {
  main();
}

module.exports = CidInjector; 