const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const matter = require('gray-matter');
const { ethers } = require('hardhat');
require('dotenv').config();

class ContentUploader {
  constructor() {
    this.pinataApiKey = process.env.PINATA_API_KEY;
    this.pinataSecretApiKey = process.env.PINATA_SECRET_API_KEY;
    this.contractAddress = process.env.CONTRACT_ADDRESS;
    this.network = process.env.NETWORK || 'polygon';
    
    if (!this.pinataApiKey || !this.pinataSecretApiKey) {
      throw new Error('请在.env文件中设置PINATA_API_KEY和PINATA_SECRET_API_KEY');
    }
    
    if (!this.contractAddress) {
      throw new Error('请在.env文件中设置CONTRACT_ADDRESS');
    }
  }

  async uploadToIPFS(filePath, fileName) {
    try {
      console.log(`正在上传文件到IPFS: ${fileName}`);
      
      const fileContent = fs.readFileSync(filePath);
      const formData = new FormData();
      
      formData.append('file', fileContent, {
        filename: fileName,
        contentType: 'text/markdown'
      });

      const response = await axios.post('https://api.pinata.cloud/pinning/pinFileToIPFS', formData, {
        headers: {
          'Authorization': `Bearer ${this.pinataApiKey}`,
          ...formData.getHeaders()
        }
      });

      const cid = response.data.IpfsHash;
      console.log(`文件上传成功! CID: ${cid}`);
      return cid;
    } catch (error) {
      console.error('IPFS上传失败:', error.response?.data || error.message);
      throw error;
    }
  }

  async postToBlockchain(cid, title, description) {
    try {
      console.log('正在调用智能合约...');
      
      const [signer] = await ethers.getSigners();
      const BlogContract = await ethers.getContractFactory("BlogContract");
      const contract = BlogContract.attach(this.contractAddress).connect(signer);
      
      const tx = await contract.postArticle(cid, title, description);
      console.log(`交易已发送: ${tx.hash}`);
      
      const receipt = await tx.wait();
      console.log(`交易已确认! 区块号: ${receipt.blockNumber}`);
      
      return receipt;
    } catch (error) {
      console.error('区块链交易失败:', error.message);
      throw error;
    }
  }

  extractMetadata(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const { data, content: markdownContent } = matter(content);
      
      return {
        title: data.title || path.basename(filePath, '.md'),
        description: data.description || data.excerpt || '',
        tags: data.tags || [],
        date: data.date || new Date().toISOString(),
        content: markdownContent
      };
    } catch (error) {
      console.error('解析文件元数据失败:', error.message);
      return {
        title: path.basename(filePath, '.md'),
        description: '',
        tags: [],
        date: new Date().toISOString(),
        content: fs.readFileSync(filePath, 'utf8')
      };
    }
  }

  async uploadSingleFile(filePath) {
    try {
      const fileName = path.basename(filePath);
      const metadata = this.extractMetadata(filePath);
      
      console.log(`\n=== 处理文件: ${fileName} ===`);
      console.log(`标题: ${metadata.title}`);
      console.log(`描述: ${metadata.description}`);
      
      // 上传到IPFS
      const cid = await this.uploadToIPFS(filePath, fileName);
      
      // 调用智能合约
      const receipt = await this.postToBlockchain(cid, metadata.title, metadata.description);
      
      // 保存上传记录
      const uploadRecord = {
        fileName,
        filePath,
        cid,
        title: metadata.title,
        description: metadata.description,
        tags: metadata.tags,
        date: metadata.date,
        txHash: receipt.transactionHash,
        blockNumber: receipt.blockNumber,
        timestamp: new Date().toISOString()
      };
      
      this.saveUploadRecord(uploadRecord);
      
      console.log(`✅ 文件处理完成!`);
      console.log(`📄 文件: ${fileName}`);
      console.log(`🔗 IPFS CID: ${cid}`);
      console.log(`⛓️  交易哈希: ${receipt.transactionHash}`);
      console.log(`📊 区块号: ${receipt.blockNumber}`);
      
      return uploadRecord;
    } catch (error) {
      console.error(`❌ 文件处理失败: ${filePath}`, error.message);
      throw error;
    }
  }

  async uploadDirectory(directoryPath) {
    try {
      console.log(`\n=== 开始批量上传目录: ${directoryPath} ===`);
      
      const files = this.getMarkdownFiles(directoryPath);
      console.log(`找到 ${files.length} 个Markdown文件`);
      
      const results = [];
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        console.log(`\n[${i + 1}/${files.length}] 处理文件: ${path.basename(file)}`);
        
        try {
          const result = await this.uploadSingleFile(file);
          results.push(result);
          
          // 添加延迟避免API限制
          if (i < files.length - 1) {
            console.log('等待3秒...');
            await new Promise(resolve => setTimeout(resolve, 3000));
          }
        } catch (error) {
          console.error(`文件 ${file} 上传失败:`, error.message);
          results.push({
            fileName: path.basename(file),
            filePath: file,
            error: error.message,
            timestamp: new Date().toISOString()
          });
        }
      }
      
      console.log(`\n=== 批量上传完成 ===`);
      console.log(`成功: ${results.filter(r => !r.error).length} 个文件`);
      console.log(`失败: ${results.filter(r => r.error).length} 个文件`);
      
      return results;
    } catch (error) {
      console.error('批量上传失败:', error.message);
      throw error;
    }
  }

  getMarkdownFiles(directoryPath) {
    const files = [];
    
    function scanDirectory(dir) {
      const items = fs.readdirSync(dir);
      
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
          scanDirectory(fullPath);
        } else if (path.extname(item).toLowerCase() === '.md') {
          files.push(fullPath);
        }
      }
    }
    
    scanDirectory(directoryPath);
    return files;
  }

  saveUploadRecord(record) {
    const recordsPath = path.join(__dirname, '../uploads');
    if (!fs.existsSync(recordsPath)) {
      fs.mkdirSync(recordsPath, { recursive: true });
    }
    
    const recordFile = path.join(recordsPath, 'upload-records.json');
    let records = [];
    
    if (fs.existsSync(recordFile)) {
      records = JSON.parse(fs.readFileSync(recordFile, 'utf8'));
    }
    
    records.push(record);
    fs.writeFileSync(recordFile, JSON.stringify(records, null, 2));
  }

  async getContractInfo() {
    try {
      const [signer] = await ethers.getSigners();
      const BlogContract = await ethers.getContractFactory("BlogContract");
      const contract = BlogContract.attach(this.contractAddress).connect(signer);
      
      const articleCount = await contract.getArticleCount();
      console.log(`\n=== 合约信息 ===`);
      console.log(`合约地址: ${this.contractAddress}`);
      console.log(`网络: ${this.network}`);
      console.log(`文章总数: ${articleCount}`);
      
      if (articleCount > 0) {
        console.log(`\n=== 最新文章 ===`);
        const latestArticle = await contract.getArticle(articleCount);
        console.log(`文章ID: ${articleCount}`);
        console.log(`标题: ${latestArticle.title}`);
        console.log(`CID: ${latestArticle.contentCid}`);
        console.log(`作者: ${latestArticle.author}`);
        console.log(`时间: ${new Date(latestArticle.timestamp * 1000).toLocaleString()}`);
      }
    } catch (error) {
      console.error('获取合约信息失败:', error.message);
    }
  }
}

// 命令行接口
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  
  if (!command) {
    console.log(`
使用方法:
  node upload-content.js single <文件路径>     # 上传单个文件
  node upload-content.js batch <目录路径>     # 批量上传目录
  node upload-content.js info                # 查看合约信息
    `);
    return;
  }
  
  try {
    const uploader = new ContentUploader();
    
    switch (command) {
      case 'single':
        if (!args[1]) {
          console.error('请指定文件路径');
          return;
        }
        await uploader.uploadSingleFile(args[1]);
        break;
        
      case 'batch':
        if (!args[1]) {
          console.error('请指定目录路径');
          return;
        }
        await uploader.uploadDirectory(args[1]);
        break;
        
      case 'info':
        await uploader.getContractInfo();
        break;
        
      default:
        console.error('未知命令:', command);
        break;
    }
  } catch (error) {
    console.error('执行失败:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = ContentUploader; 