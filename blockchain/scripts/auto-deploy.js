const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

class AutoDeployer {
  constructor() {
    this.projectRoot = path.join(__dirname, '../..');
    this.blockchainDir = path.join(__dirname, '..');
  }

  log(message, type = 'info') {
    const timestamp = new Date().toLocaleString('zh-CN');
    const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️';
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  async checkEnvironment() {
    this.log('检查环境配置...');
    
    const requiredEnvVars = [
      'PRIVATE_KEY',
      'PINATA_API_KEY', 
      'PINATA_SECRET_API_KEY'
    ];

    const missing = [];
    for (const envVar of requiredEnvVars) {
      if (!process.env[envVar]) {
        missing.push(envVar);
      }
    }

    if (missing.length > 0) {
      throw new Error(`缺少必需的环境变量: ${missing.join(', ')}`);
    }

    this.log('环境配置检查通过', 'success');
  }

  async installDependencies() {
    this.log('安装依赖包...');
    
    try {
      execSync('npm install', { 
        cwd: this.blockchainDir, 
        stdio: 'inherit' 
      });
      this.log('依赖安装完成', 'success');
    } catch (error) {
      throw new Error(`依赖安装失败: ${error.message}`);
    }
  }

  async compileContracts() {
    this.log('编译智能合约...');
    
    try {
      execSync('npm run compile', { 
        cwd: this.blockchainDir, 
        stdio: 'inherit' 
      });
      this.log('合约编译完成', 'success');
    } catch (error) {
      throw new Error(`合约编译失败: ${error.message}`);
    }
  }

  async testContracts() {
    this.log('测试智能合约...');
    
    try {
      execSync('npm run test:contract', { 
        cwd: this.blockchainDir, 
        stdio: 'inherit' 
      });
      this.log('合约测试完成', 'success');
    } catch (error) {
      this.log('合约测试失败，但继续部署', 'warning');
    }
  }

  async deployContract(network = 'mumbai') {
    this.log(`部署智能合约到 ${network} 网络...`);
    
    try {
      const deployCommand = network === 'polygon' ? 'npm run deploy' : 'npm run deploy:testnet';
      execSync(deployCommand, { 
        cwd: this.blockchainDir, 
        stdio: 'inherit' 
      });
      
      // 读取部署信息
      const deploymentFile = path.join(this.blockchainDir, 'deployments', `${network}.json`);
      if (fs.existsSync(deploymentFile)) {
        const deployment = JSON.parse(fs.readFileSync(deploymentFile, 'utf8'));
        this.log(`合约部署成功: ${deployment.contractAddress}`, 'success');
        return deployment.contractAddress;
      } else {
        throw new Error('未找到部署信息文件');
      }
    } catch (error) {
      throw new Error(`合约部署失败: ${error.message}`);
    }
  }

  async updateEnvFile(contractAddress) {
    this.log('更新环境变量文件...');
    
    const envPath = path.join(this.blockchainDir, '.env');
    const envExamplePath = path.join(this.blockchainDir, 'env.example');
    
    if (!fs.existsSync(envPath) && fs.existsSync(envExamplePath)) {
      // 复制示例文件
      fs.copyFileSync(envExamplePath, envPath);
    }
    
    if (fs.existsSync(envPath)) {
      let envContent = fs.readFileSync(envPath, 'utf8');
      
      // 更新合约地址
      if (envContent.includes('CONTRACT_ADDRESS=')) {
        envContent = envContent.replace(
          /CONTRACT_ADDRESS=.*/,
          `CONTRACT_ADDRESS=${contractAddress}`
        );
      } else {
        envContent += `\nCONTRACT_ADDRESS=${contractAddress}`;
      }
      
      fs.writeFileSync(envPath, envContent);
      this.log('环境变量文件已更新', 'success');
    }
  }

  async uploadContent(contentPath) {
    this.log('上传内容到IPFS和区块链...');
    
    try {
      const uploadCommand = `node scripts/upload-content.js batch "${contentPath}"`;
      execSync(uploadCommand, { 
        cwd: this.blockchainDir, 
        stdio: 'inherit' 
      });
      this.log('内容上传完成', 'success');
    } catch (error) {
      throw new Error(`内容上传失败: ${error.message}`);
    }
  }

  async buildHugo() {
    this.log('构建Hugo静态网站...');
    
    try {
      execSync('hugo', { 
        cwd: this.projectRoot, 
        stdio: 'inherit' 
      });
      this.log('Hugo构建完成', 'success');
    } catch (error) {
      throw new Error(`Hugo构建失败: ${error.message}`);
    }
  }

  async injectCidToHugo() {
    this.log('注入CID到Hugo页面...');
    
    try {
      execSync('npm run inject-cid all', { 
        cwd: this.blockchainDir, 
        stdio: 'inherit' 
      });
      this.log('CID注入完成', 'success');
    } catch (error) {
      throw new Error(`CID注入失败: ${error.message}`);
    }
  }

  async fullDeploy(options = {}) {
    const {
      network = 'mumbai',
      contentPath = '../content',
      skipTests = false,
      skipUpload = false
    } = options;

    try {
      this.log('=== 开始自动化部署流程 ===');
      
      // 1. 检查环境
      await this.checkEnvironment();
      
      // 2. 安装依赖
      await this.installDependencies();
      
      // 3. 编译合约
      await this.compileContracts();
      
      // 4. 测试合约 (可选)
      if (!skipTests) {
        await this.testContracts();
      }
      
      // 5. 部署合约
      const contractAddress = await this.deployContract(network);
      
      // 6. 更新环境变量
      await this.updateEnvFile(contractAddress);
      
      // 7. 上传内容 (可选)
      if (!skipUpload) {
        await this.uploadContent(contentPath);
      }
      
      // 8. 构建Hugo
      await this.buildHugo();
      
      // 9. 注入CID
      await this.injectCidToHugo();
      
      this.log('=== 自动化部署完成 ===', 'success');
      
      return {
        success: true,
        contractAddress,
        network,
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      this.log(`部署失败: ${error.message}`, 'error');
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  async quickDeploy() {
    this.log('=== 快速部署模式 ===');
    
    // 检查是否已有合约地址
    if (!process.env.CONTRACT_ADDRESS) {
      this.log('未找到合约地址，请先运行完整部署', 'error');
      return { success: false, error: '未找到合约地址' };
    }
    
    try {
      // 1. 构建Hugo
      await this.buildHugo();
      
      // 2. 注入CID
      await this.injectCidToHugo();
      
      this.log('快速部署完成', 'success');
      return { success: true, timestamp: new Date().toISOString() };
      
    } catch (error) {
      this.log(`快速部署失败: ${error.message}`, 'error');
      return { success: false, error: error.message };
    }
  }
}

// 命令行接口
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';
  
  const deployer = new AutoDeployer();
  
  switch (command) {
    case 'full':
      const options = {
        network: args[1] || 'mumbai',
        contentPath: args[2] || '../content',
        skipTests: args.includes('--skip-tests'),
        skipUpload: args.includes('--skip-upload')
      };
      await deployer.fullDeploy(options);
      break;
      
    case 'quick':
      await deployer.quickDeploy();
      break;
      
    case 'deploy-only':
      const network = args[1] || 'mumbai';
      const contractAddress = await deployer.deployContract(network);
      await deployer.updateEnvFile(contractAddress);
      break;
      
    case 'upload-only':
      const contentPath = args[1] || '../content';
      await deployer.uploadContent(contentPath);
      break;
      
    case 'hugo-only':
      await deployer.buildHugo();
      await deployer.injectCidToHugo();
      break;
      
    case 'help':
    default:
      console.log(`
自动化部署脚本使用方法:

完整部署:
  node auto-deploy.js full [network] [content-path] [options]
  示例: node auto-deploy.js full mumbai ../content
  选项: --skip-tests, --skip-upload

快速部署 (仅Hugo构建和CID注入):
  node auto-deploy.js quick

仅部署合约:
  node auto-deploy.js deploy-only [network]

仅上传内容:
  node auto-deploy.js upload-only [content-path]

仅构建Hugo:
  node auto-deploy.js hugo-only

网络选项:
  mumbai    - Polygon测试网 (默认)
  polygon   - Polygon主网
  arbitrum  - Arbitrum主网
  optimism  - Optimism主网
      `);
      break;
  }
}

if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error('执行失败:', error.message);
      process.exit(1);
    });
}

module.exports = AutoDeployer; 