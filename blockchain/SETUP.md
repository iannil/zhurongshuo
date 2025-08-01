# 祝融说 - 去中心化博客系统设置指南

## 🎯 系统概述

这是一个完全去中心化的博客系统，具有以下特点：

- **内容存储**: IPFS (去中心化文件系统)
- **元数据存储**: Polygon 区块链 (智能合约)
- **前端**: Hugo 静态网站生成器
- **不可篡改**: 所有内容永久存储在区块链上
- **可验证**: 通过CID和交易哈希验证内容真实性

## 🚀 快速开始

### 1. 环境准备

确保你的系统已安装：

- Node.js (v16+)
- npm
- Hugo (可选，用于本地预览)

### 2. 获取必要的API密钥

#### Pinata API (IPFS存储)

1. 访问 [Pinata](https://app.pinata.cloud/)
2. 注册账户并登录
3. 在API Keys页面创建新的API Key
4. 记录下 `API Key` 和 `Secret API Key`

#### 区块链网络

- **测试网**: Mumbai (Polygon测试网) - 免费测试
- **主网**: Polygon - 需要MATIC代币

### 3. 钱包准备

1. 安装 MetaMask 或其他Web3钱包
2. 创建或导入钱包
3. 导出私钥 (⚠️ 安全风险，请妥善保管)
4. 如果是主网，确保钱包中有足够的MATIC代币

### 4. 一键启动

```bash
cd blockchain
./quick-start.sh
```

按照提示完成配置即可。

## 📋 详细设置步骤

### 步骤1: 安装依赖

```bash
cd blockchain
npm install
```

### 步骤2: 配置环境变量

复制示例文件并编辑：

```bash
cp env.example .env
```

编辑 `.env` 文件，填写以下信息：

```env
# 区块链网络配置
NETWORK=mumbai
PRIVATE_KEY=your_private_key_here

# IPFS配置 (Pinata)
PINATA_API_KEY=your_pinata_api_key
PINATA_SECRET_API_KEY=your_pinata_secret_api_key

# 合约地址 (部署后自动填写)
CONTRACT_ADDRESS=
```

### 步骤3: 部署智能合约

```bash
# 测试网部署
npm run deploy:testnet

# 主网部署 (需要MATIC代币)
npm run deploy
```

### 步骤4: 上传内容

```bash
# 上传单个文件
node scripts/upload-content.js single ../content/posts/2024/example.md

# 批量上传目录
node scripts/upload-content.js batch ../content/posts/2024
```

### 步骤5: 构建网站并注入CID

```bash
# 构建Hugo网站
hugo

# 注入CID到HTML页面
node scripts/add-cid-to-hugo.js all
```

## 🔧 自动化部署

使用自动化脚本可以一键完成所有操作：

```bash
# 完整部署 (推荐)
node scripts/auto-deploy.js full mumbai ../content

# 快速部署 (仅Hugo构建和CID注入)
node scripts/auto-deploy.js quick

# 仅部署合约
node scripts/auto-deploy.js deploy-only mumbai

# 仅上传内容
node scripts/auto-deploy.js upload-only ../content
```

## 📊 费用估算

### Polygon 网络费用

- **合约部署**: ~0.01-0.05 MATIC
- **每篇文章发布**: ~0.001-0.005 MATIC
- **测试网**: 完全免费

### IPFS 存储费用 (Pinata)

- **免费计划**: 1GB 存储，每月 100 个 pin
- **付费计划**: 按使用量计费

## 🔍 验证和监控

### 查看合约信息

```bash
node scripts/upload-content.js info
```

### 区块链浏览器查看

- **Mumbai测试网**: [PolygonScan Mumbai](https://mumbai.polygonscan.com/)
- **Polygon主网**: [PolygonScan](https://polygonscan.com/)

### IPFS网关查看

- 通过CID在 [IPFS.io](https://ipfs.io/) 查看内容
- 使用 [Pinata](https://app.pinata.cloud/) 管理文件

## 🛠️ 故障排除

### 常见问题

**1. 合约部署失败**

```
错误: 合约部署失败
解决: 检查私钥和网络配置，确保账户有足够代币
```

**2. IPFS上传失败**

```
错误: IPFS上传失败
解决: 检查Pinata API密钥，确认网络连接
```

**3. 交易失败**

```
错误: 交易失败
解决: 检查Gas费设置，确认合约地址正确
```

**4. CID注入失败**

```
错误: CID注入失败
解决: 检查文件路径，确认Hugo已生成静态文件
```

### 日志文件位置

- 上传记录: `uploads/upload-records.json`
- 部署信息: `deployments/<network>.json`

## 🔒 安全注意事项

1. **私钥安全**
   - 永远不要将私钥提交到代码仓库
   - 使用 `.env` 文件存储敏感信息
   - 定期更换私钥

2. **测试网优先**
   - 先在测试网测试所有功能
   - 确认无误后再部署到主网

3. **备份重要数据**
   - 定期备份上传记录
   - 保存部署信息
   - 备份环境配置文件

## 📈 扩展功能

### 可能的改进

1. **多链支持**
   - 支持 Arbitrum、Optimism 等其他L2网络
   - 跨链内容同步

2. **存储优化**
   - 支持 Arweave 永久存储
   - 内容压缩和去重

3. **功能增强**
   - 内容加密
   - 去中心化域名解析
   - 内容版本控制

4. **监控和分析**
   - 区块链事件监听
   - 访问统计
   - 存储状态监控

## 🤝 获取帮助

如果遇到问题，可以：

1. 查看详细文档: `README.md`
2. 检查日志文件
3. 在GitHub上提交Issue
4. 查看区块链浏览器确认交易状态

## 📄 许可证

MIT License - 可自由使用和修改
