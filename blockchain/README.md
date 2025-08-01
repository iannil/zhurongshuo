# 祝融说 - 去中心化博客系统

这是一个完全去中心化的博客系统，使用 IPFS 存储内容，Polygon 区块链记录元数据。

## 🏗️ 系统架构

- **内容存储**: IPFS (通过 Pinata API)
- **元数据存储**: Polygon 智能合约
- **前端**: Hugo 静态网站生成器
- **区块链网络**: Polygon (主网) / Mumbai (测试网)

## 📁 项目结构

```
blockchain/
├── contracts/
│   └── BlogContract.sol          # 智能合约
├── scripts/
│   ├── deploy.js                 # 合约部署脚本
│   ├── upload-content.js         # 内容上传脚本
│   └── add-cid-to-hugo.js        # CID注入脚本
├── uploads/                      # 上传记录
├── deployments/                  # 部署信息
├── package.json
├── hardhat.config.js
└── env.example
```

## 🚀 快速开始

### 1. 环境准备

```bash
cd blockchain
npm install
```

### 2. 配置环境变量

复制 `env.example` 为 `.env` 并填写以下信息：

```bash
cp env.example .env
```

**必需配置:**

- `PRIVATE_KEY`: 你的钱包私钥
- `PINATA_API_KEY`: Pinata API Key
- `PINATA_SECRET_API_KEY`: Pinata Secret API Key

**可选配置:**

- `NETWORK`: 区块链网络 (polygon/mumbai/arbitrum/optimism)
- `POLYGONSCAN_API_KEY`: PolygonScan API Key (用于合约验证)

### 3. 部署智能合约

```bash
# 部署到测试网
npm run deploy:testnet

# 部署到主网
npm run deploy
```

部署完成后，将合约地址添加到 `.env` 文件中的 `CONTRACT_ADDRESS`。

### 4. 上传内容

```bash
# 上传单个文件
node scripts/upload-content.js single ../content/posts/2024/example.md

# 批量上传目录
node scripts/upload-content.js batch ../content/posts/2024

# 查看合约信息
node scripts/upload-content.js info
```

### 5. 注入CID到Hugo页面

```bash
# 处理所有记录，注入CID到HTML和Markdown
node scripts/add-cid-to-hugo.js process

# 创建区块链索引页面
node scripts/add-cid-to-hugo.js index

# 执行所有操作
node scripts/add-cid-to-hugo.js all
```

## 📋 智能合约功能

### BlogContract.sol

**主要功能:**

- `postArticle()`: 发布新文章，记录CID和元数据
- `getArticle()`: 根据ID查询文章信息
- `getAllArticles()`: 获取所有文章ID
- `getArticleCount()`: 获取文章总数

**事件:**

- `ArticlePosted`: 文章发布时触发
- `OwnershipTransferred`: 所有权转移时触发

## 🔧 脚本说明

### upload-content.js

自动完成以下流程：

1. 解析 Markdown 文件的 front matter
2. 上传文件到 IPFS (Pinata)
3. 调用智能合约记录 CID
4. 保存上传记录

**使用方法:**

```bash
# 上传单个文件
node scripts/upload-content.js single <文件路径>

# 批量上传目录
node scripts/upload-content.js batch <目录路径>

# 查看合约信息
node scripts/upload-content.js info
```

### add-cid-to-hugo.js

将区块链信息注入到 Hugo 生成的页面中：

1. 更新 Markdown 文件的 front matter
2. 在 HTML 页面中插入区块链信息区块
3. 创建区块链存储索引页面

**使用方法:**

```bash
# 处理所有记录
node scripts/add-cid-to-hugo.js process

# 创建索引页面
node scripts/add-cid-to-hugo.js index

# 执行所有操作
node scripts/add-cid-to-hugo.js all
```

## 🌐 支持的区块链网络

### Polygon (推荐)

- **主网**: 低成本，快速确认
- **测试网**: Mumbai，用于测试

### Arbitrum

- **主网**: 低Gas费，高吞吐量
- **测试网**: Goerli

### Optimism

- **主网**: 低Gas费，Layer2解决方案
- **测试网**: Goerli

## 📊 费用估算

### Polygon 主网

- 合约部署: ~0.01-0.05 MATIC
- 每篇文章发布: ~0.001-0.005 MATIC

### IPFS 存储 (Pinata)

- 免费计划: 1GB 存储，每月 100 个 pin
- 付费计划: 按使用量计费

## 🔒 安全注意事项

1. **私钥安全**: 永远不要将私钥提交到代码仓库
2. **环境变量**: 使用 `.env` 文件存储敏感信息
3. **测试网**: 先在测试网测试所有功能
4. **备份**: 定期备份上传记录和部署信息

## 🛠️ 故障排除

### 常见问题

**1. 合约部署失败**

- 检查私钥和网络配置
- 确保账户有足够的代币支付Gas费

**2. IPFS上传失败**

- 检查 Pinata API 密钥
- 确认网络连接正常

**3. 交易失败**

- 检查Gas费设置
- 确认合约地址正确

**4. CID注入失败**

- 检查文件路径是否正确
- 确认 Hugo 已生成静态文件

### 日志文件

- 上传记录: `uploads/upload-records.json`
- 部署信息: `deployments/<network>.json`

## 📈 扩展功能

### 可能的改进

1. 支持 Arweave 存储
2. 添加内容哈希验证
3. 实现去中心化域名解析
4. 添加内容加密功能
5. 支持批量操作优化

### 监控和统计

1. 区块链事件监听
2. IPFS 存储状态监控
3. 访问统计和分析

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## �� 许可证

MIT License
