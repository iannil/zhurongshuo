#!/bin/bash

# 祝融说 - 去中心化博客系统快速启动脚本

set -e

echo "🚀 祝融说 - 去中心化博客系统快速启动"
echo "=================================="

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到Node.js，请先安装Node.js"
    exit 1
fi

# 检查npm
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: 未找到npm，请先安装npm"
    exit 1
fi

echo "✅ Node.js版本: $(node --version)"
echo "✅ npm版本: $(npm --version)"

# 检查Hugo
if ! command -v hugo &> /dev/null; then
    echo "⚠️  警告: 未找到Hugo，请先安装Hugo"
    echo "   安装方法: https://gohugo.io/installation/"
    echo "   继续执行其他步骤..."
else
    echo "✅ Hugo版本: $(hugo version)"
fi

# 进入blockchain目录
cd "$(dirname "$0")"

echo ""
echo "📦 安装依赖包..."
npm install

echo ""
echo "🔧 检查环境配置..."

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  未找到.env文件，正在创建..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ 已创建.env文件，请编辑并填写以下信息："
        echo "   - PRIVATE_KEY: 你的钱包私钥"
        echo "   - PINATA_API_KEY: Pinata API Key"
        echo "   - PINATA_SECRET_API_KEY: Pinata Secret API Key"
        echo ""
        echo "📝 编辑完成后，运行以下命令继续："
        echo "   ./quick-start.sh"
        exit 0
    else
        echo "❌ 错误: 未找到env.example文件"
        exit 1
    fi
fi

# 检查必需的环境变量
source .env

if [ -z "$PRIVATE_KEY" ] || [ "$PRIVATE_KEY" = "your_private_key_here" ]; then
    echo "❌ 错误: 请在.env文件中设置PRIVATE_KEY"
    exit 1
fi

if [ -z "$PINATA_API_KEY" ] || [ "$PINATA_API_KEY" = "your_pinata_api_key" ]; then
    echo "❌ 错误: 请在.env文件中设置PINATA_API_KEY"
    exit 1
fi

if [ -z "$PINATA_SECRET_API_KEY" ] || [ "$PINATA_SECRET_API_KEY" = "your_pinata_secret_api_key" ]; then
    echo "❌ 错误: 请在.env文件中设置PINATA_SECRET_API_KEY"
    exit 1
fi

echo "✅ 环境配置检查通过"

echo ""
echo "🔨 编译智能合约..."
npm run compile

echo ""
echo "🧪 测试智能合约..."
npm run test:contract

echo ""
echo "🎯 选择部署网络:"
echo "1) Mumbai (Polygon测试网) - 推荐用于测试"
echo "2) Polygon (主网) - 生产环境"
echo "3) 跳过部署，仅测试功能"

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo "🚀 部署到Mumbai测试网..."
        npm run deploy:testnet
        ;;
    2)
        echo "🚀 部署到Polygon主网..."
        echo "⚠️  警告: 这将消耗真实的MATIC代币"
        read -p "确认继续? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            npm run deploy
        else
            echo "部署已取消"
        fi
        ;;
    3)
        echo "⏭️  跳过部署"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "📚 使用说明:"
echo "=========="
echo ""
echo "1. 上传内容到IPFS和区块链:"
echo "   node scripts/upload-content.js single <文件路径>"
echo "   node scripts/upload-content.js batch <目录路径>"
echo ""
echo "2. 构建Hugo网站并注入CID:"
echo "   hugo"
echo "   node scripts/add-cid-to-hugo.js all"
echo ""
echo "3. 自动化部署 (推荐):"
echo "   node scripts/auto-deploy.js full mumbai ../content"
echo "   node scripts/auto-deploy.js quick"
echo ""
echo "4. 查看合约信息:"
echo "   node scripts/upload-content.js info"
echo ""
echo "📖 详细文档: README.md"
echo "🔗 区块链索引: docs/blockchain-index.html (部署后生成)"
echo ""
echo "🎉 快速启动完成！" 