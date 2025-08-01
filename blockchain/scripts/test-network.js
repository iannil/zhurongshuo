const { ethers } = require("hardhat");

async function testNetwork() {
  console.log("🔍 测试网络连接...");
  
  const networks = [
    {
      name: "Mumbai (Alchemy)",
      url: "https://polygon-mumbai.g.alchemy.com/v2/demo"
    },
    {
      name: "Mumbai (Infura)",
      url: "https://polygon-mumbai.infura.io/v3/9aa3d95b3bc440fa88ea12eaa4456161"
    },
    {
      name: "Mumbai (MaticVigil)",
      url: "https://rpc-mumbai.maticvigil.com"
    },
    {
      name: "Polygon (Alchemy)",
      url: "https://polygon-mainnet.g.alchemy.com/v2/demo"
    }
  ];

  for (const network of networks) {
    try {
      console.log(`\n📡 测试 ${network.name}...`);
      const provider = new ethers.providers.JsonRpcProvider(network.url);
      
      // 测试基本连接
      const chainId = await provider.getNetwork();
      console.log(`✅ ${network.name} 连接成功! Chain ID: ${chainId.chainId}`);
      
      // 测试获取最新区块
      const blockNumber = await provider.getBlockNumber();
      console.log(`📦 最新区块: ${blockNumber}`);
      
    } catch (error) {
      console.log(`❌ ${network.name} 连接失败: ${error.message}`);
    }
  }
}

testNetwork()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("测试失败:", error);
    process.exit(1);
  }); 