const hre = require("hardhat");

async function main() {
  console.log("开始部署智能合约...");

  // 获取部署账户
  const [deployer] = await ethers.getSigners();
  console.log("部署账户:", deployer.address);
  console.log("账户余额:", (await deployer.getBalance()).toString());

  // 部署BlogContract
  const BlogContract = await hre.ethers.getContractFactory("BlogContract");
  const blogContract = await BlogContract.deploy();
  await blogContract.deployed();

  console.log("BlogContract 已部署到:", blogContract.address);

  // 等待几个区块确认
  console.log("等待区块确认...");
  await blogContract.deployTransaction.wait(5);

  console.log("部署完成!");
  console.log("合约地址:", blogContract.address);
  console.log("网络:", hre.network.name);
  
  // 验证合约（如果在支持的网络上）
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("开始验证合约...");
    try {
      await hre.run("verify:verify", {
        address: blogContract.address,
        constructorArguments: [],
      });
      console.log("合约验证成功!");
    } catch (error) {
      console.log("合约验证失败:", error.message);
    }
  }

  // 保存部署信息
  const deploymentInfo = {
    network: hre.network.name,
    contractAddress: blogContract.address,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber()
  };

  const fs = require("fs");
  const path = require("path");
  const deploymentPath = path.join(__dirname, "../deployments");
  
  if (!fs.existsSync(deploymentPath)) {
    fs.mkdirSync(deploymentPath, { recursive: true });
  }
  
  fs.writeFileSync(
    path.join(deploymentPath, `${hre.network.name}.json`),
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("部署信息已保存到:", path.join(deploymentPath, `${hre.network.name}.json`));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 