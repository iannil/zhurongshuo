const { ethers } = require("hardhat");

async function main() {
  console.log("=== 智能合约测试 ===");

  // 获取测试账户
  const [owner, user1, user2] = await ethers.getSigners();
  console.log("测试账户:");
  console.log("  所有者:", owner.address);
  console.log("  用户1:", user1.address);
  console.log("  用户2:", user2.address);

  // 部署合约
  console.log("\n1. 部署智能合约...");
  const BlogContract = await ethers.getContractFactory("BlogContract");
  const blogContract = await BlogContract.deploy();
  await blogContract.deployed();
  console.log("合约地址:", blogContract.address);

  // 测试基本功能
  console.log("\n2. 测试基本功能...");
  
  // 检查初始状态
  const initialCount = await blogContract.getArticleCount();
  console.log("初始文章数量:", initialCount.toString());
  
  const contractOwner = await blogContract.owner();
  console.log("合约所有者:", contractOwner);
  console.log("当前调用者:", owner.address);
  console.log("所有者匹配:", contractOwner === owner.address);

  // 测试发布文章 (所有者)
  console.log("\n3. 测试发布文章 (所有者)...");
  const testCid = "QmTest123456789abcdef";
  const testTitle = "测试文章标题";
  const testDescription = "这是一个测试文章的描述";
  
  const tx = await blogContract.postArticle(testCid, testTitle, testDescription);
  console.log("发布文章交易哈希:", tx.hash);
  
  const receipt = await tx.wait();
  console.log("交易确认，区块号:", receipt.blockNumber);
  
  // 检查文章数量
  const newCount = await blogContract.getArticleCount();
  console.log("发布后文章数量:", newCount.toString());

  // 获取文章信息
  console.log("\n4. 获取文章信息...");
  const article = await blogContract.getArticle(1);
  console.log("文章1信息:");
  console.log("  CID:", article.contentCid);
  console.log("  作者:", article.author);
  console.log("  标题:", article.title);
  console.log("  描述:", article.description);
  console.log("  时间戳:", new Date(article.timestamp * 1000).toLocaleString());

  // 测试非所有者发布文章 (应该失败)
  console.log("\n5. 测试非所有者发布文章 (应该失败)...");
  try {
    const nonOwnerTx = await blogContract.connect(user1).postArticle(
      "QmNonOwnerTest",
      "非所有者文章",
      "非所有者发布的文章"
    );
    console.log("❌ 非所有者发布成功 (不应该发生)");
  } catch (error) {
    console.log("✅ 非所有者发布失败 (符合预期):", error.message);
  }

  // 测试发布多篇文章
  console.log("\n6. 测试发布多篇文章...");
  const articles = [
    { cid: "QmArticle1", title: "文章1", description: "第一篇文章" },
    { cid: "QmArticle2", title: "文章2", description: "第二篇文章" },
    { cid: "QmArticle3", title: "文章3", description: "第三篇文章" }
  ];

  for (let i = 0; i < articles.length; i++) {
    const article = articles[i];
    console.log(`发布文章 ${i + 2}: ${article.title}`);
    
    const tx = await blogContract.postArticle(article.cid, article.title, article.description);
    await tx.wait();
  }

  // 获取所有文章
  console.log("\n7. 获取所有文章...");
  const allArticles = await blogContract.getAllArticles();
  console.log("所有文章ID:", allArticles.map(id => id.toString()));

  // 显示所有文章信息
  for (let i = 1; i <= allArticles.length; i++) {
    const article = await blogContract.getArticle(i);
    console.log(`\n文章 ${i}:`);
    console.log("  标题:", article.title);
    console.log("  CID:", article.contentCid);
    console.log("  作者:", article.author);
    console.log("  时间:", new Date(article.timestamp * 1000).toLocaleString());
  }

  // 测试所有权转移
  console.log("\n8. 测试所有权转移...");
  const transferTx = await blogContract.transferOwnership(user1.address);
  await transferTx.wait();
  
  const newOwner = await blogContract.owner();
  console.log("新所有者:", newOwner);
  console.log("转移成功:", newOwner === user1.address);

  // 测试新所有者发布文章
  console.log("\n9. 测试新所有者发布文章...");
  const newOwnerTx = await blogContract.connect(user1).postArticle(
    "QmNewOwnerArticle",
    "新所有者文章",
    "新所有者发布的文章"
  );
  await newOwnerTx.wait();
  console.log("✅ 新所有者发布文章成功");

  // 测试原所有者发布文章 (应该失败)
  console.log("\n10. 测试原所有者发布文章 (应该失败)...");
  try {
    const oldOwnerTx = await blogContract.connect(owner).postArticle(
      "QmOldOwnerArticle",
      "原所有者文章",
      "原所有者发布的文章"
    );
    console.log("❌ 原所有者发布成功 (不应该发生)");
  } catch (error) {
    console.log("✅ 原所有者发布失败 (符合预期):", error.message);
  }

  // 最终统计
  console.log("\n=== 测试完成 ===");
  const finalCount = await blogContract.getArticleCount();
  console.log("最终文章数量:", finalCount.toString());
  console.log("合约地址:", blogContract.address);
  console.log("当前所有者:", await blogContract.owner());

  return {
    contractAddress: blogContract.address,
    articleCount: finalCount.toString(),
    owner: await blogContract.owner()
  };
}

// 运行测试
if (require.main === module) {
  main()
    .then((result) => {
      console.log("\n测试结果:", result);
      process.exit(0);
    })
    .catch((error) => {
      console.error("测试失败:", error);
      process.exit(1);
    });
}

module.exports = { main }; 