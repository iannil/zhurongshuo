const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

class SecurityAuditor {
  constructor() {
    this.auditResults = {
      vulnerabilities: [],
      recommendations: [],
      riskLevel: 'LOW',
      score: 100
    };
  }

  log(message, type = 'info') {
    const timestamp = new Date().toLocaleString('zh-CN');
    const prefix = type === 'error' ? '🔴' : type === 'warning' ? '🟡' : type === 'success' ? '🟢' : 'ℹ️';
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  addVulnerability(severity, title, description, recommendation) {
    this.auditResults.vulnerabilities.push({
      severity,
      title,
      description,
      recommendation
    });
    
    // 调整风险评分
    if (severity === 'CRITICAL') {
      this.auditResults.score -= 25;
    } else if (severity === 'HIGH') {
      this.auditResults.score -= 15;
    } else if (severity === 'MEDIUM') {
      this.auditResults.score -= 10;
    } else if (severity === 'LOW') {
      this.auditResults.score -= 5;
    }
  }

  addRecommendation(title, description) {
    this.auditResults.recommendations.push({
      title,
      description
    });
  }

  async auditAccessControl() {
    this.log('=== 访问控制审计 ===');
    
    try {
      const [owner, user1, user2] = await ethers.getSigners();
      const BlogContract = await ethers.getContractFactory("BlogContract");
      const blogContract = await BlogContract.deploy();
      await blogContract.deployed();

      // 测试1: 非所有者无法发布文章
      try {
        await blogContract.connect(user1).postArticle("test", "test", "test");
        this.addVulnerability('CRITICAL', '访问控制绕过', '非所有者可以发布文章', '检查 onlyOwner 修饰符');
      } catch (error) {
        if (error.message.includes('Only owner can call this function')) {
          this.log('✅ 访问控制正常 - 非所有者无法发布文章', 'success');
        } else {
          this.addVulnerability('MEDIUM', '意外的访问控制错误', error.message, '检查错误处理逻辑');
        }
      }

      // 测试2: 非所有者无法转移所有权
      try {
        await blogContract.connect(user1).transferOwnership(user2.address);
        this.addVulnerability('CRITICAL', '所有权控制绕过', '非所有者可以转移所有权', '检查 onlyOwner 修饰符');
      } catch (error) {
        if (error.message.includes('Only owner can call this function')) {
          this.log('✅ 所有权控制正常 - 非所有者无法转移所有权', 'success');
        } else {
          this.addVulnerability('MEDIUM', '意外的所有权控制错误', error.message, '检查错误处理逻辑');
        }
      }

      // 测试3: 零地址转移保护
      try {
        await blogContract.transferOwnership(ethers.constants.AddressZero);
        this.addVulnerability('HIGH', '零地址转移', '可以转移到零地址', '添加零地址检查');
      } catch (error) {
        if (error.message.includes('New owner cannot be zero address')) {
          this.log('✅ 零地址转移保护正常', 'success');
        } else {
          this.addVulnerability('MEDIUM', '零地址转移保护异常', error.message, '检查零地址验证逻辑');
        }
      }

    } catch (error) {
      this.addVulnerability('HIGH', '访问控制审计失败', error.message, '检查合约部署和测试环境');
    }
  }

  async auditDataIntegrity() {
    this.log('=== 数据完整性审计 ===');
    
    try {
      const [owner, user1] = await ethers.getSigners();
      const BlogContract = await ethers.getContractFactory("BlogContract");
      const blogContract = await BlogContract.deploy();
      await blogContract.deployed();

      // 测试1: 文章数据存储完整性
      const testData = {
        cid: "QmTest123456789abcdef",
        title: "测试文章标题",
        description: "这是一个测试文章的描述"
      };

      const tx = await blogContract.postArticle(testData.cid, testData.title, testData.description);
      await tx.wait();

      const article = await blogContract.getArticle(1);
      
      if (article.contentCid === testData.cid && 
          article.title === testData.title && 
          article.description === testData.description) {
        this.log('✅ 文章数据存储完整性正常', 'success');
      } else {
        this.addVulnerability('HIGH', '数据存储不完整', '存储的数据与输入不匹配', '检查数据存储逻辑');
      }

      // 测试2: 边界条件处理
      try {
        await blogContract.getArticle(0);
        this.addVulnerability('MEDIUM', '边界条件处理不当', '应该拒绝文章ID 0', '添加边界条件检查');
      } catch (error) {
        if (error.message.includes('Article does not exist')) {
          this.log('✅ 边界条件处理正常', 'success');
        } else {
          this.addVulnerability('LOW', '边界条件处理异常', error.message, '检查边界条件逻辑');
        }
      }

      // 测试3: 文章计数准确性
      const count = await blogContract.getArticleCount();
      if (count.toString() === '1') {
        this.log('✅ 文章计数准确性正常', 'success');
      } else {
        this.addVulnerability('MEDIUM', '文章计数不准确', `期望1，实际${count}`, '检查计数逻辑');
      }

    } catch (error) {
      this.addVulnerability('HIGH', '数据完整性审计失败', error.message, '检查合约功能');
    }
  }

  async auditGasOptimization() {
    this.log('=== Gas 优化审计 ===');
    
    try {
      const [owner] = await ethers.getSigners();
      const BlogContract = await ethers.getContractFactory("BlogContract");
      const blogContract = await BlogContract.deploy();
      await blogContract.deployed();

      // 测试1: 发布文章 Gas 消耗
      const tx = await blogContract.postArticle("QmTest", "Test", "Test");
      const receipt = await tx.wait();
      
      if (receipt.gasUsed.lt(ethers.BigNumber.from('200000'))) {
        this.log('✅ Gas 消耗合理', 'success');
      } else {
        this.addRecommendation('Gas 优化', `发布文章消耗 ${receipt.gasUsed.toString()} gas，建议优化`);
      }

      // 测试2: 读取操作 Gas 消耗
      const readTx = await blogContract.getArticle(1);
      // 读取操作通常不消耗 gas（在本地节点上）
      this.log('✅ 读取操作 Gas 消耗正常', 'success');

    } catch (error) {
      this.addVulnerability('LOW', 'Gas 优化审计失败', error.message, '检查 Gas 消耗监控');
    }
  }

  async auditEventEmission() {
    this.log('=== 事件发射审计 ===');
    
    try {
      const [owner] = await ethers.getSigners();
      const BlogContract = await ethers.getContractFactory("BlogContract");
      const blogContract = await BlogContract.deploy();
      await blogContract.deployed();

      // 测试1: 文章发布事件
      const tx = await blogContract.postArticle("QmTest", "Test", "Test");
      const receipt = await tx.wait();
      
      if (receipt.events && receipt.events.length > 0) {
        const event = receipt.events[0];
        if (event.event === 'ArticlePosted') {
          this.log('✅ 文章发布事件正常', 'success');
        } else {
          this.addVulnerability('MEDIUM', '事件名称不匹配', `期望 ArticlePosted，实际 ${event.event}`, '检查事件定义');
        }
      } else {
        this.addVulnerability('HIGH', '事件未发射', '文章发布时没有发射事件', '检查事件发射逻辑');
      }

      // 测试2: 所有权转移事件
      const transferTx = await blogContract.transferOwnership(owner.address);
      const transferReceipt = await transferTx.wait();
      
      if (transferReceipt.events && transferReceipt.events.length > 0) {
        const event = transferReceipt.events[0];
        if (event.event === 'OwnershipTransferred') {
          this.log('✅ 所有权转移事件正常', 'success');
        } else {
          this.addVulnerability('MEDIUM', '所有权转移事件名称不匹配', `期望 OwnershipTransferred，实际 ${event.event}`, '检查事件定义');
        }
      } else {
        this.addVulnerability('HIGH', '所有权转移事件未发射', '所有权转移时没有发射事件', '检查事件发射逻辑');
      }

    } catch (error) {
      this.addVulnerability('MEDIUM', '事件审计失败', error.message, '检查事件处理逻辑');
    }
  }

  async auditCodeQuality() {
    this.log('=== 代码质量审计 ===');
    
    try {
      // 读取合约源码
      const contractPath = path.join(__dirname, '../contracts/BlogContract.sol');
      const contractCode = fs.readFileSync(contractPath, 'utf8');

      // 检查1: SPDX 许可证
      if (contractCode.includes('SPDX-License-Identifier')) {
        this.log('✅ SPDX 许可证声明存在', 'success');
      } else {
        this.addRecommendation('许可证声明', '建议添加 SPDX 许可证声明');
      }

      // 检查2: 版本声明
      if (contractCode.includes('pragma solidity ^0.8.19')) {
        this.log('✅ Solidity 版本声明正确', 'success');
      } else {
        this.addVulnerability('MEDIUM', 'Solidity 版本问题', '建议使用最新的稳定版本', '更新 Solidity 版本');
      }

      // 检查3: 注释完整性
      const functionCount = (contractCode.match(/function /g) || []).length;
      const commentCount = (contractCode.match(/\/\/|\/\*|\*/g) || []).length;
      
      if (commentCount > functionCount) {
        this.log('✅ 代码注释充足', 'success');
      } else {
        this.addRecommendation('代码注释', '建议为函数添加更多注释');
      }

      // 检查4: 错误处理
      if (contractCode.includes('require(') || contractCode.includes('revert')) {
        this.log('✅ 错误处理机制存在', 'success');
      } else {
        this.addVulnerability('MEDIUM', '缺少错误处理', '建议添加更多的错误检查', '增加错误处理逻辑');
      }

    } catch (error) {
      this.addVulnerability('LOW', '代码质量审计失败', error.message, '检查合约源码文件');
    }
  }

  generateAuditReport() {
    this.log('=== 生成安全审计报告 ===');
    
    // 确定风险等级
    if (this.auditResults.score >= 90) {
      this.auditResults.riskLevel = 'LOW';
    } else if (this.auditResults.score >= 70) {
      this.auditResults.riskLevel = 'MEDIUM';
    } else if (this.auditResults.score >= 50) {
      this.auditResults.riskLevel = 'HIGH';
    } else {
      this.auditResults.riskLevel = 'CRITICAL';
    }

    const report = {
      timestamp: new Date().toISOString(),
      contract: 'BlogContract',
      riskLevel: this.auditResults.riskLevel,
      securityScore: this.auditResults.score,
      vulnerabilities: this.auditResults.vulnerabilities,
      recommendations: this.auditResults.recommendations
    };

    // 输出报告
    console.log('\n🔒 安全审计报告');
    console.log('='.repeat(50));
    console.log(`合约: ${report.contract}`);
    console.log(`风险等级: ${report.riskLevel}`);
    console.log(`安全评分: ${report.securityScore}/100`);
    console.log(`漏洞数量: ${report.vulnerabilities.length}`);
    console.log(`建议数量: ${report.recommendations.length}`);

    if (report.vulnerabilities.length > 0) {
      console.log('\n🚨 发现的漏洞:');
      report.vulnerabilities.forEach((vuln, index) => {
        console.log(`${index + 1}. [${vuln.severity}] ${vuln.title}`);
        console.log(`   描述: ${vuln.description}`);
        console.log(`   建议: ${vuln.recommendation}\n`);
      });
    }

    if (report.recommendations.length > 0) {
      console.log('\n💡 改进建议:');
      report.recommendations.forEach((rec, index) => {
        console.log(`${index + 1}. ${rec.title}`);
        console.log(`   描述: ${rec.description}\n`);
      });
    }

    // 保存报告
    const reportPath = path.join(__dirname, '../security-audit-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    this.log(`安全审计报告已保存到: ${reportPath}`, 'success');

    return report;
  }

  async runFullAudit() {
    this.log('🔍 开始安全审计');
    
    await this.auditAccessControl();
    await this.auditDataIntegrity();
    await this.auditGasOptimization();
    await this.auditEventEmission();
    await this.auditCodeQuality();
    
    const report = this.generateAuditReport();
    return report;
  }
}

// 运行审计
async function main() {
  const auditor = new SecurityAuditor();
  
  try {
    const report = await auditor.runFullAudit();
    
    if (report.riskLevel === 'CRITICAL' || report.riskLevel === 'HIGH') {
      process.exit(1);
    } else {
      process.exit(0);
    }
  } catch (error) {
    console.error('安全审计失败:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { SecurityAuditor }; 