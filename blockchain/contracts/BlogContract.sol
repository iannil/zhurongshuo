// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract BlogContract {
    struct Article {
        string contentCid;
        address author;
        uint256 timestamp;
        string title;
        string description;
    }
    
    mapping(uint256 => Article) public articles;
    uint256 public articleCount;
    address public owner;
    
    event ArticlePosted(
        uint256 indexed articleId,
        string contentCid,
        address indexed author,
        uint256 timestamp,
        string title,
        string description
    );
    
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        articleCount = 0;
    }
    
    function postArticle(
        string memory _contentCid,
        string memory _title,
        string memory _description
    ) public onlyOwner {
        articleCount++;
        
        articles[articleCount] = Article({
            contentCid: _contentCid,
            author: msg.sender,
            timestamp: block.timestamp,
            title: _title,
            description: _description
        });
        
        emit ArticlePosted(
            articleCount,
            _contentCid,
            msg.sender,
            block.timestamp,
            _title,
            _description
        );
    }
    
    function getArticle(uint256 _articleId) public view returns (
        string memory contentCid,
        address author,
        uint256 timestamp,
        string memory title,
        string memory description
    ) {
        require(_articleId > 0 && _articleId <= articleCount, "Article does not exist");
        Article memory article = articles[_articleId];
        return (
            article.contentCid,
            article.author,
            article.timestamp,
            article.title,
            article.description
        );
    }
    
    function getAllArticles() public view returns (uint256[] memory) {
        uint256[] memory articleIds = new uint256[](articleCount);
        for (uint256 i = 1; i <= articleCount; i++) {
            articleIds[i - 1] = i;
        }
        return articleIds;
    }
    
    function transferOwnership(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "New owner cannot be zero address");
        address oldOwner = owner;
        owner = _newOwner;
        emit OwnershipTransferred(oldOwner, _newOwner);
    }
    
    function getArticleCount() public view returns (uint256) {
        return articleCount;
    }
} 