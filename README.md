### 若服务器无梯子，使用ssh反向端口，在本地开梯子下载
$env:HTTPS_PROXY="http://127.0.0.1:7890"

### 下载代码
#### 方式一：
git clone https://<token>@github.com/Lijwwww/MAGame.git（私有）
git clone https://github.com/Lijwwww/MAGame.git（公有）

#### 方式二：直接
git clone https://gh-proxy.com/https://github.com/Lijwwww/MAGame.git 代替前两步
git pull origin main

#### 方式三：若是下载压缩包的
git init
git remote add origin https://github.com/Lijwwww/MAGame.git 或 https://<token>@github.com/Lijwwww/MAGame.git
1. 抓取远程所有分支信息
git fetch origin
2. 尝试强制合并远程 main 分支
git merge origin/main --allow-unrelated-histories
git reset --hard origin/main
git pull origin main


### 上传
git add .
git commit -m "Backup at $(date +'%Y-%m-%d %H:%M:%S')"（linux）
git commit -m "Backup at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"  （windows）
git pull origin main
git push origin main