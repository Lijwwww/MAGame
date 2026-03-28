TOKEN_FILE="scripts/backup/.tokens.env"
source "$TOKEN_FILE"

# 1. 提交代码到 GitHub
echo "Pushing code to GitHub..."
# git init
# git branch -M main
# git remote add origin https://github.com/Lijwwww/MAGame.git

# github官网->Settings->Developer settings获取，放入上述路径的GITHUB_TOKEN变量中
git remote set-url origin https://${GITHUB_TOKEN}@github.com/Lijwwww/MAGame.git

git add .
git commit -m "Backup at $(LANG=en_US date '+%Y-%m-%d %H:%M:%S')"
git push origin main