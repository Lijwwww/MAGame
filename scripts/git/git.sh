# 1. 提交代码到 GitHub
echo "Pushing code to GitHub..."
git init
git branch -M main
git remote add origin https://github.com/Lijwwww/MAGame.git

git add .
git commit -m "Backup at $(date)"
git push origin main