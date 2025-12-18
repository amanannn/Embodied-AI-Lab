#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)" || { echo "❌ 不在 Git 仓库中"; exit 1; }
BRANCH=$(git branch --show-current)
REMOTE="origin"
echo "🚀 当前分支: $BRANCH"
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "⚠️  工作区干净，无更改需要提交"
    exit 0
else
    echo "📝 添加所有更改（包括新文件）..."
    git add .
fi
COMMIT_MSG="${1:-chore: auto-commit}"
echo "📨 提交信息: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"
echo "📤 推送到 $REMOTE/$BRANCH ..."
git push
echo "✅ 提交并推送成功！"
