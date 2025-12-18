#!/bin/bash
set -e
BRANCH=$(git branch --show-current)
REMOTE="origin"
echo "🚀 当前分支: $BRANCH"
if ! git diff --quiet || ! git ls-files --others --exclude-standard | grep -q .; then
    if ! git diff --cached --quiet; then
        echo "✅ 有更改待提交"
    else
        echo "⚠️  工作区干净，无更改需要提交"
        exit 0
    fi
else
    echo "📝 添加所有更改..."
    git add .
fi
COMMIT_MSG="${1:-chore: auto-commit}"
echo "📨 提交信息: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"
echo "📤 推送到 $REMOTE/$BRANCH ..."
git push
echo "✅ 提交并推送成功！"
