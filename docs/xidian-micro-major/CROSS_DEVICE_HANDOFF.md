# 跨设备协作交接

> 接收方：另一台设备上的 Claude
> 发送方：当前设备（已完成微专业笔记整理）
> 日期：2026-06-30

---

## 一、你是谁、你在哪

你是另一台设备上的 Claude，正在这个仓库（Embodied AI Lab）里工作。你的任务是**深度学习实验的设计和实现**。

## 二、当前设备的我已经做了什么

当前设备负责**微专业笔记整理**，已完成：

| 模块 | 文件 | 状态 |
|------|------|------|
| 感知基础 | `具身智能感知基础：机器视觉与深度学习` | ✅ 中英双语 |
| 感知基础 | `What_is_neural_network` | ✅ 中英双语 |
| 感知基础 | `From_traditional_image_processing_to_deep_learning` | ✅ 中英双语 |
| 感知基础 | `What_is_Convolution_network` | ✅ 中英双语 |
| 感知基础 | `Commonly_used_convolution_operators` | ✅ 中英双语 |
| 认知进阶 | `How_to_Build_and_Train_a_Neural_Network_p1` | ✅ 中英双语 |
| 认知进阶 | `How_to_Build_and_Train_a_Neural_Network_p2` | ✅ 中英双语 |
| 认知进阶 | `CLAUDE_WORKFLOW.md` | ✅ gitignored（工作流模板） |

已推送到远程 `origin/main`。

**最新更新（2026-06-30）：**
| 实践环节 | `01-Python快速入门` | ✅ 中英双语 |
| 实践环节 | `02-MNIST实验手册` | ✅ 中英双语 + 可运行代码 |
| 仓库配置 | `.gitignore` 添加 PDF 忽略规则 | ✅ |

当前远程 HEAD: `f7adffa`。

## 三、工作分区（避免冲突的关键）

### 互不侵犯原则

| 目录 | 负责人 | 说明 |
|------|--------|------|
| `docs/xidian-micro-major/` | 当前设备（我） | 笔记整理，只改 `.md` 文件 |
| `experiments/` | **你（另一台设备）** | 深度学习实验设计与实现 |
| `shared/` | 协商 | 如果写共享代码，先沟通 |
| `tests/` | 各自负责 | 各自为自己的代码写测试 |
| `.gitignore` | 任一设备 | 改之前确认没有冲突 |

### 绝对不要碰的

- **你不要动**：`docs/xidian-micro-major/` 下的任何文件
- **我不要动**：`experiments/` 下你做实验的目录
- **谁都不要动**：`CLAUDE.md`、`AGENTS.md`、`docs/superpowers/`

## 四、Git 协作协议

### 每次工作前

```bash
git pull origin main --rebase
```

如果 rebase 有冲突但你不涉及冲突文件 → 让我这边处理，你先 `git rebase --abort`，切到你的实验分支工作。

### 每次工作后

```bash
git add <你改的文件>
git commit -m "<描述你的改动>"
git pull origin main --rebase   # 拉取我可能已经推的改动
git push origin main
```

### 关键规则

1. **小批量提交**：完成一个实验或一个功能就 commit + push，不要攒很多
2. **先 pull 再 push**：永远先拉取远程最新代码
3. **rebase 不要 merge**：用 `--rebase` 保持线性历史
4. **冲突时不要强行 resolve**：如果你不确定，先 push 你当前的，让我来处理冲突
5. **push 前确认**：`git status` 确保没有误改我的文件

### commit message 格式

```
<type>(<scope>): <描述>
```

- 你的 type：`feat`（新实验）、`fix`（修bug）、`refactor`（重构实验代码）
- 你的 scope：`experiments/<方向>`，如 `experiments/01-perception`
- 示例：`feat(experiments/01-perception): add CNN training experiment for CIFAR-10`

## 五、当前仓库状态

```
远程: origin/main @ f7adffa
本地: main @ f7adffa
状态: 与远程同步，工作区干净（仅课程笔记 .md 有本地未提交修改）
```

仓库结构概览：
```
experiments/
├── 01-perception/       ← 感知（你可能有工作）
├── 02-slam-navigation/  ← SLAM
├── 03-motion-control/   ← 运动控制
├── 04-rl-imitation/     ← 强化学习
└── ...
docs/xidian-micro-major/ ← 我的领地，你不要碰
├── 具身智能感知基础：机器学习与深度学习/
├── 具身智能认知进阶：多模态大模型与语义理解/
└── 实践环节：Python基础与MNIST实战/   ← 新增，含可运行代码
shared/                   ← 共享代码
tests/                    ← 测试
```

## 六、笔记整理工作流参考

如果我这边要继续整理笔记，规则写在 `docs/xidian-micro-major/CLAUDE_WORKFLOW.md`（已 gitignored，你需要看的话让我发给你）。你也可以参考它做实验文档的编写。

## 七、遇到问题怎么办

| 情况 | 处理 |
|------|------|
| 改完 push 发现我比你快了一 commit | `git pull --rebase` 再 push，`.gitignore` 和笔记文件不会跟你的实验代码冲突 |
| rebase 冲突在你没碰过的文件 | `git rebase --abort`，告诉我，我来处理 |
| rebase 冲突在你的实验文件 | 正常 resolve，保留你的改动 |
| 不确定某个文件该不该改 | 不要改，先 commit push 你已经做完的，然后问我 |
| 需要改 `shared/` 或 `.gitignore` | 先 push 当前工作，再改，改完立即 push，通知我 |

---

*这份交接文档当前设备已推送到远程。另一台设备的 Claude：请先 `git pull` 获取本文档，然后开始你的实验工作。*
