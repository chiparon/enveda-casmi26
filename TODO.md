# TODO

比赛：[Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)。方案和已有结果见 `README.md`。

## 现状

- 谱库检索基线已完成：留出 `enveda-np-examples`，MRR@25 = 0.934。这是已知结构的跨库检索成绩，不代表未知结构泛化。
- 全量训练集的结构隔离五折已生成（`baseline.py folds`），规范化结构跨折冲突为 0。
- 主体模型选型：谱峰 Transformer → Morgan 指纹 → 候选排序。**尚未实现。**
- `submission.csv` 已在本地生成并通过格式检查，未上传。

数据不在仓库里：从 Kaggle 下载后放到 `datas/`（`train.parquet`、`test.parquet`、`sample_submission.csv`），再运行 `baseline.py folds` 重建 `artifacts/structure_folds.parquet`。

## P0 训练环境

- [ ] 恢复 DGX Spark 连接（当前与本机不在同一网段，mDNS 和 IPv6 备用地址都不可达）。
- [ ] 在 Spark 上建 venv 装 PyTorch（aarch64 + CUDA）。系统 Python 里没有 torch，用户也没有 Docker 权限。不要动已在运行的 `ninfer-serve`。
- [ ] 本机（RTX 4070 Laptop 8GB）也装一份 PyTorch，用于调试。
- [ ] 两台机器用同一批谱图、同一模型、BF16 各跑一次短测，比较谱图数/秒，再决定正式训练放哪台。

## P1 谱峰 Transformer

- [ ] 数据管线：从 `train.parquet` 流式读取，每张谱图保留最强 N 个峰（基线用 128），输入 m/z、强度（平方根变换）、中性丢失；元数据为加合物、正负离子模式、碰撞能量。正负离子都要覆盖。
- [ ] 训练目标：Morgan 指纹（先用 radius 2、2048 位），多标签 BCE。标签用 `evaluation.py` 里同一套 RDKit 规范化。
- [ ] 模型：6 层，256 维，8 头，作为起点。
- [ ] 划分：严格按 `structure_folds.parquet` 的 `fold`。验证折结构的全部谱图都不能进训练。
- [ ] 分子级融合：同一 `molecule_id` 的多张谱图预测取平均（或做注意力池化），再做排序。

## P2 候选排序与评估

- [ ] 候选库：先用训练集结构，按 10 种加合物换算中性质量，在 ppm 窗口内筛选。
- [ ] 打分：预测指纹与候选指纹的相似度（Tanimoto / 余弦），输出前 25 个。
- [ ] 在结构隔离折上算 MRR@25，必须用 `evaluation.py`，它与官方结构匹配逻辑一致。这是衡量“没有参考谱图的分子”的主要指标。
- [ ] 与检索分支融合（分数加权或学习排序），在两种验证上各报一次：跨库检索验证、结构隔离验证。

## P3 覆盖率与后续

- [ ] 候选库覆盖率：统计验证/测试分子有多少不在训练结构里。不在库中的结构，指纹排序找不出来，需要外部结构库（如 PubChem / COCONUT 子集，注意许可）。
- [ ] DreaMS 迁移学习对照：其预训练只覆盖正离子、m/z ≤ 1000，负离子和大分子要单独评估。
- [ ] 全新结构的生成分支（暂缓，等 P2 有了结构隔离分数再定）。

## P4 提交

- [ ] 打包成 Kaggle Notebook：代码、依赖（RDKit `2026.03.3`）、模型权重。隐藏测试会替换 `test.parquet`，分子 ID 必须从当次文件读取。
- [ ] 提交前用 `baseline.py check` 检查 ID 覆盖、候选数和 SMILES 有效性。

## 约定

- 不把数据或由数据派生的结构标签推到仓库（见 `.gitignore`）。
- 报分数时写清楚是哪种验证，不要把跨库检索分数当成泛化分数。
- 改动后跑 `python -m unittest -v`。
