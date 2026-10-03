# Enveda CASMI 2026

根据 MS/MS 谱图预测分子结构，每个 `molecule_id` 输出最多 25 个有序 SMILES，按 MRR@25 评分。

## 主体模型选择

选择轻量谱峰 Transformer：6 层、256 隐藏维度、8 个注意力头。输入为谱峰 m/z、强度、中性丢失及加合物、离子模式、碰撞能量信息；预测分子指纹，再对质量约束下的候选结构排序。先使用 Morgan 指纹作为目标，多谱信息在分子级融合。

这是模型选型，不是已训练权重。本目录当前可执行的是谱库检索基线、验证和提交流程。没有创建空的神经网络模块，也没有启动长时间训练。

不把原版 DreaMS 预训练权重直接当作全覆盖方案：[原论文 Methods](https://www.nature.com/articles/s41587-025-02663-3) 的预训练筛选只保留正离子谱图，并排除 m/z > 1000 的谱图。比赛同时包含正负离子和更大的分子；DreaMS 可作为后续迁移学习对照，其负离子和质量范围外表现必须单独验证。

检索和指纹排序都受候选库覆盖率限制。训练集外、但结构已知的分子需要外部结构库；真正全新的结构需要生成模型。这份基线不声称解决后两类问题。

## 本地数据

原始文件保留在 `datas/`，不修改、不纳入版本控制：

- `train.parquet`：实测 2,539,608 张谱图，275,810 个原始 `inchikey14`。
- `test.parquet`：当前可见文件为 1,213 张谱图、400 个分子。
- `sample_submission.csv`：官方格式示例，不作为预测来源。

当前可见测试集是训练数据中的示例，只用于跑通流程。隐藏测试集在 Kaggle 重跑时替换；预测命令直接从当次 `test.parquet` 获取分子 ID，不依赖示例提交中的固定 ID。

## 安装与运行

Python 3.12。以下命令在本目录的 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe baseline.py audit
.\.venv\Scripts\python.exe baseline.py folds
.\.venv\Scripts\python.exe baseline.py validate
.\.venv\Scripts\python.exe baseline.py predict
.\.venv\Scripts\python.exe baseline.py check submission.csv
.\.venv\Scripts\python.exe -m unittest -v
```

已存在 `.venv` 时无需重新创建。RDKit 固定为 `2026.03.3`，与官方结构匹配版本一致。

更换数据位置时，将全局参数放在子命令前：

```powershell
.\.venv\Scripts\python.exe baseline.py --data-dir D:\casmi-data predict --output submission.csv
```

这些命令不需要联网，不会注册比赛、上传数据或自动提交。Kaggle 上仍须将代码和依赖打包到 Notebook 中运行；本地 CSV 不是已完成的 Kaggle 提交。

## 已实现的检索方法

1. 根据 10 种比赛加合物，把 precursor m/z 转为中性质量。
2. 流式读取训练谱库，先用同离子极性及默认 20 ppm 中性质量窗口筛选。
3. 保留最强 128 个峰，进行平方根强度变换及归一化。
4. 在默认 0.02 Da 峰匹配容差内，使用一对一贪心峰匹配计算余弦相似度。
5. 每个候选结构对每张查询谱取最佳参考谱分数，再对同分子的查询谱求均值；未匹配的谱贡献 0。
6. 按得分排序，结构去重，输出最多 25 个候选；写提交时再次按官方互变异构体规范化后的结构标识去重。

没有谱图证据的候选不补入列表。若某个分子完全找不到候选，报告中保留其 ID；提交生成会明确报错，而不是填入 `CCO` 等假答案。训练中不支持的加合物和无效谱图会计数；不支持的查询加合物会报错。

可调整 `--ppm`、`--peak-tolerance`、`--max-peaks`、`--top-k`、`--batch-size`。这些是基线参数，尚未经过调参。

## 验证如何解释

### 跨谱库检索验证

`validate` 留出整个 `enveda-np-examples` 库：实测 1,184 张谱图、250 个结构。检索时排除该库的所有谱图，查询对象不包含 SMILES 标签。

其他谱库中允许存在同一分子的独立测量。因此，这个分数衡量的是**已知结构的跨谱库检索**，不是未知结构泛化能力，也不是 Kaggle 隐藏测试分数。

本机默认参数实测（250 个分子）：

| 指标 | 结果 |
|---|---:|
| MRR@25 | 0.9341889 |
| Top-1 | 89.6% |
| Top-5 | 98.0% |
| Top-25 | 99.2% |

当前可见测试的 400 个分子均已生成候选，提交文件共 9,573 个候选，每个分子 1–25 个；ID 覆盖和化学结构有效性检查通过。不报告该可见测试集的预测准确率。

### 神经网络的结构隔离划分

`folds` 对全部训练标签执行 RDKit 互变异构体规范化和 InChIKey14 转换。实际数据中，同一个原始键的不同表示可能得到多个规范化键；因此将共享原始键或规范化键的结构按连通关系合并，再对组做固定种子的稳定哈希，分为 5 折。这样既保留全部谱图，又避免这些相互关联的表示跨折。

输出每个原始 `inchikey14` 一行，包含 `structure_keys`（实际规范化键列表）、`split_group`（连通组标识）和 `fold`。统计中记录 `ambiguous_raw_structures`，不修改原始标签或官方评分逻辑。RDKit 的默认枚举上限和诊断信息保持不变。

全量运行得到 274,195 个规范化键、274,184 个隔离组，发现并合并了 11 个存在多规范化键的原始结构。已检查全部 2,539,608 张训练谱图均有折号，且规范化键跨折冲突数为 0。5 个回归测试通过，包含真实冲突分子的修复前失败、修复后通过案例。

后续模型训练时，验证折结构的全部标注谱图必须从训练中移除。若评估“已知结构、无参考谱图”，候选结构库可以保留该结构，但不得将其谱图或预测目标用于训练。

## 输出

- `artifacts/data_audit.json`：真实数据规模、来源、模式及加合物分布。
- `artifacts/structure_folds.parquet` / `.json`：结构隔离映射及统计。
- `artifacts/validation_metrics.json`：跨谱库 MRR@25、Top-1/5/25 和检索覆盖率。
- `artifacts/validation_predictions.json`：可审查的验证候选。
- `artifacts/prediction_report.json`：生成提交的参数、覆盖率及耗时。
- `submission.csv`：分子级提交文件，经过 ID、候选数量及 SMILES 有效性检查。

## 官方来源

- [比赛概览与评价指标](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)
- [数据说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)
- [比赛规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)

数据限比赛及非商业用途；不要将原始数据推送到公开代码仓库。
