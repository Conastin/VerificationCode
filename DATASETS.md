# 数据集说明

本项目公开随仓库分发全部真实采集数据集。**所有图片均为验证码字符图像（字母+数字），
不含任何个人信息、凭据或会话数据**；采集均基于作者所在**已授权测试环境**。详见 [免责声明](README.md#️-免责声明)。

## 一览

| 目录 | 内容 | 样本数 | 标注来源 | 大小 |
|---|---|---|---|---|
| `data/real_test_independent/` | 旧站独立测试集（永久留出，勿用于训练） | 1000 | 人工全量核对 | 4.3MB |
| `data/real_human/` | 旧站人工精确标注子集 | 1791 | 人工 | 7.4MB |
| `data/real_all/` | 旧站训练集（人工核对过） | 10201 | 人工核对+置信度筛选 | 42MB |
| `data/old_slot_labeled*/` | 旧站 slot 模型自动标注可信子集（conf≥0.99） | ~2934 | 自动（高置信） | 12.4MB |
| `data/portal_label/` | portal 人工标注：train 240 / **test 60（永久留出）** / 难例 4 轮 346 / 重标 27 | 673 | 人工 | 4.0MB |
| `data/portal_dataset/` | portal 金标准归档（6 轮飞轮 + 3 场实时人在环） | ~8900 | **服务器判定**（预测=真值） | 36MB |
| `checkpoints/universal_crnn_ft_portal_v11.pt` | 通用模型 v11 权重（定稿） | — | — | 11MB |
| `checkpoints/best.pt` | 旧站专用 slot 模型权重（99.9%） | — | — | 28MB |

不随仓库分发：`data/train31|val31`（237MB 合成数据，可用 `synth_universal.py` 完全复现）、
其余实验中间 checkpoint。

## 标注可靠性分级

1. **服务器判定金标准**（`portal_dataset/server_confirmed/`）：登录接口返回"验证码通过"，
   意味着模型预测即真值——比人工标注更可靠。其中 `H*` 前缀文件为"人工书写+服务器确认"双保险。
2. **人工标注**（`portal_label/`）：`test/` 为永久留出测试集，任何训练不得使用；
   `hard*/` 为模型识别失败的人工标注难例（最有价值的训练信号）。
3. **高置信自动标注**（`old_slot_labeled*/`）：旧模型 conf≥0.99 的预测，实测准确率与人工一致。

## 目录约定

- `labels.txt`：`<文件名> <标签>` 每行一条（portal 标签保留原始大小写，训练时折叠大写——
  目标站点登录校验大小写不敏感）
- `manifest.json` / `MANIFEST.md`：批次元数据（时间、来源 checkpoint、计数）
- `failed.csv` / `failed_submissions_DO_NOT_TRAIN.txt`：被服务器拒绝的提交记录，
  **含错误标签，禁止用于训练**（文件名已显式标注）
- 归档纪律：append-only、SHA-256 去重、永不覆盖

## 复现训练

```bash
# 通用模型（portal + 旧站混合微调）
python -m verification_code.finetune_universal \
    --checkpoint <基线> --steps 8000 --real-prob 0.65 \
    --real "data/portal_dataset/server_confirmed/<批次>:权重" \
           "data/portal_label/train:权重" ...
```

数据飞轮的完整方法论见 [AGENTS.md](AGENTS.md)。
