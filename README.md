# 验证码识别器

> 60×20 四字符验证码识别模型与完整工具链：训练、评估、微调、ONNX 导出，以及浏览器扩展（自动识别填充 + 失败样本收集修正闭环）。仅用于已获授权的测试环境。

[![Version](https://img.shields.io/badge/version-2.3.0-2f6b0f)](https://github.com/Conastin/VerificationCode)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C)]()
[![ONNX](https://img.shields.io/badge/ONNX-1.17-005C97)]()
[![Chrome Extension](https://img.shields.io/badge/Chrome-MV3-blue)]()
[![Test](https://img.shields.io/badge/test-pytest-9e9e9e)]()

## 目录

- [特性](#特性)
- [性能](#性能)
- [快速开始](#快速开始)
- [数据](#数据)
- [使用](#使用)
- [浏览器扩展](#浏览器扩展)
- [数据闭环方法](#数据闭环方法难例挖掘)
- [项目结构](#项目结构)
- [运行测试](#运行测试)
- [已知边界](#已知边界)
- [免责声明](#免责声明)

## 特性

- **高精度**：独立测试集 1000 张全对 **99.9%**（char-level 99.98%）；配合"置信度 ≥0.9 才接受"拒识策略，实际准确率 100%
- **SlotCaptchaModel**：整图按固定 15px 槽位裁剪成 4 张 15×20，共享单字符 CNN 分类（全图共享 backbone 因图像过小只能到 74% 字符级，槽位裁剪达 99%+）
- **31 类字符集**：`23456789ABCDEFGHJKMNPQRSTUVWXYZ`，排除 0/1/I/L/O 易混淆字符（36 类模型难例集 88% → 97.7%）
- **数据闭环**：采集 → 置信度分级（≥0.9 直接信任 / <0.9 人工核对）→ 混合微调 → 独立测试集回归验证，实测样本库 3200 → 10200 张后准确率 98.9% → 99.9%
- **置信度校准**：≥0.9 分档经多轮独立验证零错误，生产可直接拒识
- **浏览器扩展（MV3）**：onnxruntime-web WASM 本地推理（零外网依赖），任意站点右键标记即用；**失败样本自动收集**（识别失败/提交失败）并导出 ZIP 回注训练
- **ONNX 导出**：单文件含全部权重（9.6MB），Python / JS 双端推理结果一致

## 性能

**独立测试集**：1000 张全新采集、未参与训练、全量人工核对（`data/real_test_independent`）。

```text
full-image accuracy: 0.9990   (999/1000)
char-level accuracy: 0.9998
per-slot accuracy:   [1.0000, 1.0000, 1.0000, 0.9990]

拒识阈值 0.9：覆盖率 99.0%，准确率 100%
```

**真实登录校验（服务器实测）**：使用测试账号向真实登录接口提交识别结果，由服务器验证码校验判定：

```text
50 次真实登录提交：验证码通过 50/50（100%），验证码错误 0
判定证据明细：reports/login_validation.csv
```

判定规则：响应含"验证码不正确" → 识别错误；否则验证码通过（进入凭据校验）。页面固定文案不能作为错误标记，判定只认服务器动态提示。

## 快速开始

环境：Python 3.14 + PyTorch 2.12（CUDA 13.0）+ onnxruntime

```bash
python -m venv .venv
.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
.venv/Scripts/pip install -r requirements.txt
```

## 数据

> 注意：`data/`（真实 + 合成样本，约 300MB）与 `checkpoints/`（训练权重）体积大且含内网真实样本，**不随仓库分发**（见 `.gitignore`）。按"数据闭环方法"自行采集微调，或用 `export_onnx.py` 重新导出模型。

```text
data/real_all/                  # 10200 张训练样本（真实，含 labels.txt）
data/real_human/                # 1791 张人工精确标注子集
data/real_trusted/              # 609 张置信度≥0.9 直接信任子集
data/real_test_independent/     # 1000 张独立测试集（人工标注，勿用于训练）
data/train31/ data/val31/       # 31 类合成样本（微调防遗忘用）
data/raw/                       # 历史真实样本
```

## 使用

### 评估

```bash
.venv/Scripts/python -m verification_code.evaluate \
    --checkpoint checkpoints/best.pt --data data/real_test_independent
```

### 推理（单张/多张）

```bash
.venv/Scripts/python -m verification_code.infer \
    --checkpoint checkpoints/best.pt data/xxx.jpg
```

### 采集与微调（数据闭环）

```bash
# 采集并按识别结果命名（manifest 记录置信度）
.venv/Scripts/python -m verification_code.capture_real --count 1000 \
    --out data/real_captcha_new --checkpoint checkpoints/best.pt

# 人工核对：低置信度 + 易混字符（--chars 取并集，断点续核）
.venv/Scripts/python -m verification_code.verify_captcha \
    --dir data/real_captcha_new --max-conf 0.9 --chars MN3BQC

# 整理：人工核对批次 → 人工集；--max-conf 批次按阈值拆分
.venv/Scripts/python -m verification_code.organize_real \
    --human-dirs data/real_captcha_new \
    --split-dirs data/real_captcha_new2 --threshold 0.9

# 微调（人工集 + 合成混合，防遗忘）
.venv/Scripts/python -m verification_code.finetune_real \
    --base checkpoints/best.pt --real data/real_all \
    --syn-train data/train31 --syn-val data/val31 \
    --out checkpoints/finetuned --epochs 60

# 导出 ONNX（单文件，含全部权重）
.venv/Scripts/python -m verification_code.export_onnx \
    --checkpoint checkpoints/best.pt --out checkpoints/best.onnx
```

### 真实登录校验（服务器端验证识别准确率）

```bash
.venv/Scripts/python -m verification_code.login_validate \
    --user <测试账号> --password <密码> --attempts 50 --interval 2.5
```

结果写入 `reports/login_validation.csv`。

### 其他工具

```text
analyze_confusion.py   # 易错字符/混淆对分析
convert_charset.py     # checkpoint 字符集转换（36→31 类）
login_validate.py      # 真实登录校验验证（服务器端判定识别准确率）
slot_crop_check.py     # 单字符裁剪可行性诊断
vlm_eval.py            # 多模态 LLM 识别实测（结论：不如专用 CNN）
```

## 浏览器扩展

`browser/extension/` 是 Manifest V3 扩展（**验证码自动识别填充**），**不绑定站点**：任意页面的验证码图片上右键 → "标记为验证码并自动填充"。识别完全在浏览器本地（onnxruntime-web WASM）。

```text
extension/
├── manifest.json        # <all_urls> 注入 + 右键菜单 + popup + storage 持久化
├── background.js        # 右键菜单注册 + IndexedDB 样本存储 + ZIP 导出
├── content.js           # 标记流程 + 配置驱动自动填充 + 失败样本收集
├── popup.html/js        # 样本统计 + 导出 ZIP / 清空
├── shared/recognizer.js # 共享识别核心（与演示页同一份代码）
├── icons/               # 扩展图标（16/32/48/128）
├── model.onnx           # 9.6MB 模型
└── vendor/              # onnxruntime-web WASM 运行时（零外网依赖）
```

### 安装

**方式一（源码版）**：`chrome://extensions` → 开发者模式 → 加载已解压的扩展程序 → 选择 `browser/extension`。

**方式二（发布版）**：从仓库 [Releases](https://github.com/Conastin/VerificationCode/releases) 页下载 `verification-code-extension-vX.Y.Z.zip`，解压后按方式一加载。zip 由 GitHub Actions 自动打包发布，版本号与 `manifest.json` 一致；更新内容见对应 Release 说明。

### 使用

1. 打开含验证码的登录页，在**验证码图片上右键** → "标记为验证码并自动填充"；
2. 扩展自动找到验证码输入框（按 name/id/placeholder 特征，其次按图片相邻/页面唯一文本输入框），保存配置并刷新页面；
3. 刷新后自动识别填充（右下角提示：绿=已填充 / 橙=重试中 / 红=失败，数秒后自动消失）；
4. 置信度 <0.9 自动点击验证码刷新重试（最多 5 次，60 秒后重置）；
5. 手动点击验证码刷新也会触发重新识别（MutationObserver 监听 `src`）；用户手动输入不会被覆盖。

每个站点只需标记一次，配置按 `location.origin` 持久化在 `chrome.storage.local`。

### 失败样本收集（数据闭环）

识别置信度不足（<0.9）或提交后服务器提示"验证码不正确"时，扩展自动保存该验证码图片、预测、置信度与失败原因（本地 IndexedDB，上限 2000 条，内容自动去重）。点击扩展工具栏图标 → **导出 ZIP**：`*.jpg` + `capture_manifest.csv`（与 `capture_real.py` 同格式，追加第 5 列 `reason`，现有工具不受影响）+ `failed_meta.json`（完整元数据）。

导入修正闭环：

```bash
# 1. 解压 ZIP 到 data/fail_batch，人工核对（低置信度优先）
.venv/Scripts/python -m verification_code.verify_captcha --dir data/fail_batch --max-conf 0.9

# 2. 整理并入人工集（输出 data/real_human）
.venv/Scripts/python -m verification_code.organize_real --human-dirs data/fail_batch

# 3. 微调（真实 + 合成混合防遗忘）
.venv/Scripts/python -m verification_code.finetune_real \
    --base checkpoints/best.pt --real data/real_all \
    --syn-train data/train31 --syn-val data/val31 \
    --out checkpoints/finetuned --epochs 60
```

失败判定说明：提交失败当页面出现"验证码不正确"类提示时保存——支持动态插入文本/元素、既有元素文本变化、以及整页刷新后已存在于 DOM 的静态提示三种形态（配合填充时持久化的图片快照，60 秒内且同站点才触发，固定文案与跨站快照不会误报）；保存的是提交时缓存的图片快照，站点刷新验证码也不影响。跨域图片无法读取像素时跳过保存。

浏览器端识别演示页：本地服务启动 `python browser/server.py` 后访问 `http://127.0.0.1:8000/browser/`（含 JS 与 Python 基准一致性对比）。

## 数据闭环方法（难例挖掘）

1. 采集 → 识别并记录置信度
2. 置信度 ≥0.9 直接信任（多轮独立验证 100% 准确）
3. <0.9 或含易混字符（M/N 等）人工核对
4. 合并 → 微调 → 用独立测试集回归验证

实测效果：样本库 3200 → 10200 张后，独立测试集全对 99.9%（此前 98.9%）。

## 项目结构

```text
├── verification_code/       # 模型、训练、评估、采集、微调、导出等模块
├── browser/                 # 浏览器端：MV3 扩展 + 识别演示页
├── tests/                   # pytest 测试
├── reports/                 # 评估报告、登录校验明细
├── data/                    # 数据（不入库，见 .gitignore）
├── checkpoints/             # 模型权重（不入库，见 .gitignore）
└── requirements.txt
```

## 运行测试

```bash
.venv/Scripts/python -m pytest -q
```

## 已知边界

- 模型针对目标站点字符渲染风格训练，站点改版后需重新采集验证（独立测试集可检测漂移）
- 字符集为 31 类，若目标站点加入 0/1/I/L/O 需重新生成数据并训练
- 验证码识别仅用于已授权测试环境的自动填充集成，请遵守目标系统的使用条款

## 免责声明

本项目仅用于**已授权测试环境**的安全研究与自动化填充集成，不包含登录提交或验证码绕过逻辑。请遵守目标系统的使用条款与当地法律法规，使用者需自行承担使用后果。
