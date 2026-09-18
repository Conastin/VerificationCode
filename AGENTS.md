# AGENTS.md

验证码识别器项目（60×20 四字符验证码模型 + 通用 CRNN 模型 + 工具链 + MV3 浏览器扩展）的工作约定与踩坑记录。

## 发布流程（Release Extension workflow）

- 触发：推送 `v*` 标签（如 `git tag v2.6.0 && git push origin v2.6.0`），或 Actions 页手动 `workflow_dispatch`
- 版本号取自 `browser/extension/manifest.json` 的 `version` 字段（不是 tag 名）；打 tag 前先确认 manifest 版本已更新
- 产物：`verification-code-extension-v<version>.zip` + 分组 release notes（按 Conventional Commits 前缀：feat/fix/perf/refactor/test/docs/ci-chore/其他）
- 扩展版本更新后记得同步 README 徽章（version-<x.y.z>）与使用说明

## CI 踩坑记录（已修复，勿再犯）

### 1. YAML `run: |` 块标量缩进

- 块标量内所有行必须保持与首行相同的基缩进（当前文件 10 空格），**顶格行会提前结束块标量**，GitHub 直接判定 workflow 无效（0 秒 failure run，日志 "This run likely failed because of a workflow file issue"）
- 多行字符串（如 `NOTES="..."`）内容行、闭合引号行都要带基缩进；赋值后可用 `sed 's/^ \{10\}//'` 去除
- 修改 workflow 后先本地验证：PyYAML `yaml.safe_load` 解析 + 提取 run 块模拟执行（注意 Windows 工作区 CRLF 需先替换，`$GITHUB_OUTPUT` 本地需手动设置）

### 2. actions/checkout 浅克隆

- `actions/checkout` 默认 `fetch-depth: 1`，**只含触发 commit，没有历史 tag**；release notes 里 `git log <prev_tag>..HEAD` 会报 `unknown revision`，必须显式 `fetch-depth: 0`
- `gh release list` 走 API 能拿到旧 tag，但 git 对象不存在——两者不能混用判断

### 3. notes 内联命令行时的反引号命令替换

- `gh release create --notes "${{ steps.notes.outputs.notes }}"` 中，`${{ }}` 内联后 bash 会重新解析该行，**notes 内容里的反引号会触发命令替换**（内容被当命令执行吞掉）
- 安装说明/URL 等不要用反引号包裹（`\`` 转义也没用），用纯文本；release notes 是纯文本场景，Markdown 代码格式不需要

### 4. 本地 Windows TLS/网络（push 相关）

- 本机 git HTTPS 用 schannel 后端，遇证书吊销检查离线（`CRYPT_E_REVOCATION_OFFLINE` / `TLS connect error`）时用一次性参数绕开：
  - `git -c http.sslBackend=openssl push`
  - 或 `git -c http.schannelCheckRevoke=false push`
  - 大包推送再加 `-c http.postBuffer=524288000`
- 网络抖动会间歇性失败（git/gh/curl 同时 EOF），重试脚本化：循环 + sleep，最多 6 次
- 不要在本地改全局 git config 绕 TLS（除非用户要求）

## 约定

- 提交信息：`type(scope): 中文描述`，type ∈ feat/fix/docs/ci/chore（release notes 按此前缀分组，写规范些）
- 测试：`.venv/Scripts/python -m pytest -q`（Python 侧）；扩展改动用 playwright + 系统 Edge（channel="msedge"）端到端验证，headless 加载扩展：`--disable-extensions-except=<dir> --load-extension=<dir>`，扩展 id 从 service worker URL 解析
- 内网站点（如 oas.example.com 自签名证书）：playwright context 需 `ignore_https_errors=True`；curl 需 `--ssl-no-revoke`
- 数据分发：`checkpoints/best.pt`、`data/real_test_independent`、`data/real_human` 随仓库分发；真实采集样本（`real_all`/`train31`/`raw` 等）本地保留（.gitignore 白名单规则，父目录忽略后无法再包含子项，用 `data/*` + `!` 否定）
- 发布类操作（commit/push/tag/release 创建）需用户明确要求后执行

## portal.example.com 验证码（2026-09 探明）

- 接口：`GET https://portal.example.com/image/getRandcode/<lt>_KEY`；`lt` 是登录页 HTML 隐藏域 `input[name=lt]` 的 CAS ticket，每次打开 `/login` 都会新发；只依赖 JSESSIONID cookie
- "输账号后点密码才显示验证码"是前端行为（`POST /service/api/v1/authType/checkRandomCodeRequired` 控制显隐），图片接口随时可 GET；同 KEY 加 `?d=随机数` 刷新每次返回新图 → 纯 HTTP 批量采集可行（`tmp/portal/capture.py`，样本在 `data/portal_raw/`，已被 gitignore）
- 样式：80×34 JPEG、4 字符、**不排除易混字符**（0/1/I/L/O 都出现且 0 与 O 并存）、混合大小写显示但**登录校验大小写不敏感**（实测全大写提交 47 次通过证实）；旧 slot 模型直接用识别率极低
- 登录反馈验证（`portal_login_validate.py`）：POST `/login`（表单字段 lt/execution/_eventId/csrfToken/randomCode 等，账密明文、乱填即可）；判定信号是响应里服务器注入的 `<div id="msg">` 元素——**验证码错**显示"验证码输入错误 请您重新输入!"，**验证码对**（乱账号）显示"用户不存在"。注意：页面固有 JS 模板里的"人脸认证失败"等字样不是判定信号；响应编码随请求头协商可能是 GBK，解码要 utf-8/gbk 双试
- 自动标注闭环（`portal_autolabel.py`）：采集→模型预测→提交→服务器判定，"通过"的样本即**服务器确认的金标准**（比人工标注更可靠）归档到 `data/portal_dataset/server_confirmed/`，"拒绝"的进 `server_failed/`（难例，供人工标注）。数据集是珍贵资产：**append-only、带 manifest、SHA-256 去重、永不覆盖**；`data/portal_label/test`（60 张）是永久留出测试集，任何训练不得使用
- **重放可行性（2026-09-18 实测）**：抓到的验证码与其会话（JSESSIONID+lt/execution/csrf）绑定**至少存活 10 分钟**——延迟 0/60/600 秒后提交，错码仍判"验证码输入错误"、对码仍判"用户不存在"。因此"低置信先不提交、人工即时标注、再提交验证"的工作流成立（每张图只有一次提交机会，票据一次性）
- **实时人在环标注**（`live_label_server.py`，http://localhost:8765）：后台线程持续采集，conf≥0.97 自动提交收金标准；低置信的**不提交**、挂入待标队列（TTL 480s 内有效），浏览器页面即时人工输入→服务器判定→"人工书写+服务器确认"双保险金标准（`server_confirmed/live_*/`，文件名 H 前缀）；人工被拒样本也归档（标注质检）。实测首小时：自动 678 金 + 人工 152 确认 + 5 张人工标注被服务器纠正

## VLM 标注验证码的坑（重要，勿再踩）

- analyze_image（GLM-4.5v）转录此类彩色涂鸦验证码**不可靠**，失败模式三种：
  1. **并行调用共享上下文**：同批多图调用会互相复读（后调用逐字复读前面批次的输出）
  2. **按 URL 缓存响应**：同一 URL 换 prompt 仍返回旧结果；换内容哈希（改渲染参数）才绕得过
  3. **系统性幻觉**：连续输出"字母数字严格交替"等规整串（真实分布无此规律）；且语气自信，无法从输出本身辨别
- 可靠的 portal 真值只能来自：人工标注、或测试账号 + 登录接口反馈（`login_validate` 模式）；旧站当年就是 VLM 初标 + 人工核对（`verify_captcha.py`）
- 判定技巧：跨图出现重复"标签"而图像 MD5 各不相同 → 必是幻觉；输出全按规整模式排列 → 高度可疑

## 通用 CRNN 模型管线（2026-09 新增）

- 模块：`synth_universal.py`（域随机合成：40 字体/随机尺寸 20-40/变长 3-6/彩字/涂鸦线/形变/JPEG 伪影）→ `train_universal.py`（CRNN+CTC，变宽输入、36 字符集大小写折叠、`IMG_H` 归一）→ `finetune_universal.py`（真实+合成混合微调）→ `analyze_portal.py`/`old_model_portal.py`/`compare_portal_sheet.py`（对比分析）
- 实测：纯合成基线旧站只有 ~24-47%；混入旧站真实数据微调后一个模型同时做到旧站 93.7%+ 且 portal 预测可信（置信分布/已验证样本 5/5）。专用 slot 模型旧站仍 99.9%，通用模型是"一套权重服务所有站"的取舍
- **Windows DataLoader spawn 陷阱**：`num_workers>0` 时 worker 重新 import 模块，**主进程运行期改的模块级全局（如 IMG_H）不会传过去**；配置必须存进 Dataset 对象（可 pickle）、collate 里的高度从数组 shape 取，不要读全局
- **训练吞吐排查结论**（RTX 5060 Laptop）：小模型瓶颈常在 CPU 渲染与每步同步，不在 GPU——warp 用 numpy 向量化、解码/打印每 200 步一次（`.tolist()` 会强制同步打断流水线）；实测最优配置 **batch 128 + workers 6**（94ms/step、1358 imgs/s、RAM~5.2GB、显存仅 0.55GB），加大 batch 到 192/256 反而按图劣化（LSTM/CTC 随步内规模变慢），加 workers 只涨内存不涨速度。内存大头是渲染 worker 进程，与显存无关，"把数据放显存"没有收益
- CTC 训练特征：前几千步 loss 卡在 ~ln(类别数)、精度 0 是正常"对齐锁定期"，突破后快速上升；合成域过宽（比真实还难）会拖慢收敛，域随机要覆盖真实站风格而非穷举极端
- **数据飞轮实测**（portal，2026-09-18）：v1(240 人工) 78.3% → v6(1686 金标准) 93.3% → v8(3554 同质金标准) **平台期 90~93%** → v10(+live 第一场 1988：低置信挂起+人工即时标注+服务器双确认) **96.7%/98.3%** → v11(+live 第二场 1251，道闸下人工被拒率 0.5%) 95.0%×2 → v12(真难例 373 张×3 加权对照实验) 95.0%×2。旧站 95.4%→98.0% 区间。稳态带 95-97%。
- **瓶颈归因（四组对照实验定论）**：①容量对照（LSTM 256×3）无差异→不是模型；②真难例加权对照（v12）无差异 + 记忆诊断（训练内难例仅 84%，连记忆都不完全）；③slot 切格探针否决（字符中心抖动 mean 5.5px/p90 9px，格宽 20px，固定切格不可行）；④**饱和度第 4 通道（sat4）实测也无差异**（portal 94.2%≈v11 95.0%，旧站反降至 93.9%）——尽管"彩字 vs 灰线"在 HSV 饱和度轴上确实可分（1.1:1 像素比），显式喂给模型也不涨分 → **瓶颈是像素级信息上限（贝叶斯误差）**：低置信池 ~70-85% 模型其实是对的，真错部分的判别特征被涂鸦遮挡，新旧难例间无可迁移结构。结论：portal 单次 95-97% 即物理上限附近；突破只能靠提交侧策略（低置信刷新/双模型仲裁/重试）。旧站数据堆量仍可涨（专用模型证明上限 99.9%）
