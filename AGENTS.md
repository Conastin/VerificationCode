# AGENTS.md

验证码识别器项目（60×20 四字符验证码模型 + 工具链 + MV3 浏览器扩展）的工作约定与踩坑记录。

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
- 内网站点（如 oas.example.com.com 自签名证书）：playwright context 需 `ignore_https_errors=True`；curl 需 `--ssl-no-revoke`
- 数据分发：`checkpoints/best.pt`、`data/real_test_independent`、`data/real_human` 随仓库分发；真实采集样本（`real_all`/`train31`/`raw` 等）本地保留（.gitignore 白名单规则，父目录忽略后无法再包含子项，用 `data/*` + `!` 否定）
- 发布类操作（commit/push/tag/release 创建）需用户明确要求后执行
