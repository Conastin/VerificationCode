# 用户人工标注数据（金标准来源）

- 采集时间: 2026-09-17 22:57-22:59（lt=LT-14350 批次）
- 标注人: 用户本人，通过 tmp/portal/make_label_tool.py 生成的标注工具
- 样本数: 300 张，labels.txt 为原始标注（保留大小写原样）
- 用途: 训练（data/portal_label/train 为 240 子集）/ 其中 60 张为留出测试集（data/portal_label/test）
- 注意: 标签保留原始大小写；登录校验大小写不敏感，训练时统一折叠大写
