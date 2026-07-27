# 使用说明：基于自定义 LLM 地址的批量提示评估

这个文档说明了如何在仓库中使用新增脚本，通过自定义 LLM 地址/Key/Model 对一组提示进行评估并将结果写入 CSV，支持断点续跑和增量保存。

文件：
- scripts/llm_client.py — 轻量级 LLM 客户端，支持 OpenAI 风格 chat/completions 和通用 prompt 接口
- scripts/evaluate_prompts.py — 主运行脚本，读取 prompts CSV、调用 LLM、将结果增量写入输出 CSV

依赖：
- python >= 3.8
- pandas
- requests
- tqdm

安装示例：

pip install pandas requests tqdm

运行示例：

python scripts/evaluate_prompts.py \
  --prompts_csv prompts.csv \
  --prompt_col prompt \
  --output_csv results.csv \
  --endpoint https://api.openai.com/v1/chat/completions \
  --api_key sk-... \
  --model gpt-4o \
  --checkpoint_interval 5

主要参数：
- --prompts_csv: 输入的提示文件（CSV）
- --prompt_col: CSV 中的提示列名，默认 `prompt`
- --output_csv: 输出结果 CSV，脚本会以追加模式写入，支持断点续跑
- --endpoint: LLM 的完整 URL（必须包含协议）
- --api_key: 可选，放在 Authorization: Bearer <key> 中
- --model: 要发送给 LLM 的模型名字（可选）
- --checkpoint_interval: 每多少条写一次（脚本每条也会追加写入以防丢失）

断点续跑：
- 如果 output_csv 已存在并包含 prompt_index 列，脚本会跳过已处理的提示

信号处理：
- 脚本安装了 SIGINT/SIGTERM 处理，会在收到中断时尽快退出并保留已写入的数据

扩展建议：
- 将 evaluate_prompts.py 中对模型返回的解析替换为项目内的评估逻辑（例如 FactSelfCheck 的打分函数），只需在获取 `text` 后调用评分函数并把分数加入输出即可
- 支持流式响应或并发批处理以提升吞吐
- 支持更多认证方式（例如 X-API-Key、自定义 header）

