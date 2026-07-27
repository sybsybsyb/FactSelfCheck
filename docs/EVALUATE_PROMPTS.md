# 使用说明：基于自定义 LLM 地址的批量提示评估（.env 配置）

已更新：现在脚本会优先读取项目根目录或当前工作目录下的 `.env` 文件（如果安装了 `python-dotenv`），并从环境变量中加载配置。你可以仍然通过命令行参数覆盖这些设置，但默认无需传参即可运行。

新增依赖（可选但推荐）：
- python-dotenv — 用于从 .env 文件加载环境变量

安装：

pip install pandas requests tqdm python-dotenv

示例 .env（放在仓库根目录或运行目录）：

PROMPTS_CSV=prompts.csv
PROMPT_COL=prompt
OUTPUT_CSV=results.csv
DELIMITER=,
ENDPOINT=https://api.openai.com/v1/chat/completions
API_KEY=sk-...
MODEL=gpt-4o
CHECKPOINT_INTERVAL=5
TEMPERATURE=0.0
MAX_TOKENS=
MAX_RETRIES=3
BACKOFF=1.0
SYSTEM_PROMPT=
START_INDEX=

说明：
- 脚本会按以下优先级读取配置：命令行参数 > 环境变量（.env-loaded） > 内置默认值
- 如果没有在任何位置配置 `ENDPOINT`，脚本会退出并显示错误；请把你的 LLM endpoint 写入 .env 的 ENDPOINT 项。
- 对于 API key，脚本支持从下列环境变量读取（优先级按顺序）： API_KEY, LLM_API_KEY, OPENAI_API_KEY

运行：

python scripts/evaluate_prompts.py

或者（覆盖 .env 中的配置）：

python scripts/evaluate_prompts.py --prompts_csv other_prompts.csv --endpoint https://...

其它说明与之前相同：脚本逐行追加输出 CSV、支持断点续跑、捕获 SIGINT/SIGTERM 以保留已写入数据、支持将 FactSelfCheck 的评分函数集成到 response 后进行评估并写入额外列。
