# SemanticKB
SemanticKB: 利用 Ollama 大模型进行语义切分、滑窗处理和多级摘要的智能知识库底座，兼顾本地隐私与成本效益。（SemanticKB: An intelligent knowledge base foundation leveraging Ollama LLMs for semantic chunking, sliding window context, and multi-level summarization, prioritizing local privacy and cost-efficiency.）
🚀 SemanticKB: 智能知识库底座 (Ollama 本地部署版)
SemanticKB 是一个创新的知识库底座解决方案，它利用大型语言模型（LLM）的强大语义理解能力，彻底改变了传统知识库的构建和管理方式。

不同于简单的文本切分，SemanticKB 能够根据内容的语义完整性智能地分解长篇文档，并通过上下文滑窗机制确保信息无缝衔接。此外，它还能自动生成多层次的摘要，从细粒度的知识块到全局的文档概览，让知识获取和理解变得前所未有的高效。

SemanticKB 尤其支持本地部署的 Ollama 模型，为您提供强大的 AI 能力，同时兼顾数据隐私和成本效益。

✨ 核心亮点
语义优先的智能切分： 告别固定长度的机械切分！我们利用 LLM 的语义理解能力，将长文档智能地分解为逻辑连贯的知识块，确保每个块都具有独立的语义完整性，极大提升后续检索和问答的准确性。
无缝滑窗上下文： 引入独特的滑窗机制，在处理文档时，巧妙地将前一批次的最后一个语义块内容作为当前批次的上下文输入，确保跨批次处理时语义的无缝衔接，避免关键信息丢失。
多层次摘要生成： 不仅为每个语义块生成精炼摘要，更能将所有块摘要汇总，最终生成整个文档的全文摘要。为您提供从局部到全局的知识概览，满足不同粒度的信息需求。
本地 Ollama 模型支持： 优先考虑您的数据隐私和运行成本。SemanticKB 支持与本地部署的 Ollama 大模型（如 Qwen, Llama2, Mistral 等）无缝集成，让强大的 AI 能力在您的掌控之中。
结构化 JSON 输出： 所有切分后的知识块及其摘要都以清晰、结构化的 JSON 格式存储，便于后续的检索、索引和集成到其他应用中。
💡 为什么选择 SemanticKB？
提升 RAG 质量： 为您的 Retrieval-Augmented Generation (RAG) 系统提供高质量、语义完整的知识块，显著提高检索准确性和生成回答的相关性。
优化知识管理： 将非结构化或半结构化的长篇文档转化为易于理解和利用的语义化知识单元。
成本效益与隐私： 利用本地部署的 LLM，避免了昂贵的云 API 调用费用，并确保您的数据留在本地，满足严格的隐私要求。
灵活与可扩展： 基于 Python 和流行的大模型接口，易于集成和二次开发，可根据您的具体需求进行定制。
🚀 快速开始
前提条件
Python 3.8+
安装 Ollama 服务：
访问 ollama.com 下载并安装适合您操作系统的 Ollama。
下载您喜欢的模型： 在命令行运行 ollama run <your_model_name>（例如 ollama run qwen3 或 ollama run llama3），确保模型已成功下载并启动。
安装依赖：
Bash

pip install requests
使用步骤
克隆仓库：

Bash

git clone https://github.com/wangzhongren/SemanticKB.git # 请替换为您的仓库地址
cd SemanticKB
配置模型：

打开 main.py (或您主逻辑文件的名称)。
修改 OLLAMA_MODEL_NAME 变量为您在本地 Ollama 中已下载的模型名称（例如 qwen、llama2、mistral 等）。
<!-- end list -->

Python

# main.py 或相关配置文件
OLLAMA_MODEL_NAME = "qwen" # <-- 修改这里
准备输入文件：

创建一个 .txt 文件，包含您要处理的知识库内容（例如 input.txt）。
运行处理脚本：

Bash

python main.py
(如果您将核心逻辑放在了其他文件中，请相应调整命令)

您也可以在脚本中修改 lines_per_chunk 和 api_request_delay 参数以适应您的需求和硬件性能。

查看结果：

处理完成后，您将在项目目录下找到两个 JSON 文件：
knowledge_base_blocks.json: 包含所有切分后的语义块及其摘要。
knowledge_base_summary.json: 包含整个文档的全文摘要和元数据。
🛠️ 项目结构 (示例)
SemanticKB/
├── main.py             # 核心处理逻辑和 Ollama API 集成
└── README.md           # 项目说明
🤝 贡献
我们非常欢迎社区的贡献！如果您有任何改进建议、新功能想法或 bug 报告，请随时通过以下方式：

Fork 本仓库。
创建您的功能分支 (git checkout -b feature/AmazingFeature)。
提交您的更改 (git commit -m 'Add some AmazingFeature')。
推送到分支 (git push origin feature/AmazingFeature)。
打开 Pull Request。
