import json
import uuid
import os
import time

import requests # 用于模拟延迟，避免过于频繁的请求

# --- Ollama 配置 ---
OLLAMA_API_BASE_URL = "http://localhost:11434/api"
OLLAMA_MODEL_NAME = "qwen" # 请替换为您已下载并希望使用的模型名称，例如 "llama2", "qwen", "mistral" 等
REQUEST_TIMEOUT = 300 # API 请求超时时间（秒），根据模型响应速度调整

def call_ollama_api(endpoint: str, payload: dict) -> dict:
    """
    通用函数，用于调用 Ollama REST API。
    """
    url = f"{OLLAMA_API_BASE_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # 如果状态码不是 2xx，则抛出 HTTPError
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ollama API 调用失败: {e}")
        # 这里可以添加更复杂的重试逻辑
        return {"error": str(e)}

# --- 实际的 LLM 交互函数 ---
def ollama_semantic_splitter(text_content: str, last_block_context: str = "") -> list:
    """
    使用 Ollama 模型进行语义切分。
    此函数将结合上下文，并指示模型进行切分。
    """
    print(f"--- 调用 Ollama: 正在切分文本 (上下文长度: {len(last_block_context)})...")

    combined_text = last_block_context + "\n\n" + text_content if last_block_context else text_content

    # 针对 Ollama 模型的 Prompt 设计
    # 我们要求模型使用特定的分隔符来标识切分后的块
    prompt = f"""
你是一个专业的文本分析助手。请将以下文本内容按照语义完整性进行切分，生成多个独立的、语义完整的文本块。
每个切分后的文本块必须以特殊标记 `<BLOCK_START>` 开头，并以 `<BLOCK_END>` 结尾。
请确保不要在文本块内部插入这些标记。只在块的开始和结束处使用。
如果两个相邻的块之间存在很强的语义关联，请尽量将它们合并。
例如：
<BLOCK_START>第一段内容。</BLOCK_END>
<BLOCK_START>第二段内容，和第一段没有直接关联。</BLOCK_END>

以下是需要切分的文本：

{combined_text}
"""
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False # 不需要流式响应
    }

    response_data = call_ollama_api("generate", payload)

    if response_data and "response" in response_data:
        llm_output = response_data["response"].strip()
        # 解析 LLM 的响应，根据我们定义的 <BLOCK_START> 和 <BLOCK_END> 分隔符
        blocks = []
        # 将 LLM 输出中的所有换行符标准化
        llm_output_normalized = llm_output.replace('\r\n', '\n').replace('\r', '\n')
        
        # 寻找所有块
        start_tag = "<BLOCK_START>"
        end_tag = "<BLOCK_END>"
        
        current_pos = 0
        while True:
            start_idx = llm_output_normalized.find(start_tag, current_pos)
            if start_idx == -1:
                break
            
            content_start_idx = start_idx + len(start_tag)
            end_idx = llm_output_normalized.find(end_tag, content_start_idx)
            
            if end_idx == -1:
                # 块未闭合，可能是模型输出不完整，或者格式错误
                print(f"警告: 发现未闭合的块从位置 {start_idx} 开始。")
                break
            
            block_content = llm_output_normalized[content_start_idx:end_idx].strip()
            if block_content:
                blocks.append({"content": block_content, "block_id": str(uuid.uuid4())})
            
            current_pos = end_idx + len(end_tag)

        if not blocks and llm_output.strip():
            # 如果模型没有严格遵循分隔符，但仍有输出，则将整个输出作为一个块
            print("警告: LLM 未按预期生成分隔符，将整个输出作为一个块处理。")
            blocks.append({"content": llm_output.strip(), "block_id": str(uuid.uuid4())})

        print(f"--- Ollama 切分: 生成了 {len(blocks)} 个语义块。")
        return blocks
    else:
        print("Ollama 语义切分失败或返回空响应。")
        return []

def ollama_summarizer(text_content: str) -> str:
    """
    使用 Ollama 模型进行文本摘要。
    """
    print(f"--- 调用 Ollama: 正在摘要文本 (长度: {len(text_content)})...")

    prompt = f"""
请为以下文本内容生成一份简洁、准确的摘要，突出其核心要点。
摘要长度应控制在 50 到 200 字之间。

文本内容：
{text_content}
"""
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response_data = call_ollama_api("generate", payload)

    if response_data and "response" in response_data:
        summary = response_data["response"].strip()
        print(f"--- Ollama 摘要: 成功生成摘要。")
        return summary
    else:
        print("Ollama 摘要失败或返回空响应。")
        return f"无法生成摘要：{text_content[:100]}..." # 失败时返回一个简短的替代

def ollama_full_document_summarizer(all_block_summaries: list) -> str:
    """
    使用 Ollama 模型从所有块摘要生成完整文档摘要。
    """
    print(f"--- 调用 Ollama: 正在从 {len(all_block_summaries)} 个块摘要生成完整文档摘要...")

    combined_summaries_text = "\n\n".join(all_block_summaries)
    
    # 检查汇总的摘要是否过长，如果过长，可以考虑分层摘要或截断
    if len(combined_summaries_text) > 10000: # 假设模型上下文窗口有限
        print("警告: 汇总摘要过长，可能需要分层摘要策略或截断。")
        # 简单截断，实际应用中应更智能地处理
        combined_summaries_text = combined_summaries_text[:10000] + "...\n(文本过长已截断)"

    prompt = f"""
以下是文档的各个章节摘要，请根据这些摘要内容，
生成一份简洁、准确的全文摘要，概括文档的核心主题和主要内容。
全文摘要应具有连贯性，并全面反映文档的整体信息。

章节摘要：
{combined_summaries_text}
"""
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response_data = call_ollama_api("generate", payload)

    if response_data and "response" in response_data:
        full_summary = response_data["response"].strip()
        print(f"--- Ollama 全文摘要: 成功生成。")
        return full_summary
    else:
        print("Ollama 全文摘要失败或返回空响应。")
        return "无法生成全文摘要。"

# --- 主处理逻辑 ---

def process_knowledge_base_with_ollama(
    input_file_path: str,
    output_blocks_json_path: str,
    output_summary_json_path: str,
    lines_per_chunk: int = 1000,
    api_request_delay: float = 1.0 # 每次 API 调用之间的延迟，防止过载
):
    """
    使用 Ollama 模型处理知识库文件，进行语义切分、生成摘要并存储。

    Args:
        input_file_path (str): 原始知识库文本文件路径。
        output_blocks_json_path (str): 保存语义块及其摘要的 JSON 文件路径。
        output_summary_json_path (str): 保存完整文档摘要的 JSON 文件路径。
        lines_per_chunk (int): 每次迭代读取的行数（“滑窗”大小）。
        api_request_delay (float): 每次 Ollama API 调用之间的延迟（秒）。
    """
    all_processed_blocks = []
    last_processed_semantic_block_content = "" # 用于滑窗的上下文
    current_line_start_index = 0

    print(f"开始使用 Ollama 处理 '{input_file_path}'...")
    print(f"使用的 Ollama 模型: {OLLAMA_MODEL_NAME}")
    print(f"每批处理行数: {lines_per_chunk}")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            while True:
                current_chunk_lines = []
                # 读取指定行数的数据
                for _ in range(lines_per_chunk):
                    line = f.readline()
                    if not line:
                        break  # 文件结束
                    current_chunk_lines.append(line)

                if not current_chunk_lines:
                    break # 没有更多行可读

                current_chunk_text = "".join(current_chunk_lines)
                print(f"\n正在处理第 {current_line_start_index + 1} 到 {current_line_start_index + len(current_chunk_lines)} 行...")

                # 调用 Ollama 进行语义切分
                semantic_blocks_from_llm = ollama_semantic_splitter(
                    text_content=current_chunk_text,
                    last_block_context=last_processed_semantic_block_content
                )

                # 更新 'last_processed_semantic_block_content' 以供下一次迭代使用
                if semantic_blocks_from_llm:
                    # 获取 LLM 切分后的最后一个语义块内容作为下一次的上下文
                    last_processed_semantic_block_content = semantic_blocks_from_llm[-1]["content"]
                else:
                    last_processed_semantic_block_content = "" # 没有生成块，重置上下文
                
                # 为确保语义连贯性，我们可能需要调整原始行号的映射逻辑
                # 因为 LLM 可能会合并或截断上下文，这里的行号仅作粗略参考
                temp_block_start_line = current_line_start_index + 1

                # 处理每个语义块
                for block in semantic_blocks_from_llm:
                    block_content = block["content"]
                    block_id = block.get("block_id", str(uuid.uuid4())) # 使用 LLM 的 ID 或生成一个

                    # 为每个块生成摘要
                    block_summary = ollama_summarizer(block_content)

                    processed_block_info = {
                        "block_id": block_id,
                        "original_text_approx_start_line": temp_block_start_line, # 近似开始行号
                        "original_text_content_snippet": block_content[:200] + "..." if len(block_content) > 200 else block_content,
                        "summary": block_summary
                    }
                    all_processed_blocks.append(processed_block_info)
                    # 每次添加一个块，可以粗略地认为原始行号增加了，但实际切分可能不对应
                    # 真实场景中，可能需要更精细的文本位置映射
                    temp_block_start_line += block_content.count('\n') + 1 # 粗略估计

                current_line_start_index += len(current_chunk_lines)
                
                # 添加延迟以避免过快地请求 Ollama API
                time.sleep(api_request_delay)

        # 将所有处理后的块保存到 JSON 文件
        print(f"\n正在保存 {len(all_processed_blocks)} 个语义块到 '{output_blocks_json_path}'...")
        with open(output_blocks_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_processed_blocks, f, ensure_ascii=False, indent=4)

        # 生成完整文档摘要
        all_block_summaries = [block["summary"] for block in all_processed_blocks]
        full_document_summary = ollama_full_document_summarizer(all_block_summaries)

        full_summary_data = {
            "document_title": os.path.basename(input_file_path),
            "total_blocks": len(all_processed_blocks),
            "full_document_summary": full_document_summary,
            "generated_date": time.strftime("%Y-%m-%d %H:%M:%S") # 动态生成日期
        }

        # 将完整文档摘要保存到 JSON 文件
        print(f"正在保存完整文档摘要到 '{output_summary_json_path}'...")
        with open(output_summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(full_summary_data, f, ensure_ascii=False, indent=4)

        print("\n处理完成！")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file_path}' 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 示例用法 ---
if __name__ == "__main__":
    # 1. 创建一个用于测试的虚拟输入文件
    dummy_input_file = "sample_knowledge_base.txt"
    with open(dummy_input_file, 'w', encoding='utf-8') as f:
        f.write("知识库文档的第一部分。这是关于基本概念的介绍。\n")
        f.write("此部分详细阐述了核心理论及其重要性。理论A非常关键。\n")
        f.write("理论A的延伸讨论，包括其应用场景。\n\n") # 段落分隔

        f.write("接下来是文档的第二部分，聚焦于实践案例。\n")
        f.write("案例研究1：一个成功的项目实施。详细描述了挑战和解决方案。\n")
        f.write("案例研究2：另一个项目的经验教训。强调了避免的错误。\n")
        f.write("第二部分的总结，概括了实践中的主要发现。\n\n")

        f.write("第三部分是未来展望。讨论了行业趋势。\n")
        f.write("新技术的影响分析。预计这些技术将如何改变现有格局。\n")
        f.write("对未来发展方向的建议和预测。\n")
        # 添加更多行以达到多批次处理的效果
        for i in range(1, 250): # 增加行数以确保有多个分块和滑窗操作
            f.write(f"这是填充内容的通用行 {i}。有助于测试滑窗。\n")
        f.write("这是文档的最终结论。\n")
        for i in range(251, 500):
            f.write(f"进一步的补充信息，行号 {i}。填充文件。\n")
        f.write("感谢您的阅读，期待您的反馈。\n")


    output_blocks_file = "knowledge_base_ollama_blocks.json"
    output_summary_file = "knowledge_base_ollama_summary.json"

    # 执行处理
    process_knowledge_base_with_ollama(
        input_file_path=dummy_input_file,
        output_blocks_json_path=output_blocks_file,
        output_summary_json_path=output_summary_file,
        lines_per_chunk=50, # 为了快速演示，将每批处理行数设小
        api_request_delay=0.5 # 每次 Ollama API 调用之间等待 0.5 秒
    )

    # 可选：清理虚拟文件
    # os.remove(dummy_input_file)
    # os.remove(output_blocks_file)
    # os.remove(output_summary_file)