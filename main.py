import json
import uuid
import os
import time
from openai import OpenAI
from tqdm import tqdm


# ========= 配置 =========
baseurl = "https://api.deepseek.com";

client = OpenAI(api_key="", base_url=baseurl)
# model_name = "deepseek-chat"
model_name = "deepseek-chat";


REQUEST_TIMEOUT = 120
CHUNK_SIZE = 2000  # 每次读取的文本块大小
DELAY = 0.5        # 请求之间的延迟（秒）


# ========= 工具函数 =========
def call_llm_api(prompt: str) -> str:
    """调用 LLM 接口并流式输出结果"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=REQUEST_TIMEOUT,
            stream=True
        )
        response_content = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
        return response_content
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return ""


def semantic_splitter(text: str) -> list:
    """将文本按语义单元进行切分"""
    prompt = f"""
你是一个专业的文本分析助手，请对以下文本进行**完整的语义切分**，将整段内容划分为多个**语义完整且独立的文本块**。

#### 📌 切分规则：
1. 每个语义块应表达一个相对独立的语义单元；
2. 每个文本块建议长度在 **500 到 1000 汉字或 token** 之间；
3. 请优先在标点符号、段落换行、逻辑停顿等自然语义边界处进行切分；
4. ⚠️ 必须确保原始文本中的每一句话都被包含在某个语义块中，**不得省略任何内容**；
5. 输出格式如下：
   - 每个语义块以 <BLOCK_START> 开头，以 <BLOCK_END> 结尾；
   - 块与块之间用换行分隔；

#### 📄 待切分文本如下：

{text}

#### 🔁 请务必记住：
- 你必须完成对整个文本的切分；
- 不得擅自简化、概括或跳过任何句子；
- 所有语义块必须忠实还原原文本内容和顺序。
"""

    llm_output = call_llm_api(prompt)
    if not llm_output:
        return []

    blocks = []
    start_tag = "<BLOCK_START>"
    end_tag = "<BLOCK_END>"
    current_pos = 0

    while True:
        start_idx = llm_output.find(start_tag, current_pos)
        if start_idx == -1:
            break
        content_start_idx = start_idx + len(start_tag)
        end_idx = llm_output.find(end_tag, content_start_idx)
        if end_idx == -1:
            print("⚠️ 警告: 找不到 <BLOCK_END>，跳过未闭合的块。")
            break
        block_content = llm_output[content_start_idx:end_idx].strip()
        if block_content:
            blocks.append({"content": block_content, "block_id": str(uuid.uuid4())})
        current_pos = end_idx + len(end_tag)

    if not blocks and llm_output.strip():
        print("⚠️ LLM 未使用分隔符，使用完整输出作为单一块。")
        blocks.append({"content": llm_output.strip(), "block_id": str(uuid.uuid4())})

    return blocks


def summarize_block(text: str) -> str:
    """为每个语义块生成摘要"""
    prompt = f"""
请为以下文本内容生成一份简洁、准确的摘要，突出其核心要点。
摘要长度应控制在 50 到 200 字之间。

{text}
"""
    return call_llm_api(prompt)


def summarize_full_document(summaries: list) -> str:
    """根据所有块的摘要生成全文摘要"""
    combined_summary = "\n\n".join(summaries)
    if len(combined_summary) > 10000:
        combined_summary = combined_summary[:10000] + "...（摘要过长已截断）"

    prompt = f"""
以下是文档各部分的摘要，请根据它们生成一份完整、连贯的全文摘要：

{combined_summary}
"""
    return call_llm_api(prompt)


# ========= 中断恢复支持 =========
def load_resume_state(resume_file):
    if os.path.exists(resume_file):
        with open(resume_file, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    return {}


def save_resume_state(resume_file, state):
    with open(resume_file, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(state, f, ensure_ascii=False, indent=4)


# ========= 主流程函数 =========
def process_single_file(file_info):
    input_path = file_info["input_path"]
    output_dir = file_info["output_dir"]

    resume_file = os.path.join(output_dir, "resume_state.json")
    resume_state = load_resume_state(resume_file)

    file_name = os.path.basename(input_path)
    output_blocks_json = os.path.join(output_dir, f"{file_name}_blocks.json")
    output_summary_json = os.path.join(output_dir, f"{file_name}_summary.json")
    output_blocks_temp = os.path.join(output_dir, f"{file_name}_blocks_temp.json")  # 临时文件

    # 加载已有块（如果存在）
    all_blocks = []
    if os.path.exists(output_blocks_temp):
        try:
            with open(output_blocks_temp, 'r', encoding='utf-8', errors='ignore') as f:
                all_blocks = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ 检测到不完整的临时文件，将从上次位置继续。")

    last_block = ""
    file_position = resume_state.get(file_name, 0)

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_size = os.fstat(f.fileno()).st_size
            pbar = tqdm(total=file_size, desc=f"📄 {file_name}", initial=file_position, unit="chars")

            while file_position < file_size:
                f.seek(file_position)
                raw_chunk = f.read(CHUNK_SIZE)
                if not raw_chunk:
                    break

                combined_text = last_block + raw_chunk
                blocks = semantic_splitter(combined_text)

                if blocks:
                    last_block = blocks[-1]["content"]
                else:
                    last_block = ""

                total_processed_length = sum(len(block["content"]) for block in blocks)

                approx_start_pos = file_position
                new_blocks = []

                for block in blocks:
                    summary = summarize_block(block["content"])
                    new_blocks.append({
                        "block_id": block["block_id"],
                        "original_text_approx_start_pos": approx_start_pos,
                        "original_text_content_snippet": block["content"][:200] + "...",
                        "summary": summary
                    })
                    approx_start_pos += len(block["content"])

                # 追加新块到总列表，并保存到临时文件
                all_blocks.extend(new_blocks)
                with open(output_blocks_temp, 'w', encoding='utf-8', errors='ignore') as f_out:
                    json.dump(all_blocks, f_out, ensure_ascii=False, indent=4)

                file_position += (total_processed_length - len(last_block))
                resume_state[file_name] = file_position
                save_resume_state(resume_file, resume_state)

                pbar.update(total_processed_length)
                time.sleep(DELAY)

            pbar.close()

        # 最终保存完整结果
        with open(output_blocks_json, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(all_blocks, f, ensure_ascii=False, indent=4)

        full_summary = summarize_full_document([b["summary"] for b in all_blocks])
        with open(output_summary_json, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump({
                "document_title": file_name,
                "total_blocks": len(all_blocks),
                "full_document_summary": full_summary,
                "generated_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=4)

        # 删除临时文件
        if os.path.exists(output_blocks_temp):
            os.remove(output_blocks_temp)

        print(f"\n✅ 完成处理文件: {file_name}")

    except Exception as e:
        print(f"❌ 处理文件 {file_name} 时发生错误: {e}")
# ========= 单线程入口 =========
def batch_process_files(input_paths, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    file_list = [{"input_path": path, "output_dir": output_dir} for path in input_paths]

    for file_info in tqdm(file_list, desc="🧠 总体进度"):
        process_single_file(file_info)


# ========= 示例入口 =========
if __name__ == "__main__":
    input_files = [
        "SemanticKB/《西游记》.txt",
        # "SemanticKB/《红楼梦》.txt",
        # "SemanticKB/《三国演义》.txt"
    ]
    batch_process_files(input_files, output_dir="SemanticKB/output")
