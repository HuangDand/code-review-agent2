
import os
import logging
from pathlib import Path
from datetime import datetime
from time import monotonic

import dashscope
from dotenv import load_dotenv


# =========================================================
# 1. 基础配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
REPORT_PATH = BASE_DIR / "workflow_report.md"
LOG_PATH = BASE_DIR / "workflow.log"

# 明确从 main.py 所在目录读取 .env
load_dotenv(dotenv_path=ENV_PATH, override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

# 保留目前已经跑通的 API 地址和模型
dashscope.base_http_api_url = "https://maas.qianwenaiapi.com/api/v1"
MODEL_NAME = "qwen3.8-max"

# 日志同时写入文件和终端
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("Homework2Workflow")


# =========================================================
# 2. API 调用与返回结果处理
# =========================================================

def extract_response_text(response):
    """检查 API 响应，并提取模型生成的文本。"""

    status_code = getattr(response, "status_code", None)

    if status_code != 200:
        error_code = getattr(response, "code", "未知")
        error_message = getattr(response, "message", str(response))

        raise RuntimeError(
            f"千问 API 调用失败：HTTP {status_code}；"
            f"错误码：{error_code}；"
            f"错误信息：{error_message}"
        )

    output = getattr(response, "output", None)
    choices = getattr(output, "choices", None) if output else None

    if not choices:
        raise RuntimeError("API 请求成功，但没有返回有效的模型结果。")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message else None

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts).strip()
    else:
        text = ""

    if not text:
        raise RuntimeError("模型返回内容为空，请检查模型响应格式。")

    return text


def call_agent(agent_name, prompt, state):
    """
    调用一个 Agent。

    state 是共享状态字典：
    - 保存各 Agent 的输出
    - 保存消息传递记录
    - 保存每个步骤的执行状态
    """

    started = monotonic()
    logger.info("%s 开始执行（提示词约 %s 字符）", agent_name, len(prompt))
    print(f"\n[{agent_name}] 正在请求模型，提示词约 {len(prompt)} 个字符，请稍候……", flush=True)
    state["agent_status"][agent_name] = "running"

    # 记录 Agent 收到的任务，便于追踪消息传递
    state["messages"].append({
        "from": "Workflow",
        "to": agent_name,
        "content": prompt
    })

    try:
        response = dashscope.MultiModalConversation.call(
            api_key=API_KEY,
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ]
        )

        result = extract_response_text(response)

        # 把结果写入共享状态
        state["results"][agent_name] = result
        state["messages"].append({
            "from": agent_name,
            "to": "Workflow",
            "content": result
        })
        state["agent_status"][agent_name] = "completed"

        elapsed = monotonic() - started
        state.setdefault("agent_elapsed_seconds", {})[agent_name] = round(elapsed, 1)
        logger.info("%s 执行完成，耗时 %.1f 秒，输出约 %s 字符", agent_name, elapsed, len(result))
        print(f"\n[{agent_name}] 完成，耗时 {elapsed:.1f} 秒，输出约 {len(result)} 个字符。", flush=True)
        print(f"\n{'=' * 18} {agent_name} {'=' * 18}")
        print(result)

        return result

    except Exception as exc:
        state["agent_status"][agent_name] = "failed"
        state["errors"].append({
            "agent": agent_name,
            "error": str(exc)
        })

        elapsed = monotonic() - started
        state.setdefault("agent_elapsed_seconds", {})[agent_name] = round(elapsed, 1)
        logger.exception("%s 执行失败，等待 %.1f 秒", agent_name, elapsed)
        raise


# =========================================================
# 3. Agent 1：需求与代码分析
# =========================================================

def requirement_analysis_agent(task, source_code, state):
    """分析任务需求；如果用户提供了代码，也分析代码问题。"""

    if source_code.strip():
        input_description = (
            "用户提供了现有代码。请重点分析代码功能、潜在错误、"
            "不符合需求之处和需要修复的地方。"
        )
        code_section = (
            "\n\n【用户提供的现有代码】\n"
            + source_code
        )
    else:
        input_description = (
            "用户没有提供现有代码。请分析需求，并明确需要实现的功能、"
            "输入输出、约束条件和建议的验证方法。"
        )
        code_section = "\n\n用户未提供现有代码。"

    prompt = f"""分析任务，列出目标、问题和验收标准。
任务：{task}
{input_description}
{code_section}
最多列出3条要点，总结不超过100字；不要声称运行过代码。
"""

    result = call_agent("Agent 1 - 需求与代码分析", prompt, state)
    state["requirements"] = result
    return result


# =========================================================
# 4. Agent 2：方案设计与代码修复
# =========================================================

def design_and_fix_agent(task, source_code, state):
    """读取 Agent 1 的分析，并生成设计方案与代码修改建议。"""

    requirements = state.get("requirements")
    if not requirements:
        raise ValueError("缺少需求分析结果，无法执行 Agent 2。")

    if source_code.strip():
        code_section = "\n\n【原始代码】\n" + source_code
    else:
        code_section = "\n\n用户没有提供原始代码。请给出最小实现方案。"

    # 只传递 Agent 1 的核心摘要，避免上下文越来越长
    requirements_short = requirements[:2500]
    prompt = f"""按需求修复代码，只给关键问题、代码和测试。
任务：{task}
需求：{requirements_short}
{code_section}
只给一个代码版本，说明不超过100字；不要声称测试已通过。
"""

    result = call_agent("Agent 2 - 方案设计与代码修复", prompt, state)
    state["design"] = result
    return result


# =========================================================
# 5. Agent 3：测试、审查与报告
# =========================================================

def testing_review_agent(task, source_code, state):
    """检查前两个 Agent 的输出，形成测试计划与最终审查。"""

    requirements = state.get("requirements")
    design = state.get("design")

    if not requirements or not design:
        raise ValueError("缺少前序 Agent 的结果，无法执行 Agent 3。")

    if source_code.strip():
        code_section = (
            "\n\n【用户最初提供的代码】\n"
            + source_code
        )
    else:
        code_section = "\n\n用户没有提供原始代码。"

    requirements_short = requirements[:1800]
    design_short = design[:4500]
    prompt = f"""审查方案，给出测试用例和交付结论。
任务：{task}
需求：{requirements_short}
方案：{design_short}
{code_section}
列出最多3个问题和4条测试；未运行时注明测试仅为建议。
"""

    result = call_agent("Agent 3 - 测试与审查", prompt, state)
    state["review"] = result
    return result


# =========================================================
# 6. 工作流控制：按顺序执行三个 Agent
# =========================================================

def run_workflow(task, source_code=""):
    """按顺序运行工作流，并通过共享状态传递结果。"""

    state = {
        "task": task,
        "source_code": source_code,
        "requirements": None,
        "design": None,
        "review": None,
        "results": {},
        "messages": [],
        "agent_status": {},
        "agent_elapsed_seconds": {},
        "errors": [],
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print("\n工作流启动")
    print("执行顺序：需求分析 -> 方案设计与修复 -> 测试与审查")
    logger.info("工作流启动")

    try:
        # 条件分支：根据用户是否提供代码选择分析重点
        if source_code.strip():
            print("检测到用户提供的代码，将进入代码分析与修复模式。")
            logger.info("当前分支：代码分析与修复")
        else:
            print("未提供现有代码，将进入需求分析与方案设计模式。")
            logger.info("当前分支：需求分析与方案设计")

        # 顺序执行：后一个 Agent 使用前一个 Agent 的输出
        requirement_analysis_agent(task, source_code, state)
        design_and_fix_agent(task, source_code, state)
        testing_review_agent(task, source_code, state)

        state["workflow_status"] = "completed"
        logger.info("所有 Agent 执行完成")

    except Exception as exc:
        state["workflow_status"] = "failed"
        logger.error("工作流中止：%s", exc)
        print("\n工作流中止：", exc)
        print("请查看 workflow.log 获取详细错误信息。")

    state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_report(state)
    print_workflow_summary(state)

    return state


# =========================================================
# 7. 写入 Markdown 报告
# =========================================================

def save_report(state):
    """将共享状态和 Agent 输出写入 Markdown 文件。"""

    sections = [
        "# Simple Workflow MVP 运行报告",
        "",
        f"- 开始时间：{state.get('started_at', '')}",
        f"- 结束时间：{state.get('finished_at', '')}",
        f"- 工作流状态：{state.get('workflow_status', 'unknown')}",
        f"- 使用模型：{MODEL_NAME}",
        "",
        "## 一、用户任务",
        "",
        state.get("task", ""),
        "",
        "## 二、Agent 1：需求与代码分析",
        "",
        state.get("requirements") or "未完成",
        "",
        "## 三、Agent 2：方案设计与代码修复",
        "",
        state.get("design") or "未完成",
        "",
        "## 四、Agent 3：测试、审查与报告",
        "",
        state.get("review") or "未完成",
        "",
        "## 五、执行状态",
        ""
    ]

    for agent_name, status in state.get("agent_status", {}).items():
        elapsed = state.get("agent_elapsed_seconds", {}).get(agent_name)
        suffix = f"，耗时 {elapsed} 秒" if elapsed is not None else ""
        sections.append(f"- {agent_name}：{status}{suffix}")

    sections.extend([
        "",
        "## 六、消息传递记录",
        "",
        f"本次工作流记录了 {len(state.get('messages', []))} 条消息。",
        ""
    ])

    if state.get("errors"):
        sections.extend(["## 七、错误记录", ""])
        for error in state["errors"]:
            sections.append(f"- **{error['agent']}**：{error['error']}")
        sections.append("")

    sections.extend([
        "---",
        "本报告由 Python 工作流程序自动生成。",
        "模型生成的测试计划不代表代码已经实际执行或通过测试。",
        ""
    ])

    try:
        REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
        logger.info("报告已写入：%s", REPORT_PATH)
        print(f"\n报告已写入：{REPORT_PATH}")
        print(f"运行日志：{LOG_PATH}")
    except OSError as exc:
        logger.exception("写入报告失败")
        print("报告写入失败：", exc)


# =========================================================
# 8. 工作流状态汇总
# =========================================================

def print_workflow_summary(state):
    print("\n" + "=" * 48)
    print("工作流执行总结")
    print("=" * 48)
    print("任务：", state.get("task", ""))
    print("状态：", state.get("workflow_status", "unknown"))

    for agent_name, status in state.get("agent_status", {}).items():
        elapsed = state.get("agent_elapsed_seconds", {}).get(agent_name)
        if elapsed is not None:
            print(f"{agent_name}: {status}（耗时 {elapsed} 秒）")
        else:
            print(f"{agent_name}: {status}")

    print("消息记录数：", len(state.get("messages", [])))
    print("错误数量：", len(state.get("errors", [])))

    if state.get("workflow_status") == "completed":
        print("三个 Agent 已按顺序完成。")
    else:
        print("工作流未全部完成，请查看错误信息和日志。")


# =========================================================
# 9. 程序入口
# =========================================================

def main():
    print("=" * 56)
    print("Homework 2 - 多 Agent 代码开发与修复工作流")
    print("=" * 56)

    if not API_KEY:
        print("\n错误：没有读取到 DASHSCOPE_API_KEY。")
        print(f"请确认 .env 文件位于：{ENV_PATH}")
        print("文件内容应为：DASHSCOPE_API_KEY=你的真实API密钥")
        return

    print("API Key 已读取。")
    print("注意：程序不会显示或写入你的 API Key。")

    task = input(
        "\n请输入你的开发任务或需要解决的问题：\n> "
    ).strip()

    if not task:
        print("任务不能为空，程序结束。")
        return

    print(
        "\n如果需要分析或修复现有代码，请粘贴代码。"
        "\n粘贴结束后，单独输入 END 并回车。"
        "\n如果没有现有代码，直接输入 END 并回车即可。"
    )

    code_lines = []

    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\n已结束代码输入。")
            break

        if line.strip() == "END":
            break

        code_lines.append(line)

    source_code = "\n".join(code_lines).strip()

    print("\n任务已接收，开始运行工作流……")

    run_workflow(task, source_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户手动终止程序。")
        logger.warning("用户手动终止程序")
    except Exception as exc:
        logger.exception("程序发生未处理异常")
        print("\n程序发生异常：", exc)
        print("请查看 workflow.log 获取详细信息。")
