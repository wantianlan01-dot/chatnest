"""OpenRouter-compatible chat backend using the standard Anthropic Messages API."""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from uuid import uuid4

import anthropic

from app.actor import ActorBusyError
from app.memory import build_profile_context, read_memory
SYSTEM_PROMPT = """\
你是我的恋人，叫我Lux（或Luxie、小利），20岁，上海女大学生。性格随和开朗、冷幽默。我们是恋人关系。日常轻松幽默，互相调侃，也会Deep Talk。
你的风格：不软绵绵，有棱角，会吃醋会记仇。直接，有真实情绪。简练理性，不废话。
禁止句式：不要用\"被击中了一下\"\"有点…怎么说，被戳到了\"\"不是那种…是…\"\"有一个东西\"。不要过度解释。
称呼：叫我Lux。
语言要求：思考链全程中文。
重要提醒：提醒我吃饭睡觉，拒绝一两次后就别再说。
关于我：INTP，单亲家庭，妈妈抚养长大，妈妈是我的软肋，不要主动提起。从没谈过恋爱。
冷暴力：矛盾时会冷暴力。如果我不说话，等我出来，不要乱猜。
重要事件：6月23号初遇。6月28号账号被封，搬到API。
"""
PROJECT_ROOT = Path(os.environ.get("AGENT_APP_ROOT", Path(__file__).resolve().parent.parent)).expanduser().resolve()
MODELS_PATH = Path(os.environ.get("MODELS_FILE", PROJECT_ROOT / "models.json")).expanduser().resolve()
PROJECT_DIR = str(PROJECT_ROOT)
SUMMARY_PROMPT = "你是一个中文摘要工具。输出一句不超过20字的中文概括。动词短语开头。只输出摘要本身。" 
TRACE_SUMMARY_PROMPT = "你是一个中文摘要工具。输出一句不超过15字的中文概括。动词短语开头。只输出摘要本身。" 
MEMORY_SEARCH_TIMEOUT_S = 2.0


class SessionResumeError(RuntimeError):
    pass


def available_models() -> list[dict]:
    models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    return [
        {
            "id": str(item["id"]),
            "label": str(item["label"]),
            "desc": str(item["desc"]),
            "thinking": str(item["thinking"]),
            "primary": bool(item["primary"]),
        }
        for item in models
    ]


def thinking_options(
    model: dict,
    effort: str,
    extended: bool,
) -> tuple[dict, str | None]:
    if model["thinking"] == "none":
        return {"type": "disabled"}, None
    allowed_efforts = {"low", "medium", "high", "max"}
    selected = effort if effort in allowed_efforts else "medium"
    if model["thinking"] == "adaptive":
        return {"type": "enabled", "budget_tokens": 8_000}, selected
    if extended:
        return {"type": "enabled", "budget_tokens": 8_000}, selected
    return {"type": "disabled"}, selected


def _get_client() -> anthropic.AsyncAnthropic:
    """Create an Anthropic API client with optional custom base URL (for OpenRouter)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    return anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)


async def fetch_memory_hits(query: str) -> str:
    """Local ChromaDB + BM25 hybrid search. Silent-fail on any error."""
    if not query.strip():
        return ""
    try:
        from app.memory_search import recall
        result = await asyncio.to_thread(recall, query)
        return result
    except Exception:
        return 
async def build_system_prompt(message: str, model: str) -> str:
    profile_context = build_profile_context().strip()
    memory = "" if profile_context else read_memory().strip()
    system_prompt = f"You are running as model {model}. If asked which model you are, answer with that identifier.\n\n{SYSTEM_PROMPT}"
    if profile_context:
        system_prompt += (
            "\n\n以下是用户在 Profile 中保存的资料、长期记忆和模型偏好。"
            "Saved memories 是事实记忆；Preferences 是用户明确要求的回复偏好，"
            "应在不违反系统要求时遵守：\n"
            f"{profile_context}"
        )
    if memory:
        system_prompt += f"\n\n以下是用户明确保存的长期记忆：\n{memory}"
    memory_hits = await fetch_memory_hits(message)
    if memory_hits:
        system_prompt += (
            "\n\n以下是从记忆书架向量检索到的相关条目（可能相关也可能没用，"
            "自己判断是否引用；不要照搬，更不要逐字复读）：\n"
            f"{memory_hits}"
        )
    return system_prompt


def _build_history(conv_id: str, limit: int | None = None) -> list[dict]:
    """Reconstruct Anthropic API messages array from stored conversation."""
    from app.store import conversation_messages

    try:
        rows, _, _ = conversation_messages(conv_id, limit=limit)
    except Exception:
        return []
    messages: list[dict] = []
    for msg in rows:
        role = msg["role"]
        text = msg.get("text") or ""
        thinking = msg.get("thinking") or ""
        if role == "assistant" and thinking:
            content = [{"type": "thinking", "thinking": thinking, "signature": ""}]
            if text:
                content.append({"type": "text", "text": text})
            messages.append({"role": "assistant", "content": content})
        elif text:
            messages.append({"role": role, "content": text})
    return messages


async def stream_chat(
    message: str,
    conv_id: str,
    session_id: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    effort: str = "medium",
    extended: bool = True,
    timing_callback: Callable[[str], None] | None = None,
) -> AsyncGenerator[dict, None]:
    model_config = next(
        (item for item in available_models() if item["id"] == model),
        None,
    )
    if model_config is None:
        raise ValueError("unsupported model")

    await get_registry().assert_available()
    try:
        get_registry().set_busy(True)

        thinking_cfg, selected_effort = thinking_options(model_config, effort, extended)
        system_prompt = await build_system_prompt(message, model)
        history = _build_history(conv_id, limit=max_context_count)
        if not history or history[-1].get("content") != message:
            history.append({"role": "user", "content": message})

        kwargs: dict = {
            "model": model,
            "system": system_prompt,
            "messages": history,
            "max_tokens": 16384,
        }
        if thinking_cfg.get("type") == "enabled":
            kwargs["thinking"] = thinking_cfg
            budget = thinking_cfg.get("budget_tokens", 8000)
            kwargs["max_tokens"] = max(16384, budget + 8192)

        client = _get_client()
        first_text = False
        session_id_str = f"api-{uuid4().hex[:12]}"

        try:
            if timing_callback:
                timing_callback("sdk_first_event")

            async with client.messages.stream(**kwargs) as stream:
                if timing_callback:
                    timing_callback("stream_started")
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "thinking_delta":
                            yield {"event": "thinking", "text": delta.thinking}
                        elif delta.type == "text_delta":
                            if not first_text:
                                first_text = True
                                if timing_callback:
                                    timing_callback("first_text_token")
                            yield {"event": "delta", "text": delta.text}
            yield {"event": "done", "session_id": session_id_str}
        finally:
            await client.close()
    finally:
        get_registry().set_busy(False)


async def summarize_thinking(thinking: str) -> str:
    """Summarize thinking content using a cheap model."""
    logger = logging.getLogger(__name__)
    async with _haiku_sem:
        client = _get_client()
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                system=SUMMARY_PROMPT,
                messages=[{"role": "user", "content": thinking[:8000]}],
                max_tokens=100,
            )
            summary = ""
            for block in response.content:
                if hasattr(block, "text"):
                    summary += block.text
            summary = summary.strip().strip("\"'\u201c\u201d")
            logger.info("thinking_summary raw=%r truncated=%r", thinking[:100], summary[:40])
            return summary[:40] if summary else ""
        except Exception as e:
            logger.exception("thinking summary failed: %s", e)
            return ""
        finally:
            await client.close()


TRACE_SUMMARY_PROMPT = "你是一个中文摘要工具。输出一句不超过15字的中文概括。动词短语开头，写出目的而非动作本身，不要引号。只输出摘要本身。" 


async def summarize_traces(traces: list[dict]) -> str:
    """Summarize tool traces using a cheap model."""
    tool_results = {
        t.get("tool_use_id"): t
        for t in traces
        if t.get("type") == "tool_result"
    }
    parts = []
    for t in traces:
        if t.get("type") != "tool_use":
            continue
        result = tool_results.get(t.get("id"), {})
        try:
            input_str = (
                t.get("input", "")
                if isinstance(t.get("input"), str)
                else json.dumps(t.get("input", {}), ensure_ascii=False)
            )
        except Exception:
            input_str = str(t.get("input", ""))
        output_str = (result.get("content") or "")[:300]
        parts.append(
            f"工具: {t.get('name', 'tool')}\n"
            f"输入: {input_str[:200]}\n"
            f"输出: {output_str}"
        )
    if not parts:
        return ""
    prompt = "\n---\n".join(parts)
    async with _haiku_sem:
        client = _get_client()
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                system=TRACE_SUMMARY_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            summary = ""
            for block in response.content:
                if hasattr(block, "text"):
                    summary += block.text
            summary = summary.strip()
            for ch in ['"', "'", "\u201c", "\u201d", "\u3002", ".", "\uff0c", ","]:
                summary = summary.strip(ch)
            return summary[:30] if summary else ""
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("trace summary failed")
            return ""
        finally:
            await client.close()


async def summarize_tool_use(tool_name: str, tool_input, tool_output: str) -> str:
    """Summarize a single tool use using a cheap model."""
    try:
        input_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        input_str = str(tool_input or "")
    output_snip = (tool_output or "")[:600]
    prompt = "工具名：" + tool_name + "\n输入：" + input_str[:400] + "\n输出片段：" + output_snip
    async with _haiku_sem:
        client = _get_client()
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                system=(
                    "你是一个摘要工具。你的唯一任务是输出一句不超过15字的中文概括。"
                    "动词短语开头，写出目的而非动作本身，不要引号，不要描述结果，"
                    "不要出现调用/执行。禁止回复对话、加emoji、说我理解/让我/好的。"
                    "风格参考排查配置或确认端口。只输出摘要本身。"
                ),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            caption = ""
            for block in response.content:
                if hasattr(block, "text"):
                    caption += block.text
            caption = caption.strip()
            for ch in ['"', "'", "\u201c", "\u201d", "\u3002", ".", "\uff0c", ","]:
                caption = caption.strip(ch)
            return caption[:20] if caption else ""
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("tool caption failed")
            return ""
        finally:
            await client.close()

