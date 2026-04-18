"""模型适配器 - 统一封装多平台 LLM 调用，支持流式/非流式"""
import asyncio
import json
import time
import httpx
from typing import AsyncGenerator, List, Dict


# ── 超时配置（秒）──
FIRST_TOKEN_TIMEOUT = 30   # 首个token必须在30秒内到达
INTER_TOKEN_TIMEOUT = 90   # 连续两个token间隔不超过90秒
CONNECT_TIMEOUT = 15       # 连接超时
TOTAL_TIMEOUT = 300        # 总时长上限（5分钟）


async def stream_chat(
    model_config: dict,
    messages: List[Dict],
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """
    统一流式调用入口（含首token/间token超时保护）
    
    超时策略：
    - 首token：30秒内必须有输出，否则中断并抛异常
    - 间token：连续两个token间隔不超过90秒
    - 总时长：不超过5分钟
    
    model_config: {model, baseUrl, apiKey, platform}
    """
    base_url = model_config.get("baseUrl", "").rstrip("/")
    api_key = model_config.get("apiKey", "").strip()
    model = model_config.get("model", "")

    if not base_url:
        base_url = "https://api.openai.com/v1"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}

    async with httpx.AsyncClient(timeout=httpx.Timeout(
        connect=CONNECT_TIMEOUT, read=TOTAL_TIMEOUT,
        write=TOTAL_TIMEOUT, pool=TOTAL_TIMEOUT
    )) as client:
        async with client.stream(
            "POST", f"{base_url}/chat/completions",
            headers=headers, json=payload
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise Exception(f"API错误 [{resp.status_code}]: {body.decode()[:300]}")
            
            got_first_token = False
            last_token_time = time.monotonic()
            
            async for line in resp.aiter_lines():
                # 首token超时检查
                if not got_first_token and (time.monotonic() - last_token_time > FIRST_TOKEN_TIMEOUT):
                    raise TimeoutError(f"首token超时({FIRST_TOKEN_TIMEOUT}s)：模型{model}无响应，请检查API服务是否正常")
                
                # 间token超时检查（已有数据后）
                if got_first_token and (time.monotonic() - last_token_time > INTER_TOKEN_TIMEOUT):
                    raise TimeoutError(f"间token超时({INTER_TOKEN_TIMEOUT}s)：模型响应中断")
                
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            got_first_token = True
                            last_token_time = time.monotonic()
                            yield delta
                    except Exception:
                        continue
            
            # 流结束后仍未拿到首token
            if not got_first_token:
                raise TimeoutError(f"首token超时({FIRST_TOKEN_TIMEOUT}s)：流结束但无任何有效数据")


async def complete_chat(
    model_config: dict,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """非流式调用，返回完整文本"""
    result = []
    async for chunk in stream_chat(model_config, messages, temperature):
        result.append(chunk)
    return "".join(result)
