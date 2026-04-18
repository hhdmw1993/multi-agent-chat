"""Tavily 联网搜索工具"""
from typing import Optional


async def search_topic(topic: str, tavily_key: str, max_results: int = 5) -> str:
    """
    根据话题搜索最新资讯，返回格式化文本供注入 prompt。
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        results = client.search(
            query=topic,
            search_depth="basic",
            max_results=max_results,
            include_answer=True
        )

        lines = ["以下是关于该议题的最新网络资讯（搜索时间：今日）：\n"]

        # 如果有综合答案
        if results.get("answer"):
            lines.append(f"【综合摘要】{results['answer']}\n")

        # 逐条结果
        for i, r in enumerate(results.get("results", []), 1):
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            lines.append(f"[{i}] {title}\n{content}\n来源：{url}\n")

        return "\n".join(lines)

    except Exception as e:
        return f"（联网搜索失败：{e}，将基于已有知识讨论）"
