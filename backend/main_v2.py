"""FastAPI 主入口 - AI 圆桌会议系统"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models.schemas import (
    Meeting, MeetingState, Message, SpeakerType,
    HostRole, GuestRole,
    CreateMeetingRequest, AgendaFeedbackRequest,
    UserIntervention, StartMeetingRequest
)
from core.meeting_engine import MeetingEngine
from core.db import upsert_record, get_record, list_records, delete_record
from utils.file_parser import extract_text

app = FastAPI(title="AI 圆桌会议系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 内存存储（后续可换 Redis / DB）──
meetings: Dict[str, Meeting] = {}
engines: Dict[str, MeetingEngine] = {}
model_configs_store: Dict[str, Dict[str, dict]] = {}  # meeting_id -> {agent_id -> config}


def cleanup_meeting(meeting_id: str):
    """释放会议占用的内存资源（结束后调用）"""
    meetings.pop(meeting_id, None)
    engines.pop(meeting_id, None)
    model_configs_store.pop(meeting_id, None)


# ═══════════════════════════════════════
# 健康检查
# ═══════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════
# 服务配置测试 API
# ═══════════════════════════════════════


@app.post("/services/test-embedding")
async def test_embedding_connection(req: dict):
    """测试 Embedding API 连接是否可用"""
    import httpx
    try:
        platform = req.get("platform", "")
        api_key = req.get("apiKey", "").strip()
        base_url = req.get("baseUrl", "").rstrip("/")
        model = req.get("model", "")

        if not api_key:
            return {"ok": False, "error": "缺少 API Key"}

        # 根据平台确定默认值
        if not base_url and platform == "dashscope":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode"
        elif not base_url and platform == "openai":
            base_url = "https://api.openai.com"
        elif not base_url and platform == "hunyuan":
            base_url = "https://api.hunyuan.cloud.tencent.com"
        if not model:
            if platform == "dashscope":
                model = "text-embedding-v4"
            elif platform == "openai":
                model = "text-embedding-3-small"
            elif platform == "hunyuan":
                model = "hunyuan-embedding"

        url = f"{base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "input": "hello test",
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, headers=headers, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            dim = len(data["data"][0]["embedding"])
            return {"ok": True, "msg": "连接成功", "dimension": dim}
        else:
            err = resp.text[:150]
            return {"ok": False, "error": f"HTTP {resp.status_code}: {err}"}

    except httpx.TimeoutException:
        return {"ok": False, "error": "连接超时，请检查网络和地址"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════
# 会议管理 REST API
# ═══════════════════════════════════════

@app.post("/meetings")
async def create_meeting(req: CreateMeetingRequest):
    """创建会议，传入议题、嘉宾配置、模型配置"""
    meeting_id = str(uuid.uuid4())

    host = HostRole(
        id="host",
        name=req.host.get("name", "主持人"),
        system_prompt=req.host.get("system_prompt", "你是专业会议主持人。"),
        model=req.host.get("model", ""),
        color=req.host.get("color", "#f5a623")
    )

    guests = [
        GuestRole(
            id=g.get("id", str(uuid.uuid4())),
            name=g["name"],
            system_prompt=g.get("system_prompt", ""),
            model=g.get("model", ""),
            color=g.get("color", "#4f8ef7")
        )
        for g in req.guests
    ]

    meeting = Meeting(
        id=meeting_id,
        topic=req.topic,
        materials=req.materials or [],
        host=host,
        guests=guests,
        state=MeetingState.PREPARING,
        tavily_key=req.tavily_key,
        embedding_config=req.embedding_config,
        host_style=req.host_style or 'neutral',
    )
    # 存储讨论标题（用于历史卡片显示）
    meeting.discussion_title = req.discussion_title
    meetings[meeting_id] = meeting

    # 存模型配置
    model_configs: Dict[str, dict] = {}
    host_cfg = req.host.get("model_config", {})
    model_configs["host"] = host_cfg
    for g in req.guests:
        model_configs[g.get("id", "")] = g.get("model_config", host_cfg)
    model_configs_store[meeting_id] = model_configs

    # 立即持久化到 DB（state=preparing，先占坑）
    upsert_record(meeting_id, _meeting_to_record(meeting))

    return {"meeting_id": meeting_id, "state": meeting.state}


@app.post("/meetings/{meeting_id}/materials")
async def upload_material(meeting_id: str, file: UploadFile = File(...)):
    """上传背景材料文件，提取文本后存入会议"""
    m = _get_meeting(meeting_id)
    content = await file.read()
    try:
        text = extract_text(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空，无法提取文本")

    m.materials.append({
        "filename": file.filename,
        "text": text[:8000]
    })
    # 上传后立即持久化 materials
    rec = get_record(meeting_id)
    if rec:
        rec["materials"] = [{"name": item["filename"], "content": item["text"]} for item in m.materials]
        upsert_record(meeting_id, rec)
    return {
        "filename": file.filename,
        "chars": len(text),
        "preview": text[:200]
    }


@app.delete("/meetings/{meeting_id}/materials/{idx}")
async def delete_material(meeting_id: str, idx: int):
    """删除已上传的材料"""
    m = _get_meeting(meeting_id)
    if idx < 0 or idx >= len(m.materials):
        raise HTTPException(status_code=404, detail="材料不存在")
    removed = m.materials.pop(idx)
    rec = get_record(meeting_id)
    if rec:
        rec["materials"] = [{"name": item["filename"], "content": item["text"]} for item in m.materials]
        upsert_record(meeting_id, rec)
    return {"removed": removed["filename"]}


@app.post("/meetings/{meeting_id}/materials/inject")
async def inject_material(meeting_id: str, payload: dict):
    """直接注入已提取的材料内容（历史复用）"""
    m = _get_meeting(meeting_id)
    m.materials.append({
        "filename": payload.get("name", "历史材料"),
        "text": payload.get("content", "")[:8000]
    })
    rec = get_record(meeting_id)
    if rec:
        rec["materials"] = [{"name": item["filename"], "content": item["text"]} for item in m.materials]
        upsert_record(meeting_id, rec)
    return {"injected": payload.get("name")}


@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    m = _get_meeting(meeting_id)
    return {
        "id": m.id,
        "topic": m.topic,
        "state": m.state,
        "agenda": m.agenda.dict() if m.agenda else None,
        "current_section_index": m.current_section_index,
        "guests": [{"id": g.id, "name": g.name, "color": g.color} for g in m.guests],
        "report": m.report,
    }


@app.get("/meetings")
async def list_meetings():
    return [
        {"id": m.id, "topic": m.topic, "state": m.state,
         "created_at": str(m.created_at)}
        for m in meetings.values()
    ]


# ═══════════════════════════════════════
# SSE 流式接口
# ═══════════════════════════════════════

def _sse_format(event: dict) -> str:
    def _default(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False, default=_default)}\n\n"


@app.get("/meetings/{meeting_id}/agenda/stream")
async def stream_agenda(meeting_id: str):
    """SSE：生成议程"""
    engine = _get_or_create_engine(meeting_id)
    m = meetings[meeting_id]

    async def generate():
        async for ev in engine.generate_agenda():
            yield _sse_format(ev)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/meetings/{meeting_id}/agenda/feedback")
async def agenda_feedback(meeting_id: str, req: AgendaFeedbackRequest):
    """SSE：用户反馈后重新生成议程"""
    engine = _get_or_create_engine(meeting_id)

    async def generate():
        try:
            async for ev in engine.regenerate_agenda(req.user_feedback or ""):
                yield _sse_format(ev)
        except Exception as e:
            logger.error(f"[agenda/feedback] 重新生成议程异常: {e}", exc_info=True)
            yield _sse_format({"event": "error", "data": {"msg": f"议程生成失败: {e}"}})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/meetings/{meeting_id}/host/chat")
async def host_chat(meeting_id: str, req: AgendaFeedbackRequest):
    """SSE：与主持人自由对话（可回答问题，必要时重新生成议程）"""
    engine = _get_or_create_engine(meeting_id)

    async def generate():
        try:
            async for ev in engine.host_chat(req.user_feedback or ""):
                yield _sse_format(ev)
        except Exception as e:
            logger.error(f"[host/chat] 主持人对话异常: {e}", exc_info=True)
            yield _sse_format({"event": "error", "data": {"msg": f"对话异常: {e}"}})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/meetings/{meeting_id}/prepare/stream")
async def stream_prepare(meeting_id: str):
    """SSE：嘉宾准备论点"""
    engine = _get_or_create_engine(meeting_id)

    async def generate():
        try:
            async for ev in engine.prepare_guests():
                yield _sse_format(ev)
        except Exception as e:
            logger.error(f"[prepare/stream] 嘉宾准备异常: {e}", exc_info=True)
            yield _sse_format({"event": "error", "data": {"msg": f"准备失败: {e}"}})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/meetings/{meeting_id}/discuss/stream")
async def stream_discuss(meeting_id: str):
    """SSE：开始正式讨论"""
    engine = _get_or_create_engine(meeting_id)
    m = meetings[meeting_id]

    async def generate():
        try:
            async for ev in engine.run_discussion():
                yield _sse_format(ev)
                # 进入观众互动或结束时，持久化一次
                if ev.get("event") == "state_change":
                    state = ev["data"].get("state", "")
                    if state in ("audience_qa", "ended"):
                        upsert_record(meeting_id, _meeting_to_record(m))
        except Exception as e:
            logger.error(f"[discuss/stream] 讨论流程异常中断: {e}", exc_info=True)
            yield _sse_format({"event": "error", "data": {"msg": f"讨论异常: {e}"}})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/meetings/{meeting_id}/audience/ask")
async def audience_ask(meeting_id: str, req: AgendaFeedbackRequest):
    """SSE：观众互动提问"""
    engine = _get_or_create_engine(meeting_id)

    async def generate():
        try:
            async for ev in engine.audience_ask(req.user_feedback or ""):
                yield _sse_format(ev)
        except Exception as e:
            logger.error(f"[audience/ask] 错误: {e}", exc_info=True)
            yield _sse_format({"event": "error", "data": {"msg": f"提问失败: {e}"}})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/meetings/{meeting_id}/summarize")
async def summarize_meeting(meeting_id: str):
    """SSE：用户主动触发总结"""
    engine = _get_or_create_engine(meeting_id)
    m = meetings[meeting_id]

    async def generate():
        from models.schemas import MeetingState
        m.state = MeetingState.SUMMARY
        yield _sse_format({"event": "state_change", "data": {"state": "summary"}})
        async for ev in engine._host_summary():
            yield _sse_format(ev)
        m.state = MeetingState.ENDED
        yield _sse_format({"event": "state_change", "data": {"state": "ended"}})
        # 总结完成后持久化（含 summary 内容）
        upsert_record(meeting_id, _meeting_to_record(m))

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})




# ═══════════════════════════════════════
# 历史记录 CRUD
# ═══════════════════════════════════════

@app.get("/history")
def list_history():
    """获取历史会谈列表（摘要，不含完整消息）"""
    records = list_records(50)
    result = []
    for r in records:
        history = r.get("history") or []
        # 最后一条非用户发言的内容摘要
        last_msg = next(
            (m for m in reversed(history) if m.get("speaker_type") != "user"),
            None
        )
        preview = ""
        if last_msg:
            name = last_msg.get("speaker_name", "")
            content = last_msg.get("content", "")[:40]
            preview = f"{name}：{content}"
        # 参与者颜色（主持人+嘉宾）
        hc = r["host_config"] or {}
        gc = r["guests_config"] or []
        participants = [
            {"name": hc.get("name", "主持"), "color": hc.get("color", "#f5a623"), "is_host": True}
        ] + [
            {"name": g["name"], "color": g.get("color", "#4f8ef7"), "is_host": False}
            for g in gc
        ]
        result.append({
            "id": r["id"],
            "topic": r["topic"],
            "discussion_title": r.get("discussion_title"),
            "state": r["state"],
            "host_name": hc.get("name", "主持人"),
            "guest_names": [g["name"] for g in gc],
            "participants": participants,
            "msg_count": len(history),
            "preview": preview,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "has_summary": bool(r.get("summary")),
        })
    return result


@app.get("/history/{meeting_id}")
def get_history(meeting_id: str):
    """获取单条历史记录完整内容（含配置+消息+总结）"""
    r = get_record(meeting_id)
    if not r:
        raise HTTPException(404, "记录不存在")
    return r


@app.delete("/history/{meeting_id}")
def del_history(meeting_id: str):
    # 1. 删除前先获取 embedding 配置（用于同步清理 ChromaDB）
    record = get_record(meeting_id)
    if record:
        emb_cfg = record.get("embedding_config")
        try:
            from utils.vector_memory import VectorMemory
            vm = VectorMemory(embedding_config=emb_cfg or {})
            # 同步删除 ChromaDB 中的向量数据
            vm.delete_meeting(meeting_id)
        except Exception as e:
            logger.warning(f"向量清理失败(非致命): {e}")

    # 2. 删除 SQLite 记录
    delete_record(meeting_id)

    # 3. 如果会议还在内存中，也清除
    cleanup_meeting(meeting_id)

    return {"ok": True}


@app.post("/meetings/{meeting_id}/save")
def save_meeting_now(meeting_id: str):
    """强制持久化当前内存中的会谈（用户手动离开时调用）"""
    m = meetings.get(meeting_id)
    if m:
        upsert_record(meeting_id, _meeting_to_record(m))
    return {"ok": True}


@app.get("/meetings/{meeting_id}/resume/stream")
async def resume_meeting(meeting_id: str):
    """SSE：从DB恢复会谈，从上次最后一条完整发言后接着跑"""
    # 1. 从DB读出历史记录
    record = get_record(meeting_id)
    if not record:
        raise HTTPException(404, "历史记录不存在")

    # 2. 用共用函数重建 Meeting（含所有数据）
    # 不强制 target_state=SECTION，而是根据DB中实际状态决定恢复策略
    # 之前bug：一刀切设为SECTION，导致preparing/agenda_confirming阶段暂停后直接跳过所有流程
    meeting = _rebuild_meeting_from_record(record, meeting_id)

    # 3. 提取 host/guests 引用（供后续 prompt 使用）
    host = meeting.host
    guests = list(meeting.guests)

    # 4. 注册到内存 + 模型配置
    meetings[meeting_id] = meeting
    model_configs = {}
    hc_cfg = (record.get("host_config") or {}).get("model_config", {})
    model_configs["host"] = hc_cfg
    for g in (record.get("guests_config") or []):
        model_configs[g.get("id", "")] = g.get("model_config", hc_cfg)
    model_configs_store[meeting_id] = model_configs

    def _auto_save():
        try:
            upsert_record(meeting_id, _meeting_to_record(meeting))
        except Exception as e:
            logger.warning(f"[auto_save] 持久化失败(非致命): {e}")
    engine = MeetingEngine(meeting, model_configs, on_message_saved=_auto_save)
    engines[meeting_id] = engine

    # 6b. 构建议程文本供 prompt 使用
    def _fmt_agenda(ag):
        if not ag:
            return "（无议程）"
        lines = [f"议题：{ag.topic}"]
        for s in ag.sections:
            lines.append(f"  · {s.title}：{s.description}（{s.duration_minutes}分钟）")
        return "\n".join(lines)

    def _fmt_history(hist):
        if not hist:
            return "（暂无记录）"
        return "\n".join(f"【{m.speaker_name}】{m.content}" for m in hist)

    # 7. 根据DB中的实际状态，分派不同恢复策略（不再一刀切）
    async def generate():
        try:
            from utils.prompts import HOST_RESUME_PROMPT
            yield _sse_format({"event": "resumed", "data": {
                "msg": "会谈已恢复，从上次暂停处继续",
                "history_count": len(meeting.history),
                "db_state": meeting.state.value if hasattr(meeting.state, 'value') else str(meeting.state),
            }})
            
            # ── 状态分支 A：还在准备阶段（议程未生成或未确认） ──
            if meeting.state in (MeetingState.PREPARING, MeetingState.AGENDA_CONFIRMING):
                logger.info(f"[resume] 恢复到准备阶段: {meeting.state}")
                
                if meeting.state == MeetingState.PREPARING or not meeting.agenda:
                    # 议程还没生成 → 重新生成议程
                    meeting.state = MeetingState.PREPARING
                    yield _sse_format(engine._state_event(MeetingState.PREPARING))
                    async for ev in engine.generate_agenda():
                        yield _sse_format(ev)
                    return  # 等用户确认议程
                
                elif meeting.state == MeetingState.AGENDA_CONFIRMING and meeting.agenda:
                    # 议程已生成但用户没确认 → 重新推送议程让用户确认
                    yield _sse_format(engine._state_event(MeetingState.AGENDA_CONFIRMING))
                    yield _sse_format({"event": "agenda", "data": meeting.agenda.dict()})
                    return  # 等用户确认

            # ── 状态分支 B：暖场阶段中断 ──
            has_section_msgs = any(
                m.section_idx is not None and m.section_idx >= 0
                for m in meeting.history
            )
            if not has_section_msgs and meeting.history:
                logger.info("[resume] 暖场阶段中断，补跑暖场")
                # 暖场接场发言
                from utils.prompts import HOST_WARMUP_PROMPT
                warmup_resume_prompt = (
                    f"好，信号恢复了。我们刚才聊到了「{meeting.topic}」这个话题，"
                    f"现在正式开始。{host.name}，请开始你的开场介绍。"
                )
                async for ev in engine._stream_speaker(
                    host.id, host.name, SpeakerType.HOST, host.color, warmup_resume_prompt
                ):
                    yield _sse_format(ev)
                
                meeting.state = MeetingState.WARMUP
                yield _sse_format(engine._state_event(MeetingState.WARMUP))
                async for ev in engine._host_warmup():
                    yield _sse_format(ev)
                # 暖场完后走嘉宾准备流程
                meeting.state = MeetingState.PREPARING_GUESTS
                yield _sse_format(engine._state_event(MeetingState.PREPARING_GUESTS))
                async for ev in engine.prepare_guests():
                    yield _sse_format(ev)
                return  # 嘉宾准备完进入讨论

            # ── 状态分支 C：板块讨论中中断 → 用原有板块恢复逻辑 ──
            # 主持人发表"复电接场"发言
            resume_prompt = HOST_RESUME_PROMPT.format(
                host_name=host.name,
                topic=meeting.topic,
                history=_fmt_history(meeting.history),
                agenda=_fmt_agenda(meeting.agenda)
            )
            async for ev in engine._stream_speaker(
                host.id, host.name, SpeakerType.HOST, host.color, resume_prompt
            ):
                yield _sse_format(ev)

            # ── 精确判断每个板块的恢复状态 ──
            section_states = {}  # {index: state}

            if meeting.agenda and guests:
                n_guests = len(guests)
                for i, _ in enumerate(meeting.agenda.sections):
                    section_msgs = [m for m in meeting.history if m.section_idx == i]
                    guest_spoken = set(m.speaker_id for m in section_msgs if m.speaker_type == SpeakerType.GUEST)
                    host_spoken_in_section = any(m.speaker_type == SpeakerType.HOST for m in section_msgs)

                    all_guests_done = len(guest_spoken) >= n_guests
                    host_msg_count = sum(1 for m in section_msgs if m.speaker_type == SpeakerType.HOST)
                    total_content_len = sum(len(m.content) for m in section_msgs)

                    has_wrapup = (
                        all_guests_done and host_msg_count >= 2 and total_content_len > 200
                    )

                    if has_wrapup:
                        section_states[i] = "COMPLETED"
                    elif all_guests_done:
                        section_states[i] = "NEED_WRAPUP"
                    elif len(guest_spoken) > 0:
                        section_states[i] = "IN_GUESTS"
                    elif host_spoken_in_section:
                        section_states[i] = "IN_INTRO"
                    else:
                        section_states[i] = "NOT_STARTED"

                    logger.info(
                        f"[resume] 板块[{i}] '{meeting.agenda.sections[i].title}' → "
                        f"{section_states[i]} | guests={len(guest_spoken)}/{n_guests} "
                        f"host_msgs={host_msg_count} content_len={total_content_len}"
                    )

            # 输出被跳过的已完成板块
            if meeting.agenda:
                skipped = [i for i, s in section_states.items() if s == "COMPLETED"]
                if skipped:
                    skip_names = [meeting.agenda.sections[i].title for i in skipped]
                    logger.info(f"[resume] 跳过已完成板块({len(skipped)}个): {skip_names}")
                    yield _sse_format({"event": "sections_skipped", "data": {
                        "skipped_indices": skipped,
                        "skipped_titles": skip_names,
                        "reason": "这些板块在上次会谈中已完整结束"
                    }})

            # 跑所有未完成板块
            if meeting.agenda:
                first_resume_section = True
                for i, section in enumerate(meeting.agenda.sections):
                    state = section_states.get(i, "NOT_STARTED")
                    if state == "COMPLETED":
                        continue

                    meeting.current_section_index = i
                    meeting.state = MeetingState.SECTION
                    yield _sse_format({"event": "section_start", "data": {"index": i, "section": section.dict()}})
                    yield _sse_format(engine._state_event(MeetingState.SECTION))

                    async for ev in engine._run_section(
                        section,
                        skip_spoken_guests=True,
                        is_resuming=first_resume_section,
                        resume_state=state,
                    ):
                        yield _sse_format(ev)

                    first_resume_section = False
                    while engine._paused:
                        await asyncio.sleep(0.5)

            # 自由讨论补跑
            all_sections_done = all(
                s == "COMPLETED" for s in section_states.values()
            ) if section_states else False
            has_free_talk = any(
                m.section_idx is None or m.section_idx < 0
                for m in meeting.history if m.speaker_type == SpeakerType.GUEST
            )
            if all_sections_done and not has_free_talk:
                logger.info("[resume] 所有板块完成，补跑自由讨论")
                meeting.state = MeetingState.FREE_TALK
                yield _sse_format(engine._state_event(MeetingState.FREE_TALK))
                async for ev in engine._free_talk():
                    yield _sse_format(ev)

            # 进入观众互动
            meeting.state = MeetingState.AUDIENCE_QA
            yield _sse_format(engine._state_event(MeetingState.AUDIENCE_QA))
            yield _sse_format({"event": "audience_qa_start", "data": {
                "msg": "议程已完成，进入观众互动环节。"
            }})
            upsert_record(meeting_id, _meeting_to_record(meeting))

        except Exception as e:
            logger.error(f"[resume/stream] 续跑流程异常中断: {e}", exc_info=True)
            yield _sse_format({"event": "error", "data": {"msg": f"续跑异常: {e}"}})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════
# WebSocket 用户干预
# ═══════════════════════════════════════

@app.websocket("/meetings/{meeting_id}/ws")
async def websocket_intervention(websocket: WebSocket, meeting_id: str):
    """
    用户通过 WebSocket 发送干预指令：
    {"action": "pause"} | {"action": "resume"} |
    {"action": "skip_section"} |
    {"action": "call_on_guest", "target": "guest_id"} |
    {"action": "custom_instruction", "instruction": "..."}
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            engine = engines.get(meeting_id)
            if engine:
                engine.apply_intervention(
                    action=cmd.get("action"),
                    target=cmd.get("target"),
                    instruction=cmd.get("instruction")
                )
                await websocket.send_json({"status": "ok", "action": cmd.get("action")})
            else:
                await websocket.send_json({"status": "error", "msg": "会议不存在或未开始"})
    except WebSocketDisconnect:
        pass


# ═══════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════

def _meeting_to_record(m: Meeting) -> dict:
    """把 Meeting 对象序列化为 DB 存储格式"""
    return {
        "topic": m.topic,
        "state": m.state.value if hasattr(m.state, "value") else m.state,
        "host_style": getattr(m, "host_style", "neutral"),
        "embedding_config": getattr(m, "embedding_config", None),
        # 续跑需要：tavily key、报告
        "tavily_key": getattr(m, 'tavily_key', ''),
        # 话题拆分：标题（短，用于历史卡片显示）
        "discussion_title": getattr(m, 'discussion_title', None),
        # 话题拆分：内容（详细描述）
        "topic_content": getattr(m, 'topic_content', None),
        "summary": m.report,
        "materials": [
            {"filename": item.get("filename", ""), "content": item.get("text", "")}
            for item in (m.materials or [])
        ],
        "host_config": {
            "id": m.host.id, "name": m.host.name,
            "system_prompt": m.host.system_prompt,
            "model": m.host.model, "color": m.host.color,
            "model_config": model_configs_store.get(m.id, {}).get("host", {})
        },
        "guests_config": [
            {
                "id": g.id, "name": g.name,
                "system_prompt": g.system_prompt,
                "model": g.model, "color": g.color,
                "model_config": model_configs_store.get(m.id, {}).get(g.id, {}),
                # 续跑需要：嘉宾准备好的论点
                "prepared_arguments": g.prepared_arguments or {},
            }
            for g in m.guests
        ],
        "history": [
            {
                "speaker_id": msg.speaker_id,
                "speaker_name": msg.speaker_name,
                "speaker_type": msg.speaker_type.value if hasattr(msg.speaker_type, "value") else msg.speaker_type,
                "content": msg.content,
                "color": msg.color,
                "section_idx": msg.section_idx,
                "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, "isoformat") else str(msg.timestamp)
            }
            for msg in m.history
        ],
        "summary": m.report,
        "agenda": {
            "topic": m.agenda.topic,
            "total_duration": m.agenda.total_duration,
            "sections": [
                {
                    "title": s.title,
                    "description": s.description,
                    "duration_minutes": s.duration_minutes,
                    "order": s.order
                }
                for s in m.agenda.sections
            ]
        } if m.agenda else None,
        "created_at": m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at),
    }


def _get_meeting(meeting_id: str) -> Meeting:
    m = meetings.get(meeting_id)
    if not m:
        raise HTTPException(404, "会议不存在")
    return m


def _rebuild_meeting_from_record(record: dict, meeting_id: str,
                                   target_state: MeetingState = None) -> Meeting:
    """从 DB record 重建 Meeting 对象的共用函数。
    
    resume_meeting 和 _restore_meeting_from_db 共用此逻辑，
    避免维护两份几乎相同的代码。
    
    Args:
        record: 从 get_record() 获取的完整记录
        meeting_id: 会议ID
        target_state: 目标状态，None 时从 record 中推断
    Returns:
        重建好的 Meeting 实例（已含 history / agenda / materials / prepared_arguments）
    """
    # 1. 重建主持人 + 嘉宾角色对象
    hc = record.get("host_config") or {}
    host = HostRole(
        id=hc.get("id", "host"),
        name=hc.get("name", "主持人"),
        system_prompt=hc.get("system_prompt", "你是专业会议主持人。"),
        model=hc.get("model", ""),
        color=hc.get("color", "#f5a623")
    )
    gc_list = record.get("guests_config") or []
    guests = [
        GuestRole(
            id=g.get("id", str(uuid.uuid4())),
            name=g["name"],
            system_prompt=g.get("system_prompt", ""),
            model=g.get("model", ""),
            color=g.get("color", "#4f8ef7")
        )
        for g in gc_list
    ]

    # 2. 确定状态
    if target_state is None:
        state_str = record.get("state", "preparing")
        try:
            target_state = MeetingState(state_str)
        except ValueError:
            target_state = MeetingState.PREPARING

    # 3. 创建 Meeting 对象
    meeting = Meeting(
        id=meeting_id,
        topic=record["topic"],
        host=host,
        guests=guests,
        state=target_state,
        tavily_key=record.get("tavily_key", ""),
        embedding_config=record.get("embedding_config"),
        host_style=record.get("host_style", "neutral"),
    )

    # 4. 恢复材料
    for mat in (record.get("materials") or []):
        meeting.materials.append({
            "filename": mat.get("filename", ""),
            "text": mat.get("content", ""),
        })

    # 5. 恢复历史消息
    from models.schemas import Message, SpeakerType
    for h in (record.get("history") or []):
        try:
            st = SpeakerType(h.get("speaker_type", "guest"))
        except Exception:
            st = SpeakerType.GUEST
        meeting.history.append(Message(
            speaker_id=h.get("speaker_id", ""),
            speaker_name=h.get("speaker_name", ""),
            speaker_type=st,
            content=h.get("content", ""),
            color=h.get("color", "#888"),
            section_idx=h.get("section_idx", None)
        ))

    # 6. 恢复议程
    from models.schemas import Agenda, AgendaSection
    agenda_data = record.get("agenda")
    if agenda_data:
        sections = [
            AgendaSection(
                title=s["title"],
                description=s.get("description", ""),
                duration_minutes=s.get("duration_minutes", 10),
                order=s.get("order", i + 1)
            )
            for i, s in enumerate(agenda_data.get("sections", []))
        ]
        meeting.agenda = Agenda(
            topic=agenda_data.get("topic", record["topic"]),
            total_duration=agenda_data.get("total_duration", 60),
            sections=sections
        )

    # 7. 恢复嘉宾准备论点
    for i, g in enumerate(guests):
        if i < len(gc_list) and gc_list[i].get("prepared_arguments"):
            g.prepared_arguments = gc_list[i]["prepared_arguments"]

    # 8. 恢复报告
    if record.get("summary"):
        meeting.report = record["summary"]

    return meeting


def _register_model_configs(meeting_id: str, record: dict):
    """从 record 恢复模型配置并注册到内存"""
    hc_cfg = (record.get("host_config") or {}).get("model_config", {})
    model_configs: Dict[str, dict] = {"host": hc_cfg}
    for g in (record.get("guests_config") or []):
        model_configs[g.get("id", "")] = g.get("model_config", hc_cfg)
    model_configs_store[meeting_id] = model_configs


def _restore_meeting_from_db(meeting_id: str) -> bool:
    """尝试从DB历史记录恢复会议到内存。成功返回True，失败返回False"""
    record = get_record(meeting_id)
    if not record:
        return False

    # 用共用函数重建 Meeting（含 history / agenda / materials / arguments / report）
    meeting = _rebuild_meeting_from_record(record, meeting_id)

    # 注册到内存
    meetings[meeting_id] = meeting
    _register_model_configs(meeting_id, record)
    return True


def _get_or_create_engine(meeting_id: str) -> MeetingEngine:
    if meeting_id not in engines:
        # 先检查内存，没有则尝试从DB恢复
        if meeting_id not in meetings:
            if not _restore_meeting_from_db(meeting_id):
                raise HTTPException(404, f"会议 {meeting_id} 不存在（内存和数据库中均未找到）")
                # 如果内存有但engine没有（异常情况），直接用内存的meeting创建engine

        m = meetings[meeting_id]
        cfg = model_configs_store.get(meeting_id, {})
        def _auto_save():
            try:
                upsert_record(meeting_id, _meeting_to_record(m))
            except Exception as e:
                logger.warning(f"[auto_save] 持久化失败(非致命): {e}")
        engines[meeting_id] = MeetingEngine(m, cfg, on_message_saved=_auto_save)
    return engines[meeting_id]
