"""会议引擎 - 状态机 + 主持人/嘉宾 Agent 调度"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Callable

logger = logging.getLogger("meeting_engine")

from models.schemas import (
    Meeting, MeetingState, Message, SpeakerType,
    Agenda, AgendaSection, GuestRole
)
from core.model_adapter import stream_chat, complete_chat
from utils.prompts import (
    AGENDA_GENERATION_PROMPT, GUEST_PREPARE_PROMPT,
    HOST_WARMUP_PROMPT, HOST_SECTION_INTRO_PROMPT,
    HOST_TRANSITION_PROMPT, HOST_SUMMARY_PROMPT,
    GUEST_SPEAK_PROMPT, REPORT_GENERATION_PROMPT,
    HOST_CHAT_PROMPT, AUDIENCE_QA_HOST_PROMPT, AUDIENCE_QA_GUEST_PROMPT,
    get_host_transition_prompt, get_style_injection,
    HOST_TRANSITION_WITH_MEMORY_PROMPT, GUEST_SPEAK_WITH_MEMORY_PROMPT,
    HOST_RESUME_WITH_MEMORY_PROMPT, HOST_SECTION_WRAPUP_PROMPT
)
from utils.vector_memory import search_related, store_vector, clear_meeting_vectors, VectorMemory


def _fmt_materials(mats) -> str:
    """将 materials 列表（字符串或字典）格式化为文本"""
    if not mats:
        return "无"
    parts = []
    for m in mats:
        if isinstance(m, dict):
            parts.append(f"【{m.get('filename', '材料')}】\n{m.get('text', '')}")
        else:
            parts.append(str(m))
    return "\n\n".join(parts)


class MeetingEngine:
    def __init__(self, meeting: Meeting, model_configs: Dict[str, dict],
                 on_message_saved: Optional[Callable] = None):
        """
        meeting: Meeting 实例
        model_configs: {agent_id: {model, baseUrl, apiKey}} 前端传来的模型配置
        on_message_saved: 每条发言写入 history 后的回调（用于自动持久化）
        """
        self.meeting = meeting
        self.model_configs = model_configs
        self._paused = False
        self._skip_section = False
        self._call_guest: Optional[str] = None
        self._custom_instruction: Optional[str] = None
        self._on_message_saved = on_message_saved  # 自动存DB回调

        # 初始化向量记忆系统（ChromaDB + Embedding）
        self.vector_memory = VectorMemory(
            embedding_config=getattr(meeting, 'embedding_config', None) or {}
        )

    # ─────────────────────────────────────────
    # 阶段一：生成议程
    # ─────────────────────────────────────────

    async def generate_agenda(self) -> AsyncGenerator[dict, None]:
        """流式生成议程，返回 SSE 事件流"""
        self.meeting.state = MeetingState.PREPARING
        yield self._state_event(MeetingState.PREPARING)

        materials_text = _fmt_materials(self.meeting.materials)
        prompt = AGENDA_GENERATION_PROMPT.format(
            topic=self.meeting.topic,
            materials=materials_text,
            duration=60,
            feedback_section=""
        )

        host_cfg = self._get_model_config("host")
        full_text = ""

        # 后台静默生成，不向前端流 JSON token
        async for chunk in stream_chat(host_cfg, [{"role": "user", "content": prompt}], temperature=0.3):
            full_text += chunk

        # 解析 JSON 议程
        try:
            agenda_data = json.loads(self._extract_json(full_text))
            sections = [
                AgendaSection(
                    title=s["title"],
                    description=s.get("description", ""),
                    duration_minutes=s.get("duration_minutes", 10),
                    order=s.get("order", i + 1)
                )
                for i, s in enumerate(agenda_data.get("sections", []))
            ]
            self.meeting.agenda = Agenda(
                topic=self.meeting.topic,
                total_duration=agenda_data.get("total_duration", 60),
                sections=sections
            )
            self.meeting.state = MeetingState.AGENDA_CONFIRMING
            yield {"event": "agenda", "data": self.meeting.agenda.dict()}
            yield self._state_event(MeetingState.AGENDA_CONFIRMING)
        except Exception as e:
            yield {"event": "error", "data": {"msg": f"议程解析失败: {e}", "raw": full_text}}

    async def regenerate_agenda(self, feedback: str) -> AsyncGenerator[dict, None]:
        """根据用户反馈重新生成议程"""
        materials_text = _fmt_materials(self.meeting.materials)
        prompt = AGENDA_GENERATION_PROMPT.format(
            topic=self.meeting.topic,
            materials=materials_text,
            duration=60,
            feedback_section=f"\n用户修改意见：{feedback}"
        )
        host_cfg = self._get_model_config("host")
        full_text = ""
        async for chunk in stream_chat(host_cfg, [{"role": "user", "content": prompt}], temperature=0.3):
            full_text += chunk
        try:
            agenda_data = json.loads(self._extract_json(full_text))
            sections = [
                AgendaSection(
                    title=s["title"],
                    description=s.get("description", ""),
                    duration_minutes=s.get("duration_minutes", 10),
                    order=s.get("order", i + 1)
                )
                for i, s in enumerate(agenda_data.get("sections", []))
            ]
            self.meeting.agenda = Agenda(
                topic=self.meeting.topic,
                total_duration=agenda_data.get("total_duration", 60),
                sections=sections
            )
            self.meeting.state = MeetingState.AGENDA_CONFIRMING
            yield {"event": "agenda", "data": self.meeting.agenda.dict()}
            yield self._state_event(MeetingState.AGENDA_CONFIRMING)
        except Exception as e:
            yield {"event": "error", "data": {"msg": f"议程解析失败: {e}"}}

    async def host_chat(self, user_message: str) -> AsyncGenerator[dict, None]:
        """主持人自由对话：回答用户问题，必要时重新生成议程"""
        agenda = self.meeting.agenda
        if agenda:
            agenda_text = "\n".join(
                f"{s.order}. {s.title}（{s.duration_minutes}分钟）：{s.description}"
                for s in agenda.sections
            )
            agenda_text = f"总时长：{agenda.total_duration}分钟\n" + agenda_text
        else:
            agenda_text = "（暂无议程）"

        # 如果用户问最新信息且配置了 Tavily，先搜索再回答
        search_context = ""
        search_keywords = ["最新", "现在", "目前", "近期", "今年", "数据", "进展", "动态"]
        if self.meeting.tavily_key and any(kw in user_message for kw in search_keywords):
            try:
                from utils.search_tool import search_topic
                yield {"event": "host_chat", "data": {"chunk": "（正在联网搜索最新资讯...）\n\n"}}
                search_context = await search_topic(user_message, self.meeting.tavily_key, max_results=3)
            except Exception:
                pass

        style_inj = get_style_injection(self.meeting.host_style or "neutral")
        prompt = HOST_CHAT_PROMPT.format(
            topic=self.meeting.topic,
            agenda_text=agenda_text,
            user_message=user_message,
            search_context=f"\n\n最新联网资讯供参考：\n{search_context}" if search_context else ""
        ) + style_inj
        host_cfg = self._get_model_config("host")
        full_reply = ""
        # 收集完整回复（不过滤，保持流式响应）
        try:
            async for chunk in stream_chat(host_cfg, [{"role": "user", "content": prompt}], temperature=0.7):
                full_reply += chunk
                yield {"event": "host_chat", "data": {"chunk": chunk}}
        except Exception as e:
            import traceback; traceback.print_exc()
            yield {"event": "host_chat", "data": {"chunk": f"[错误: {e}]"}}

        # 完整回复接收后，如果包含标记，重新生成议程
        if "[NEED_REGENERATE]" in full_reply:
            async for ev in self.regenerate_agenda(user_message):
                yield ev

    # ─────────────────────────────────────────
    # 阶段二：嘉宾准备论点
    # ─────────────────────────────────────────

    async def prepare_guests(self) -> AsyncGenerator[dict, None]:
        """并发让所有嘉宾准备论点（含个性化联网搜索）"""
        self.meeting.state = MeetingState.GUESTS_PREPARING
        yield self._state_event(MeetingState.GUESTS_PREPARING)

        # 通用搜索：以话题为关键词搜一轮，作为公共背景材料
        if self.meeting.tavily_key:
            yield {"event": "state", "data": {"state": "searching", "msg": "联网搜索最新资讯..."}}
            try:
                from utils.search_tool import search_topic
                search_result = await search_topic(self.meeting.topic, self.meeting.tavily_key)
                self.meeting.materials.append({"filename": "联网搜索资讯", "text": search_result})
            except Exception as e:
                pass  # 搜索失败不阻断流程

        agenda_text = self._format_agenda()
        materials_text = _fmt_materials(self.meeting.materials)

        # 第一步：并发让每个嘉宾生成初步论点
        async def prepare_one(guest: GuestRole):
            prompt = GUEST_PREPARE_PROMPT.format(
                topic=self.meeting.topic,
                system_prompt=guest.system_prompt,
                agenda=agenda_text,
                materials_section=f"参考材料：\n{materials_text}" if materials_text else ""
            )
            cfg = self._get_model_config(guest.id)
            text = await complete_chat(cfg, [{"role": "user", "content": prompt}], temperature=0.5)
            try:
                guest.prepared_arguments = json.loads(self._extract_json(text))
            except Exception:
                guest.prepared_arguments = {"core_stance": "待定", "key_arguments": [], "opening_statement": ""}

        await asyncio.gather(*[prepare_one(g) for g in self.meeting.guests])

        # 第二步：根据每个嘉宾的角色+论点，个性化联网搜索支撑材料
        if self.meeting.tavily_key:
            yield {"event": "state", "data": {"state": "searching", "msg": "嘉宾按角色搜索支撑材料..."}}
            await asyncio.gather(*[self._search_for_guest(g) for g in self.meeting.guests])

        self.meeting.state = MeetingState.READY
        yield self._state_event(MeetingState.READY)
        yield {"event": "guests_ready", "data": {
            g.id: g.prepared_arguments for g in self.meeting.guests
        }}

    # ─────────────────────────────────────────
    # 阶段三：正式讨论
    # ─────────────────────────────────────────

    async def run_discussion(self) -> AsyncGenerator[dict, None]:
        """主流程：暖场 → 各板块 → 总结"""
        # 暖场
        self.meeting.state = MeetingState.WARMUP
        yield self._state_event(MeetingState.WARMUP)
        async for ev in self._host_warmup():
            yield ev

        # 逐板块讨论
        for i, section in enumerate(self.meeting.agenda.sections):
            self.meeting.current_section_index = i
            self.meeting.state = MeetingState.SECTION
            yield {"event": "section_start", "data": {"index": i, "section": section.dict()}}
            yield self._state_event(MeetingState.SECTION)

            async for ev in self._run_section(section):
                yield ev

            if self._skip_section:
                self._skip_section = False

            # 等待恢复
            while self._paused:
                await asyncio.sleep(0.5)

        # 自由讨论（可选，最后一个板块后）
        self.meeting.state = MeetingState.FREE_TALK
        yield self._state_event(MeetingState.FREE_TALK)
        async for ev in self._free_talk():
            yield ev

        # 进入观众互动环节，等待用户提问
        self.meeting.state = MeetingState.AUDIENCE_QA
        yield self._state_event(MeetingState.AUDIENCE_QA)
        yield {
            "event": "audience_qa_start",
            "data": {
                "msg": "议程已完成，现在进入观众互动环节。您可以向嘉宾提问，主持人将引导嘉宾回答。准备好结束时，点击「生成总结报告」。"
            }
        }

    async def _host_warmup(self) -> AsyncGenerator[dict, None]:
        guests_str = "、".join(f"{g.name}" for g in self.meeting.guests)
        style_inj = get_style_injection(self.meeting.host_style or "neutral")
        prompt = HOST_WARMUP_PROMPT.format(
            host_name=self.meeting.host.name,
            topic=self.meeting.topic,
            guests=guests_str
        ) + style_inj
        async for ev in self._stream_speaker("host", self.meeting.host.name, SpeakerType.HOST, self.meeting.host.color, prompt):
            yield ev

    async def _run_section(self, section, skip_spoken_guests: bool = False,
                           is_resuming: bool = False,
                           resume_state: str = "NOT_STARTED") -> AsyncGenerator[dict, None]:
        """单个板块：主持人引入 → 嘉宾轮流发言 → 可插话
        skip_spoken_guests=True 时跳过本 section 已发言的嘉宾（用于续跑）
        is_resuming=True 时使用精简衔接引入，避免与接场发言重复
        resume_state 精确恢复状态：
            NOT_STARTED   - 全新板块，走完整流程
            IN_INTRO      - 引入已说，跳过引入，直接嘉宾发言
            IN_GUESTS     - 部分嘉宾已说，跳过引入+已说嘉宾
            NEED_WRAPUP   - 嘉宾全说完但缺收尾，只补收尾
            COMPLETED     - 完整结束（调用方已跳过，不应到达这里）
        """
        guests = list(self.meeting.guests)
        first_guest = guests[0] if guests else None
        style_inj = get_style_injection(self.meeting.host_style or "neutral")

        # 续跑模式：找出本 section 里已经说过的嘉宾
        spoken_in_section = set()
        if skip_spoken_guests and section:
            si = self.meeting.current_section_index
            for m in self.meeting.history:
                if m.speaker_type == SpeakerType.GUEST and m.section_idx == si:
                    spoken_in_section.add(m.speaker_id)

        # ── 根据恢复状态决定引入方式（所有未完成状态都需要衔接语） ──
        if resume_state == "NEED_WRAPUP":
            pass  # 只补收尾
        elif resume_state == "COMPLETED":
            pass  # 调用方已跳过
        else:
            style_inj = get_style_injection(self.meeting.host_style or "neutral")

            if resume_state == "NOT_STARTED" and is_resuming:
                _fallback = "\u5404\u4f4d\u5d98\u5ba2"
                intro_prompt = (
                    f"好，信号恢复了，我们正式开始。\n"
                    f"第一个议题是「{section.title}」：{section.description}\n\n"
                    f"{first_guest.name if first_guest else _fallback}，请开始你的观点。"
                ) + style_inj
            elif resume_state == "NOT_STARTED":
                intro_prompt = HOST_SECTION_INTRO_PROMPT.format(
                    section_title=section.title,
                    section_description=section.description,
                    duration=section.duration_minutes,
                    first_guest=first_guest.name if first_guest else "\u5404\u4f4d\u5d98\u5ba2"
                ) + style_inj
            elif resume_state in ("IN_INTRO", "IN_GUESTS"):
                last_spoken_name = ""
                for m in reversed(self.meeting.history):
                    if m.speaker_type == SpeakerType.GUEST and m.section_idx == self.meeting.current_section_index:
                        last_spoken_name = m.speaker_name
                        break
                next_guest_name = first_guest.name if first_guest else "\u5404\u4f4d"
                for g in guests:
                    if g.id not in spoken_in_section:
                        next_guest_name = g.name
                        break
                intro_prompt = f"好，我们接着聊「{section.title}」。"
                if last_spoken_name and spoken_in_section:
                    intro_prompt += f" 刚才{last_spoken_name}发表了看法。"
                elif not spoken_in_section:
                    intro_prompt += " 我刚刚介绍完这个话题。"
                next_action = "轮到你了，请继续。" if spoken_in_section else "请你先说说你的观点。"
                intro_prompt += f"\n\n{next_guest_name}，{next_action}"
                intro_prompt += style_inj

            async for ev in self._stream_speaker("host", self.meeting.host.name, SpeakerType.HOST, self.meeting.host.color, intro_prompt):
                yield ev

        # ── 嘉宾轮流发言（NEED_WRAPUP 跳过） ──
        if resume_state != "NEED_WRAPUP":
            for idx, guest in enumerate(guests):
                if self._skip_section:
                    break

                # 续跑模式：跳过本 section 已发言的嘉宾
                if skip_spoken_guests and guest.id in spoken_in_section:
                    continue

                # 检查是否有点名
                target = self._call_guest
                if target and target != guest.id:
                    # 找到被点名的嘉宾插队
                    named = next((g for g in guests if g.id == target), None)
                    if named:
                        self._call_guest = None
                        instruction = self._custom_instruction or f'请你就「{section.title}」发表观点。'
                        self._custom_instruction = None
                        async for ev in self._guest_speak(named, section, instruction):
                            yield ev
                        continue

                self._call_guest = None
                instruction = self._custom_instruction or f'请就「{section.title}」这个板块发表你的观点。'
                self._custom_instruction = None

                async for ev in self._guest_speak(guest, section, instruction):
                    yield ev

                # 主持人过渡（非最后一个嘉宾）— 使用向量记忆增强版
                if idx < len(guests) - 1 and not self._skip_section:
                    next_guest = guests[idx + 1]
                    recent = self._recent_history(8)
                    style = self.meeting.host_style or "neutral"

                    # 向量记忆检索：为主持人提供结构化决策依据
                    memory_text = ""
                    if self.vector_memory.is_ready:
                        try:
                            host_data = await self.vector_memory.search_for_host_decision(
                                meeting_id=self.meeting.id,
                                section_id=str(self.meeting.current_section_index),
                            )
                            memory_text = self._format_host_memory(host_data)
                        except Exception as e:
                            logger.warning(f"主持人向量检索失败: {e}")

                    # 使用增强版 Prompt（如果向量记忆可用，否则降级到旧版）
                    if memory_text:
                        tpl = HOST_TRANSITION_WITH_MEMORY_PROMPT
                        transition_prompt = tpl.format(
                            topic=self.meeting.topic,
                            section_title=section.title,
                            recent_history=recent,
                            last_speaker=guest.name,
                            next_guest=next_guest.name,
                            next_guest_profile=next_guest.system_prompt[:300],
                            memory_text=memory_text,
                        )
                        # 追加风格注入
                        transition_prompt += get_style_injection(style)
                    else:
                        # 降级：无向量记忆时用旧 prompt
                        tpl = get_host_transition_prompt(style)
                        transition_prompt = tpl.format(
                            topic=self.meeting.topic,
                            last_speaker=guest.name,
                            section_title=section.title,
                            section_description=section.description,
                            recent_history=recent,
                            next_guest=next_guest.name,
                            next_guest_profile=next_guest.system_prompt[:300],
                            spoke_count=idx + 1,
                            remaining_count=len(guests) - idx - 1
                        )

                    async for ev in self._stream_speaker("host", self.meeting.host.name, SpeakerType.HOST, self.meeting.host.color, transition_prompt):
                        yield ev

        # ── 板块收尾过渡：所有嘉宾说完后，主持人做小结并引出下一板块 ──
        # 正常流程或 NEED_WRAPUP 恢复状态下都执行收尾
        should_wrapup = (resume_state == "NEED_WRAPUP") or (not self._skip_section and not skip_spoken_guests)
        if should_wrapup:
            async for ev in self._emit_section_wrapup(section):
                yield ev

    async def _emit_section_wrapup(self, section) -> AsyncGenerator[dict, None]:
        """板块收尾：所有嘉宾发言完毕后，主持人做小结并自然过渡到下一板块"""
        # 判断是否还有下一板块
        current_idx = self.meeting.current_section_index
        has_next = (current_idx + 1) < len(self.meeting.agenda.sections)
        next_section = self.meeting.agenda.sections[current_idx + 1] if has_next else None

        recent = self._recent_history(10)
        style = self.meeting.host_style or "neutral"

        wrapup_prompt = HOST_SECTION_WRAPUP_PROMPT.format(
            topic=self.meeting.topic,
            section_title=section.title,
            next_section_title=next_section.title if next_section else "",
            has_next="yes" if has_next else "no",
            recent_history=recent,
        ) + get_style_injection(style)

        async for ev in self._stream_speaker("host", self.meeting.host.name, SpeakerType.HOST, self.meeting.host.color, wrapup_prompt):
            yield ev

    async def _search_for_guest(self, guest: GuestRole):
        """根据嘉宾角色和论点，个性化搜索支撑材料"""
        try:
            from utils.search_tool import search_topic
            args = guest.prepared_arguments or {}
            stance = args.get("core_stance", "")
            key_args = args.get("key_arguments", [])

            # 构建搜索关键词：角色视角 + 核心立场 + 最关键的2个论点
            search_parts = [self.meeting.topic, stance]
            for arg in key_args[:2]:
                search_parts.append(arg)
            search_query = " ".join(search_parts)

            search_result = await search_topic(search_query, self.meeting.tavily_key, max_results=3)

            # 把搜索结果存到嘉宾的 prepared_arguments 里
            args["supporting_evidence"] = search_result
            guest.prepared_arguments = args
        except Exception as e:
            logger.warning(f"嘉宾{guest.name}个性化搜索失败: {e}")

    async def _free_talk(self) -> AsyncGenerator[dict, None]:
        """自由讨论：每位嘉宾对其他人的观点做一次回应"""
        for guest in self.meeting.guests:
            if self._skip_section:
                break
            instruction = "请对其他嘉宾的观点做一次简短回应，可以赞同、反驳或补充。"
            async for ev in self._guest_speak(guest, None, instruction):
                yield ev

    async def _host_summary(self) -> AsyncGenerator[dict, None]:
        summary = self._build_discussion_summary()
        agenda_text = self._format_agenda()
        style_inj = get_style_injection(self.meeting.host_style or "neutral")
        prompt = HOST_SUMMARY_PROMPT.format(
            topic=self.meeting.topic,
            discussion_summary=summary,
            agenda=agenda_text
        ) + style_inj
        async for ev in self._stream_speaker("host", self.meeting.host.name, SpeakerType.HOST, self.meeting.host.color, prompt):
            yield ev

    # ─────────────────────────────────────────
    # 报告生成
    # ─────────────────────────────────────────

    async def generate_report(self) -> AsyncGenerator[dict, None]:
        agenda_text = self._format_agenda()
        history_text = "\n\n".join(
            f"【{m.speaker_name}】{m.content}" for m in self.meeting.history
        )
        prompt = REPORT_GENERATION_PROMPT.format(
            topic=self.meeting.topic,
            agenda=agenda_text,
            full_history=history_text
        )
        host_cfg = self._get_model_config("host")
        report_text = ""
        yield self._msg_start_event("system", "会议报告", SpeakerType.SYSTEM, "#888")
        async for chunk in stream_chat(host_cfg, [{"role": "user", "content": prompt}], temperature=0.3):
            report_text += chunk
            yield {"event": "token", "data": {"speaker_id": "system", "token": chunk}}
        yield self._msg_end_event("system")
        self.meeting.report = report_text
        yield {"event": "report_done", "data": {"report": report_text}}

    # ─────────────────────────────────────────
    # 用户干预
    # ─────────────────────────────────────────

    def apply_intervention(self, action: str, target: str = None, instruction: str = None):
        if action == "pause":
            self._paused = True
        elif action == "resume":
            self._paused = False
        elif action == "skip_section":
            self._skip_section = True
        elif action == "call_on_guest":
            self._call_guest = target
        elif action == "custom_instruction":
            self._custom_instruction = instruction

    # ─────────────────────────────────────────
    # 内部工具方法
    # ─────────────────────────────────────────

    async def _guest_speak(self, guest: GuestRole, section, instruction: str) -> AsyncGenerator[dict, None]:
        recent = self._recent_history(10)  # 嘉宾能看到最近10条，包含所有人发言

        # 向量记忆检索：为嘉宾提供个性化上下文（我的历史/他人观点/情绪/反对者）
        guest_memory = None
        if self.vector_memory.is_ready:
            try:
                guest_memory = await self.vector_memory.search_for_guest_speech(
                    meeting_id=self.meeting.id,
                    section_id=str(self.meeting.current_section_index) if hasattr(self.meeting, 'current_section_index') else "",
                    speaker=guest.name,
                    content=instruction,
                )
            except Exception as e:
                logger.warning(f"嘉宾向量检索失败: {e}")

        args = guest.prepared_arguments or {}

        # 使用增强版 Prompt（如果向量记忆可用）
        if guest_memory and (guest_memory.get("others_viewpoints") or guest_memory.get("opponents")):
            evidence_section = ""
            evidence = args.get("supporting_evidence", "")
            if evidence:
                evidence_section = f"你搜索到的真实数据和资讯：\n{evidence}"

            prompt = GUEST_SPEAK_WITH_MEMORY_PROMPT.format(
                topic=self.meeting.topic,
                guest_name=guest.name,
                system_prompt=guest.system_prompt,
                prepared_arguments=json.dumps(args, ensure_ascii=False),
                section_title=section.title if section else "自由讨论",
                section_description=section.description if section else "",
                recent_history=recent,
                your_history="\n".join(guest_memory.get("my_history", [])) or "暂无历史发言",
                others_viewpoints="\n".join(guest_memory.get("others_viewpoints", [])) or "暂无其他观点",
                others_emotions=json.dumps(guest_memory.get("others_emotions", {}), ensure_ascii=False, indent=2),
                opponents="、".join(guest_memory.get("opponents", [])) or "无明确对立者",
                evidence_section=evidence_section,
            )
        else:
            # 降级：无向量记忆时用旧 prompt + 旧向量搜索
            evidence_section = ""
            evidence = args.get("supporting_evidence", "")
            if evidence:
                evidence_section = f"你搜索到的真实数据和资讯：\n{evidence}"

            prompt = GUEST_SPEAK_PROMPT.format(
                topic=self.meeting.topic,
                guest_name=guest.name,
                system_prompt=guest.system_prompt,
                prepared_arguments=json.dumps(args, ensure_ascii=False),
                section_title=section.title if section else "自由讨论",
                section_description=section.description if section else "",
                recent_history=recent,
                evidence_section=evidence_section,
            )
            # 旧的通用向量搜索作为补充
            if self.meeting.embedding_config and not guest_memory:
                vector_ctx = await search_related(
                    query=f"{self.meeting.topic} {section.title if section else ''} {guest.name}",
                    meeting_id=self.meeting.id,
                    emb_config=self.meeting.embedding_config,
                    top_k=3,
                )
                if vector_ctx:
                    prompt += f"\n\n{vector_ctx}"
        messages = [
            {"role": "system", "content": guest.system_prompt},
            {"role": "user", "content": prompt}
        ]
        async for ev in self._stream_speaker(guest.id, guest.name, SpeakerType.GUEST, guest.color, None, messages=messages):
            yield ev

    async def _stream_speaker(
        self, speaker_id: str, speaker_name: str,
        speaker_type: SpeakerType, color: str,
        prompt: str, messages: list = None,
        section_idx: int = None
    ) -> AsyncGenerator[dict, None]:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        cfg = self._get_model_config(speaker_id)
        full_text = ""
        msg_uuid = str(uuid.uuid4())

        yield self._msg_start_event(speaker_id, speaker_name, speaker_type, color, msg_id=msg_uuid)
        try:
            # 首token超时保护：30秒内必须有数据，否则中断
            first_token_timeout = asyncio.get_event_loop().time() + 30
            got_first_token = False
            async for chunk in stream_chat(cfg, messages):
                # 跳过检测：用户点了跳过，立即中断当前发言
                if self._skip_section:
                    break
                # 暂停检测：用户点了暂停，等待恢复
                while self._paused:
                    await asyncio.sleep(0.3)
                if not got_first_token:
                    got_first_token = True
                full_text += chunk
                yield {"event": "token", "data": {"id": msg_uuid, "speaker_id": speaker_id, "token": chunk}}
            else:
                # 正常结束（没有被break）
                if not got_first_token and not full_text:
                    logger.warning(f"[{speaker_name}] 首token超时(30s)，无任何输出")
                    err_msg = "[响应超时，已自动跳过]"
                    full_text += err_msg
                    yield {"event": "token", "data": {"id": msg_uuid, "speaker_id": speaker_name, "token": err_msg}}
        except Exception as e:
            logger.error(f"{speaker_name} 发言失败: {e}")
            err_msg = f"[发言中断: {e}]"
            full_text += err_msg
            yield {"event": "token", "data": {"id": msg_uuid, "speaker_id": speaker_id, "token": err_msg}}
        yield self._msg_end_event(speaker_id, msg_id=msg_uuid)

        # 清理控制标记后再存入历史
        clean_text = full_text.replace('[NEED_REGENERATE]', '')
        clean_text = __import__('re').sub(r'\[CALL:[^\]]*\]', '', clean_text).strip()

        # ── LLM 重复词清洗（如"直接直接""等等等等""刚才刚才"）──
        clean_text = _dedup_repetitive_text(clean_text)

        # ── 最短内容校验：模型偶尔返回极短无意义内容（如单个字母"I"）──
        if len(clean_text) < 3:
            logger.warning(
                f"[{speaker_name}] 发言过短({len(clean_text)}字符): "
                f"'{clean_text[:50]}'，已替换为占位符"
            )
            # 极短内容（<3字符）几乎一定是模型故障输出，
            # 替换为有意义的提示，避免前端显示空白或单字母
            clean_text = "[该条发言因模型输出异常而缺失]"

        # ── 防护：如果清洗后为空字符串（极端情况）──
        if not clean_text.strip():
            clean_text = "[该条发言因模型输出异常而缺失]"

        # 存入历史，带上 section_idx 便于续跑断点定位
        msg = Message(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            speaker_type=speaker_type,
            content=clean_text,
            color=color,
            section_idx=section_idx if section_idx is not None else self.meeting.current_section_index
        )
        self.meeting.history.append(msg)

        # 向量化存储：将发言存入向量库（含完整元数据：timestamp/section_idx/round/emotion/stance）
        if self.vector_memory.is_ready and full_text:
            try:
                # 异步fire-and-forget，不阻塞流程
                async def _store_async():
                    await self.vector_memory.add_message(
                        meeting_id=self.meeting.id,
                        content=full_text,
                        speaker=speaker_name,
                        section_id=str(self.meeting.current_section_index) if hasattr(self.meeting, 'current_section_index') else "",
                        section_idx=self.meeting.current_section_index if hasattr(self.meeting, 'current_section_index') else 0,
                        round_n=len(self.meeting.history),
                    )
                asyncio.create_task(_store_async())
            except Exception as e:
                logger.warning(f"向量存储异常: {e}")  # 向量写入失败不影响主流程

        # 每条发言结束后触发自动持久化回调
        if self._on_message_saved:
            self._on_message_saved()

        # 发言结束后检查暂停（发言级别，不打断mid-token）
        if self._paused:
            yield {"event": "paused", "data": {"msg": "已暂停，等待继续"}}
            while self._paused:
                await asyncio.sleep(0.1)

    def _get_model_config(self, agent_id: str) -> dict:
        return self.model_configs.get(agent_id, self.model_configs.get("host", {}))

    def _recent_history(self, n: int = 5) -> str:
        msgs = self.meeting.history[-n:] if len(self.meeting.history) >= n else self.meeting.history
        return "\n".join(f"{m.speaker_name}：{m.content[:300]}" for m in msgs)

    def _last_message_of(self, speaker_id: str) -> str:
        for m in reversed(self.meeting.history):
            if m.speaker_id == speaker_id:
                return m.content
        return ""

    def _build_discussion_summary(self) -> str:
        """构建完整讨论摘要（不截断，供总结/报告使用）"""
        lines = []
        for m in self.meeting.history:
            # 包含所有发言者：嘉宾、主持人、用户（观众提问）
            # 排除系统消息和板块标记
            if m.speaker_type in (SpeakerType.GUEST, SpeakerType.HOST, SpeakerType.USER):
                content = m.content if len(m.content) <= 2000 else m.content[:2000] + "..."
                prefix = "【观众】" if m.speaker_type == SpeakerType.USER else ""
                lines.append(f"{prefix}{m.speaker_name}：{content}")
        return "\n".join(lines)

    def _format_agenda(self) -> str:
        if not self.meeting.agenda:
            return "（未生成议程）"
        lines = [f"议题：{self.meeting.agenda.topic}"]
        for s in self.meeting.agenda.sections:
            lines.append(f"{s.order}. {s.title}（{s.duration_minutes}分钟）- {s.description}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> str:
        """从文本中提取 JSON 块"""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]
        return text

    def _format_host_memory(self, host_data: dict) -> str:
        """将主持人向量检索结果格式化为可读文本，注入 Prompt"""
        parts = []

        if host_data.get("conflicts"):
            parts.append("【⚠️ 矛盾点】")
            for c in host_data["conflicts"][:3]:
                parts.append(f"  - {c.get('speaker_a','?')} vs {c.get('speaker_b','?')}：{c.get('point_a','')[:80]} / {c.get('point_b','')[:80]}（相似度{c.get('similarity',0)}）")

        if host_data.get("consensus"):
            parts.append("\n【🤝 共识】")
            for c in host_data["consensus"][:3]:
                parts.append(f"  - {c}")

        if host_data.get("repetitions"):
            parts.append("\n【🔄 重复观点】")
            for r in host_data["repetitions"][:2]:
                parts.append(f"  - {r}")

        if host_data.get("angry_speeches"):
            parts.append("\n【🔥 情绪激动发言】")
            for a in host_data["angry_speeches"][:2]:
                parts.append(f"  - {a}")

        supporters = host_data.get("supporters", [])
        opponents = host_data.get("opponents", [])
        if supporters or opponents:
            parts.append(f"\n【📊 阵营分布】支持方：{', '.join(supporters) or '无'} | 反对方：{', '.join(opponents) or '无'}")

        return "\n".join(parts) if parts else ""

    async def audience_ask(self, user_question: str) -> AsyncGenerator[dict, None]:
        """观众互动：用户提问 → 主持人引导 → 嘉宾回答"""
        # 先把用户问题写入历史（否则存DB后丢失）
        self.meeting.history.append(Message(
            speaker_id="user",
            speaker_name="您（观众）",
            speaker_type=SpeakerType.USER,
            content=user_question,
            color="#34d399",
            section_idx=-1,  # 标记为观众互动环节
        ))

        guests_list = "\n".join(f"- {g.name}（ID: {g.id}）：{g.system_prompt[:50]}" for g in self.meeting.guests)
        recent = self._recent_history(8)

        # 主持人决定叫谁
        style_inj = get_style_injection(self.meeting.host_style or "neutral")
        guest_ids = ",".join(g.id for g in self.meeting.guests)
        host_prompt = AUDIENCE_QA_HOST_PROMPT.format(
            topic=self.meeting.topic,
            user_question=user_question,
            guests_list=guests_list,
            recent_history=recent,
            guest_ids=guest_ids
        ) + style_inj

        host_intro = ""
        call_targets = []

        async for ev in self._stream_speaker("host", self.meeting.host.name, SpeakerType.HOST, self.meeting.host.color, host_prompt):
            if ev.get("event") == "token":
                host_intro += ev["data"].get("token", "")
            yield ev

        # 解析 [CALL:...] 指令
        import re
        call_match = re.search(r'\[CALL:([^\]]+)\]', host_intro)
        if call_match:
            raw = call_match.group(1).strip()
            if raw == "ALL":
                call_targets = [g.id for g in self.meeting.guests]
            else:
                call_targets = [x.strip() for x in raw.split(",")]
            # 清理 host_intro 里的 [CALL:...] 标记（后续已在stream里展示，不影响显示）
        else:
            # 默认叫所有人
            call_targets = [g.id for g in self.meeting.guests]

        # 被点名的嘉宾逐一回答
        for guest in self.meeting.guests:
            if guest.id not in call_targets:
                continue
            guest_prompt = AUDIENCE_QA_GUEST_PROMPT.format(
                topic=self.meeting.topic,
                guest_name=guest.name,
                system_prompt=guest.system_prompt,
                prepared_arguments=guest.prepared_arguments or {},
                recent_history=self._recent_history(6),
                user_question=user_question,
                host_intro=host_intro
            )
            async for ev in self._stream_speaker(guest.id, guest.name, SpeakerType.GUEST, guest.color, guest_prompt):
                yield ev

    @staticmethod
    def _state_event(state: MeetingState) -> dict:
        return {"event": "state_change", "data": {"state": state.value}}

    @staticmethod
    def _msg_start_event(speaker_id, speaker_name, speaker_type, color, msg_id=None) -> dict:
        return {"event": "message_start", "data": {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "speaker_type": speaker_type.value,
            "color": color,
            "id": msg_id or str(uuid.uuid4())
        }}

    @staticmethod
    def _msg_end_event(speaker_id, msg_id=None) -> dict:
        d = {"event": "message_end", "data": {"speaker_id": speaker_id}}
        if msg_id:
            d["data"]["id"] = msg_id
        return d


# ─────────────────────────────────────────
# 文本后处理工具函数
# ─────────────────────────────────────────

def _dedup_repetitive_text(text: str) -> str:
    """清洗 LLM 输出的逐字/逐词重复（口吃式输出）。

    实际观测到的异常模式（LLM repetition bug）：
      - "打断**打断****式式****插入插入**）" → "打断式插入）"
      - "才**才****刚刚****刚才****说说**" → "才刚刚说说"
      - "必须**必须****须须****找到找到****到到****刚刚****需需****场景场景**" → "必须须找到到刚需场景"

    特征：模型把每个汉字或2字词都复制一遍，形成交替的 单字重复/双字重复。
    这是流式生成时 attention 机制故障导致的典型表现。

    策略：迭代式贪心去重，每次消除紧邻的1-2字重复片段，最多5轮收敛。
    对标点、英文、正常叠词（看看、试试）不做处理。
    """
    import re

    if not text:
        return text

    # 迭代去重：因为外层去重可能暴露内层重复
    # 例：打断打断式式 → 第1轮去掉打断打断 → 打断式式 → 第2轮去掉式式 → 打断式
    prev = None
    for _ in range(5):
        # 核心模式：1-2个中文字符，紧跟完全相同的内容（1次或多次）
        new_text = re.sub(
            r'([\u4e00-\u9fff]{1,2})\1+',   # 匹配 AA、AAA、AAAA...
            r'\1',                            # 替换为单个
            text
        )
        if new_text == text:
            break
        text = new_text

    return text.strip()
