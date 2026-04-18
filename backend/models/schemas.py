from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


# ========== 枚举 ==========

class MeetingState(str, Enum):
    PREPARING = "preparing"           # 会前准备（生成议程）
    AGENDA_CONFIRMING = "agenda_confirming"  # 议程确认
    GUESTS_PREPARING = "guests_preparing"    # 嘉宾准备论点
    READY = "ready"                   # 就绪等待开始
    WARMUP = "warmup"                 # 暖场
    SECTION = "section"               # 板块进行中
    FREE_TALK = "free_talk"           # 自由讨论
    AUDIENCE_QA = "audience_qa"       # 观众互动环节
    SUMMARY = "summary"               # 总结
    PAUSED = "paused"                 # 用户暂停
    ENDED = "ended"                   # 结束

class SpeakerType(str, Enum):
    HOST = "host"
    GUEST = "guest"
    USER = "user"
    SYSTEM = "system"

class MessageType(str, Enum):
    TEXT = "text"
    INTERRUPT = "interrupt"
    STATE_CHANGE = "state_change"
    AGENDA_UPDATE = "agenda_update"
    PLAN = "plan"
    REPORT = "report"


# ========== 议程 ==========

class AgendaSection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str
    duration_minutes: int
    order: int

class Agenda(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    total_duration: int
    sections: List[AgendaSection]
    created_at: datetime = Field(default_factory=datetime.now)


# ========== 角色 ==========

class GuestRole(BaseModel):
    id: str
    name: str
    system_prompt: str
    model: str
    color: str = "#4f8ef7"
    prepared_arguments: Optional[Dict] = None  # 准备阶段的论点

class HostRole(BaseModel):
    id: str = "host"
    name: str = "主持人"
    system_prompt: str = "你是一个专业的会议主持人，负责控场、引导讨论、综合总结。"
    model: str
    color: str = "#f5a623"


# ========== 消息 ==========

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    speaker_id: str
    speaker_name: str
    speaker_type: SpeakerType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    section_id: Optional[str] = None
    section_idx: Optional[int] = None   # 所属板块索引，用于续跑断点定位
    color: Optional[str] = None


# ========== 会议实例 ==========

class Meeting(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    materials: List[str] = []
    agenda: Optional[Agenda] = None
    host: HostRole
    guests: List[GuestRole]
    state: MeetingState = MeetingState.PREPARING
    current_section_index: int = 0
    history: List[Message] = []
    report: Optional[str] = None
    tavily_key: Optional[str] = None
    embedding_config: Optional[Dict[str, Any]] = None # Embedding 模型配置 {platform, apiKey, baseUrl, model}
    host_style: Optional[str] = 'neutral'             # 主持风格 neutral/aggressive/gentle/analytical
    discussion_title: Optional[str] = None            # 讨论话题标题（短，用于历史卡片显示）
    topic_content: Optional[str] = None               # 话题详细内容
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ========== API 请求/响应 ==========

class CreateMeetingRequest(BaseModel):
    topic: str
    discussion_title: Optional[str] = None  # 讨论话题标题（短标题，用于历史卡片显示）
    materials: Optional[List[str]] = []
    duration_minutes: int = 60
    tavily_key: Optional[str] = None
    embedding_config: Optional[Dict[str, Any]] = None # Embedding 模型配置
    host_style: Optional[str] = 'neutral'             # 主持风格
    host: Dict[str, Any]
    guests: List[Dict[str, Any]]

class AgendaFeedbackRequest(BaseModel):
    meeting_id: str
    user_feedback: Optional[str] = None

class UserIntervention(BaseModel):
    meeting_id: str
    action: str  # "pause" | "resume" | "skip_section" | "call_on_guest" | "custom_instruction"
    target: Optional[str] = None
    instruction: Optional[str] = None

class StartMeetingRequest(BaseModel):
    meeting_id: str


# ========== SSE 事件 ==========

class SSEEvent(BaseModel):
    event: str   # "token" | "message_start" | "message_end" | "state_change" | "agenda" | "error"
    data: Dict[str, Any]
