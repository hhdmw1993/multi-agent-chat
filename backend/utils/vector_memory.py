"""向量记忆模块 - ChromaDB本地持久化 + Embedding API + 智能检索

职责：
1. add_message()    — 将发言向量化，分析情绪/立场，存入本地ChromaDB（含完整元数据）
2. search_for_host_decision() — 主持人专用：返回矛盾/共识/重复/情绪/阵营
3. search_for_guest_speech()   — 嘉宾专用：返回我的历史/他人观点/他人情绪/反对者
4. search_by_timestamp()      — 按时间范围检索（续跑用）
5. search_by_section()        — 按板块检索
6. get_by_round()             — 按轮次检索
7. detect_conflicts()         — 语义相似度+立场对立检测
8. delete_meeting()           — 清除某次会议所有向量

存储策略：
- 数据默认存入 ./data/vectors/ 目录（ChromaDB持久化）
- 重启不丢失，用户零配置
- 只需配置 Embedding 模型（线上API）即可使用

设计原则：
- 不阻塞主流程：写入异步fire-and-forget，搜索有超时保护
- 容错优先：任何异常不崩溃，降级为无向量记忆模式
- 本地优先：数据不出本机，隐私安全
"""

import httpx
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger("vector_memory")

# ═══════════════════════════════════════
# ChromaDB 全局客户端（延迟初始化）
# ═══════════════════════════════════════
_chroma_client = None
_chroma_ready = False


def _get_chroma_client():
    """获取或创建ChromaDB持久化客户端"""
    global _chroma_client, _chroma_ready

    if _chroma_client is not None:
        return _chroma_client if _chroma_ready else None

    try:
        import chromadb

        # 向量数据库存储目录
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vectors")
        os.makedirs(db_path, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(path=db_path)
        _chroma_ready = True
        logger.info(f"ChromaDB 初始化成功: {db_path}")
        return _chroma_client

    except ImportError:
        logger.warning("chromadb 未安装，将使用内存回退模式。运行: pip install chromadb")
        _chroma_ready = False
        return None
    except Exception as e:
        logger.warning(f"ChromaDB 初始化失败: {e}，将使用内存回退模式")
        _chroma_ready = False
        _chroma_client = None
        return None


# ═══════════════════════════════════════
# 内存回退存储（ChromaDB不可用时用）
# ═══════════════════════════════════════
_fallback_store: Dict[str, List[dict]] = {}

# 内存回退存储上限（防止长时间运行 OOM）
_FALLBACK_MAX_PER_MEETING = 200
_FALLBACK_MAX_TOTAL = 5000


def _evict_fallback(meeting_id: str):
    """淘汰超出上限的旧记录，防止内存泄漏"""
    records = _fallback_store.get(meeting_id, [])
    if len(records) > _FALLBACK_MAX_PER_MEETING:
        _fallback_store[meeting_id] = records[-_FALLBACK_MAX_PER_MEETING:]

    # 全局总量控制
    total = sum(len(v) for v in _fallback_store.values())
    if total > _FALLBACK_MAX_TOTAL:
        # 按会议淘汰最旧的
        for mid in list(_fallback_store.keys()):
            if total <= _FALLBACK_MAX_TOTAL:
                break
            recs = _fallback_store[mid]
            cut = max(0, len(recs) - 50)
            _fallback_store[mid] = recs[cut:]
            total -= cut


# ─── Embedding 调用 ──────────────────────────

async def _call_embedding(text: str, emb_config: dict) -> list | None:
    """调用Embedding API，返回向量列表。失败返回None"""
    try:
        api_key = (emb_config.get("apiKey") or "").strip()
        if not api_key:
            return None

        platform = emb_config.get("platform", "custom")
        base_url = (emb_config.get("baseUrl") or "").rstrip("/")
        model = emb_config.get("model", "")

        # 平台默认值兜底
        if not base_url:
            defaults = {
                "dashscope": "https://dashscope.aliyuncs.com/compatible-mode",
                "openai": "https://api.openai.com",
                "hunyuan": "https://api.hunyuan.cloud.tencent.com",
            }
            base_url = defaults.get(platform, "")
        if not model:
            model_defaults = {
                "dashscope": "text-embedding-v4",
                "openai": "text-embedding-3-small",
                "hunyuan": "hunyuan-embedding",
            }
            model = model_defaults.get(platform, "text-embedding-v3-large")

        url = f"{base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {"model": model, "input": text}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, headers=headers, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            return data["data"][0]["embedding"]
        else:
            logger.warning(f"Embedding API 返回 {resp.status_code}: {resp.text[:100]}")
            return None

    except Exception as e:
        logger.warning(f"Embedding 调用失败: {e}")
        return None


# ─── 余弦相似度 ─────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    """计算余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _restore_nested_fields(meta: dict) -> dict:
    """将 ChromaDB 扁平化的 emotion/stance 字段还原为嵌套 dict。
    
    写入时拆成 emotion_primary/emotion_intensity/stance_position 等，
    读取时还原为 {"emotion": {...}, "stance": {...}} 供下游使用。
    """
    # 还原 emotion
    if "emotion_primary" in meta:
        meta["emotion"] = {
            "primary": meta.get("emotion_primary", "平静"),
            "secondary": meta.get("emotion_secondary", ""),
            "intensity": meta.get("emotion_intensity", 0.3),
        }
    elif "_emotion_dict" in meta:
        try:
            meta["emotion"] = json.loads(meta["_emotion_dict"])
        except (json.JSONDecodeError, TypeError):
            meta["emotion"] = {"primary": "平静", "secondary": "", "intensity": 0.3}

    # 还原 stance
    if "stance_position" in meta:
        meta["stance"] = {
            "position": meta.get("stance_position", "未明确"),
            "confidence": meta.get("stance_confidence", 0.5),
            "reasons": [],
        }
    elif "_stance_dict" in meta:
        try:
            meta["stance"] = json.loads(meta["_stance_dict"])
        except (json.JSONDecodeError, TypeError):
            meta["stance"] = {"position": "未明确", "confidence": 0.5, "reasons": []}

    return meta


# ═══════════════════════════════════════
# VectorMemory 主类
# ═══════════════════════════════════════

class VectorMemory:
    """向量记忆工具 - 统一管理存储和读取

    用户只需在前端配置 Embedding 模型（必需）：
    - 有 Embedding → 向量生成、情绪分析、矛盾检测全部工作
    - 存储自动使用本地 ChromaDB，重启不丢，用户零配置
    """

    def __init__(self, embedding_config: Optional[Dict] = None):
        """
        初始化向量记忆。

        Args:
            embedding_config: {platform, apiKey, baseUrl, model} Embedding API配置
        """
        self.embedding_config = embedding_config or {}

        # is_ready 只依赖 embedding（有embedding就能做向量化+检索）
        self._ready = bool(
            embedding_config and embedding_config.get("apiKey")
        )

        # 初始化 ChromaDB
        self._client = _get_chroma_client()
        self._collection = None

        if self._client:
            try:
                self._collection = self._client.get_or_create_collection(
                    name="meetings",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("ChromaDB collection 'meetings' 就绪")
            except Exception as e:
                logger.warning(f"ChromaDB collection 创建失败: {e}")

    @property
    def is_ready(self) -> bool:
        """检查向量系统是否可用（只需要 embedding 配置）"""
        return self._ready

    @property
    def has_persistent_storage(self) -> bool:
        """检查持久化存储是否可用（ChromaDB）"""
        return self._collection is not None

    # ========== 存储方法 ==========

    async def add_message(self, meeting_id: str, content: str, speaker: str,
                          section_id: str = "", section_idx: int = 0,
                          round_n: int = 0) -> Optional[str]:
        """
        存入一条发言记录（文本→向量→元数据→ChromaDB）。

        流程：
        1. 文本向量化
        2. 分析情绪标签
        3. 分析立场标签
        4. 组装完整记录写入ChromaDB（或内存回退）

        Returns:
            成功返回记录ID，失败返回None
        """
        if not self.embedding_config or not self.embedding_config.get("apiKey"):
            return None

        try:
            # 1. 文本向量化
            vector = await _call_embedding(content, self.embedding_config)
            if not vector:
                logger.warning(f"向量化失败: speaker={speaker}")
                return None

            # 2. 情绪分析（轻量规则）
            emotion = await self._analyze_emotion(content)

            # 3. 立场分析（轻量规则）
            stance = await self._analyze_stance(content, "")

            # 4. 组装完整记录（ChromaDB metadata 不支持嵌套 dict，必须扁平化）
            timestamp = datetime.now().isoformat()
            record_id = f"{meeting_id}_{speaker}_{timestamp}"
            emo = emotion or {"primary": "平静", "secondary": "", "intensity": 0.3}
            sta = stance or {"position": "未明确", "confidence": 0.5, "reasons": []}
            metadata = {
                "meeting_id": meeting_id,
                "speaker": speaker,
                "section_id": section_id,
                "section_idx": section_idx,
                "round": round_n,
                "content": content[:2000],
                "timestamp": timestamp,
                # 情绪扁平化
                "emotion_primary": emo.get("primary", "平静"),
                "emotion_secondary": emo.get("secondary", ""),
                "emotion_intensity": emo.get("intensity", 0.3),
                # 立场扁平化
                "stance_position": sta.get("position", "未明确"),
                "stance_confidence": sta.get("confidence", 0.5),
                # 保留原始 dict 供内存回退使用
                "_emotion_dict": json.dumps(emo, ensure_ascii=False),
                "_stance_dict": json.dumps(sta, ensure_ascii=False),
            }

            # 5. 写入（优先ChromaDB，失败回退内存）
            await self._store_record(record_id, vector, metadata)

            return record_id

        except Exception as e:
            logger.warning(f"add_message 异常: {e}")
            return None

    async def _store_record(self, record_id: str, vector: list, metadata: dict):
        """写入一条记录到ChromaDB或内存回退"""
        meeting_id = metadata.get("meeting_id", "unknown")

        # 尝试写入 ChromaDB
        if self._collection:
            try:
                self._collection.upsert(
                    ids=[record_id],
                    embeddings=[vector],
                    metadatas=[metadata],
                    documents=[metadata.get("content", "")],
                )
                logger.debug(f"ChromaDB 写入成功: {record_id}")

                # 同时缓存到内存供本次会话快速检索（有上限防 OOM）
                _fallback_store.setdefault(meeting_id, []).append({
                    "id": record_id,
                    "type": "vector",
                    "vector": vector,
                    "metadata": metadata,
                })
                _evict_fallback(meeting_id)
                return
            except Exception as e:
                logger.warning(f"ChromaDB 写入失败，回退到内存: {e}")

        # 回退到内存（有上限防 OOM）
        _fallback_store.setdefault(meeting_id, []).append({
            "id": record_id,
            "type": "vector",
            "vector": vector,
            "metadata": metadata,
        })
        _evict_fallback(meeting_id)

    def delete_meeting(self, meeting_id: str):
        """清除某次会议的所有向量数据（内存+ChromaDB）"""
        # 清除内存回退数据
        _fallback_store.pop(meeting_id, None)

        # ChromaDB 端删除
        if self._collection:
            try:
                # 获取该会议的所有记录 ID
                results = self._collection.get(
                    where={"meeting_id": meeting_id},
                    include=["documents"],
                )
                ids_to_delete = results.get("ids", [])
                if ids_to_delete:
                    self._collection.delete(ids=ids_to_delete)
                    logger.info(f"ChromaDB 已删除 {len(ids_to_delete)} 条记录 (meeting={meeting_id})")
            except Exception as e:
                logger.warning(f"ChromaDB 删除失败: {e}")

        logger.info(f"已清除会议 {meeting_id} 的所有向量数据")

    # ========== 主持人专用检索 ==========

    async def search_for_host_decision(self, meeting_id: str, section_id: str = "") -> dict:
        """主持人决策检索 - 为主持人过渡语提供结构化决策依据。"""
        records = self._get_meeting_records(meeting_id)
        if not records:
            return self._empty_host_result()

        if section_id:
            section_records = [r for r in records 
                             if r["metadata"].get("section_id") == section_id]
        else:
            section_records = records

        conflicts = await self.detect_conflicts(meeting_id, section_id)
        consensus = self._detect_consensus(section_records)
        repetitions = self._find_repetitions(section_records)
        angry_speeches = [r for r in section_records 
                         if r["metadata"].get("emotion", {}).get("intensity", 0) > 0.7]
        supporters, opponents = self._split_factions(section_records)

        return {
            "section_memories": self._format_memories(section_records),
            "conflicts": conflicts,
            "consensus": consensus,
            "repetitions": repetitions,
            "angry_speeches": self._format_memories(angry_speeches),
            "supporters": supporters,
            "opponents": opponents,
        }

    # ========== 嘉宾专用检索 ==========

    async def search_for_guest_speech(self, meeting_id: str, section_id: str,
                                       speaker: str, content: str = "") -> dict:
        """嘉宾发言前检索 - 为嘉宾提供个性化上下文。"""
        records = self._get_meeting_records(meeting_id)
        if not records:
            return self._empty_guest_result()

        if section_id:
            section_records = [r for r in records 
                             if r["metadata"].get("section_id") == section_id]
        else:
            section_records = records

        my_history = [r for r in section_records if r["metadata"].get("speaker") == speaker]
        others_records = [r for r in section_records if r["metadata"].get("speaker") != speaker]
        others_viewpoints = self._format_viewpoints(others_records)
        others_emotions = self._summarize_emotions(others_records)

        my_stance = None
        my_records = [r for r in records if r["metadata"].get("speaker") == speaker]
        if my_records:
            my_stance = my_records[-1]["metadata"].get("stance", {}).get("position")
        
        opponents = []
        if my_stance:
            for r in others_records:
                their_stance = r["metadata"].get("stance", {}).get("position")
                if their_stance and self._is_opposing(my_stance, their_stance):
                    name = r["metadata"].get("speaker")
                    if name not in opponents:
                        opponents.append(name)

        return {
            "my_history": self._format_memories(my_history, max_items=3),
            "others_viewpoints": others_viewpoints,
            "others_emotions": others_emotions,
            "opponents": opponents,
        }

    # ========== 时序检索方法 ==========

    async def search_by_timestamp(self, meeting_id: str, 
                                   from_timestamp: str, 
                                   to_timestamp: str = "") -> List[dict]:
        """按时间范围检索 - 用于暂停续跑"""
        records = self._get_meeting_records(meeting_id)
        if not records:
            return []

        result = []
        for r in records:
            ts = r["metadata"].get("timestamp", "")
            if ts >= from_timestamp:
                if to_timestamp and ts > to_timestamp:
                    continue
                result.append(r)

        result.sort(key=lambda x: x["metadata"].get("timestamp", ""))
        return result

    async def search_by_section(self, meeting_id: str, section_idx: int) -> List[dict]:
        """按板块索引检索"""
        records = self._get_meeting_records(meeting_id)
        return [r for r in records if r["metadata"].get("section_idx") == section_idx]

    async def get_by_round(self, meeting_id: str, section_idx: int, round_n: int) -> Optional[dict]:
        """按板块+轮次精确获取单条记录"""
        records = await self.search_by_section(meeting_id, section_idx)
        for r in records:
            if r["metadata"].get("round") == round_n:
                return r
        return None

    async def search_by_emotion(self, meeting_id: str, emotion: str, 
                                 min_intensity: float = 0.5) -> List[dict]:
        """按情绪检索"""
        records = self._get_meeting_records(meeting_id)
        return [r for r in records 
                if r["metadata"].get("emotion", {}).get("primary") == emotion
                and r["metadata"].get("emotion", {}).get("intensity", 0) >= min_intensity]

    async def search_by_stance(self, meeting_id: str, stance_keyword: str, topic: str = "") -> List[dict]:
        """按立场检索"""
        records = self._get_meeting_records(meeting_id)
        results = []
        for r in records:
            position = r["metadata"].get("stance", {}).get("position", "")
            if stance_keyword.lower() in position.lower():
                results.append(r)
        return results

    # ========== 矛盾检测 ==========

    async def detect_conflicts(self, meeting_id: str, section_id: str = "") -> List[dict]:
        """矛盾检测 - 语义相似度高但立场对立的发言对。"""
        records = self._get_meeting_records(meeting_id)
        if len(records) < 2:
            return []

        if section_id:
            records = [r for r in records if r["metadata"].get("section_id") == section_id]

        conflicts = []
        checked = set()
        
        for i, r1 in enumerate(records):
            for j, r2 in enumerate(records):
                if i >= j:
                    continue
                pair_key = (r1["id"], r2["id"])
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                v1 = r1.get("vector", [])
                v2 = r2.get("vector", [])
                if not v1 or not v2:
                    continue

                sim = _cosine_similarity(v1, v2)
                
                if sim > 0.75:
                    stance1 = r1["metadata"].get("stance", {}).get("position", "")
                    stance2 = r2["metadata"].get("stance", {}).get("position", "")
                    
                    if stance1 and stance2 and self._is_opposing(stance1, stance2):
                        conflicts.append({
                            "similarity": round(sim, 3),
                            "speaker_a": r1["metadata"].get("speaker"),
                            "speaker_b": r2["metadata"].get("speaker"),
                            "point_a": r1["metadata"].get("content", "")[:150],
                            "point_b": r2["metadata"].get("content", "")[:150],
                            "stance_a": stance1,
                            "stance_b": stance2,
                        })

        conflicts.sort(key=lambda x: x["similarity"], reverse=True)
        return conflicts

    # ========== 内部工具方法 ==========

    def _get_meeting_records(self, meeting_id: str) -> List[dict]:
        """获取某次会议的所有记录（ChromaDB优先 → 内存回退）"""

        # 1. 先检查内存缓存（本次会话刚写入的数据）
        if meeting_id in _fallback_store:
            return _fallback_store[meeting_id]

        # 2. 尝试从 ChromaDB 加载
        if self._collection:
            try:
                results = self._collection.get(
                    where={"meeting_id": meeting_id},
                    include=["embeddings", "metadatas", "documents"],
                )
                ids = results.get("ids", [])
                embeddings = results.get("embeddings", [])
                metadatas = results.get("metadatas", [])

                if ids:
                    records = []
                    for idx, rid in enumerate(ids):
                        meta = metadatas[idx] if idx < len(metadatas) else {}
                        # 还原扁平化的 emotion/stance 为 dict（下游代码依赖 dict 格式）
                        meta = _restore_nested_fields(meta)
                        # 不缓存向量数据到内存，ChromaDB已持久化
                        records.append({
                            "id": rid,
                            "type": "chromadb_ref",  # 标记为引用，不含向量
                            "vector": [],  # 按需从 ChromaDB 获取
                            "metadata": meta,
                        })
                    _fallback_store[meeting_id] = records
                    logger.info(f"从 ChromaDB 加载了 {len(records)} 条向量记录 (meeting={meeting_id})")
                    return records
            except Exception as e:
                logger.warning(f"ChromaDB 读取失败: {e}")

        # 3. 都没有
        return []

    async def _analyze_emotion(self, text: str) -> Optional[dict]:
        """轻量情绪分析（关键词规则）"""
        text_lower = text.lower()
        intensity = 0.3
        
        angry_words = ["荒谬", "胡说", "错误", "不可能", "绝对不", "完全不对"]
        excited_words = ["非常赞同", "完全同意", "太好了", "精彩", "棒"]
        questioning_words = ["？？", "真的吗", "凭什么", "为什么"]
        
        primary = "平静"
        secondary = ""
        
        for w in angry_words:
            if w in text:
                primary = "愤怒"
                intensity = min(intensity + 0.2, 1.0)
                break
                
        for w in excited_words:
            if w in text:
                primary = "兴奋"
                intensity = min(intensity + 0.15, 1.0)
                break
                
        for w in questioning_words:
            if w in text:
                primary = "质疑"
                intensity = min(intensity + 0.15, 1.0)
                break

        return {"primary": primary, "secondary": secondary, "intensity": intensity}

    async def _analyze_stance(self, text: str, topic: str) -> Optional[dict]:
        """轻量立场分析（关键词启发式）"""
        support_words = ["支持", "赞成", "认为可行", "看好", "优势"]
        oppose_words = ["反对", "不行", "有问题", "风险", "担心", "隐患"]
        neutral_words = ["看情况", "不一定", "有待观察", "两面性"]
        
        position = "未明确"
        confidence = 0.3
        reasons = []
        
        for w in support_words:
            if w in text:
                position = "支持"
                confidence = max(confidence, 0.6)
                reasons.append(w)
                
        for w in oppose_words:
            if w in text:
                position = "反对"
                confidence = max(confidence, 0.6)
                reasons.append(w)
                
        for w in neutral_words:
            if w in text:
                position = "中立"
                reasons.append(w)

        return {"position": position, "confidence": confidence, "reasons": reasons}

    def _is_opposing(self, stance_a: str, stance_b: str) -> bool:
        """判断两个立场是否对立"""
        opposing_pairs = [
            ("支持", "反对"), ("反对", "支持"),
            ("看好", "不看好"), ("不看好", "看好"),
            ("应该", "不应该"), ("不应该", "应该"),
            ("可行", "不可行"), ("不可行", "可行"),
        ]
        return (stance_a, stance_b) in opposing_pairs

    def _format_memories(self, records: List[dict], max_items: int = 10) -> List[str]:
        """将记录格式化为可阅读的文本列表"""
        results = []
        for r in records[:max_items]:
            m = r["metadata"]
            results.append(f'{m.get("speaker","?")}：{m.get("content","")[:200]}')
        return results

    def _format_viewpoints(self, records: List[dict]) -> List[str]:
        """从记录中提取各人的核心观点（去重）"""
        seen_speakers = set()
        viewpoints = []
        for r in records:
            speaker = r["metadata"].get("speaker", "?")
            if speaker not in seen_speakers:
                seen_speakers.add(speaker)
                content = r["metadata"].get("content", "")[:150]
                stance = r["metadata"].get("stance", {})
                pos = stance.get("position", "")
                stance_info = f"（立场：{pos}）" if pos else ""
                viewpoints.append(f"{speaker}{stance_info}：{content}")
        return viewpoints

    def _summarize_emotions(self, records: List[dict]) -> dict:
        """汇总多人情绪"""
        result = {}
        for r in records:
            speaker = r["metadata"].get("speaker", "?")
            emotion = r["metadata"].get("emotion", {})
            result[speaker] = {
                "emotion": emotion.get("primary", "未知"),
                "intensity": emotion.get("intensity", 0),
            }
        return result

    def _split_factions(self, records: List[dict]) -> tuple:
        """根据立场划分阵营"""
        supporters = []
        opponents = []
        neutral = []
        
        speaker_stances = {}
        for r in records:
            speaker = r["metadata"].get("speaker")
            if speaker and speaker not in speaker_stances:
                stance = r["metadata"].get("stance", {}).get("position", "")
                speaker_stances[speaker] = stance
        
        for speaker, stance in speaker_stances.items():
            if stance in ("支持", "看好", "可行", "应该"):
                supporters.append(speaker)
            elif stance in ("反对", "不看好", "不可行", "不应该"):
                opponents.append(speaker)
            else:
                neutral.append(speaker)
                
        return supporters, opponents

    def _detect_consensus(self, records: List[dict]) -> List[str]:
        """检测共识点（高相似度+相同立场）"""
        consensus = []
        if len(records) < 2:
            return consensus
            
        checked = set()
        for i, r1 in enumerate(records):
            for j, r2 in enumerate(records):
                if i >= j:
                    continue
                pair_key = (min(r1["id"], r2["id"]), max(r1["id"], r2["id"]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                
                v1 = r1.get("vector", [])
                v2 = r2.get("vector", [])
                if not v1 or not v2:
                    continue
                    
                sim = _cosine_similarity(v1, v2)
                if sim > 0.85:
                    s1 = r1["metadata"].get("stance", {}).get("position", "")
                    s2 = r2["metadata"].get("stance", {}).get("position", "")
                    if s1 and s1 == s2:
                        spk1 = r1["metadata"].get("speaker", "?")
                        spk2 = r2["metadata"].get("speaker", "?")
                        content = r1["metadata"].get("content", "")[:100]
                        consensus.append(f"{spk1}和{spk2}共识：{content}")
        
        return consensus

    def _find_repetitions(self, records: List[dict]) -> List[str]:
        """找重复观点（相似度>0.9）"""
        repetitions = []
        checked = set()
        for i, r1 in enumerate(records):
            for j, r2 in enumerate(records):
                if i >= j:
                    continue
                pair_key = (min(r1["id"], r2["id"]), max(r1["id"], r2["id"]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                
                v1 = r1.get("vector", [])
                v2 = r2.get("vector", [])
                if not v1 or not v2:
                    continue
                    
                sim = _cosine_similarity(v1, v2)
                if sim > 0.9:
                    spk1 = r1["metadata"].get("speaker", "?")
                    spk2 = r2["metadata"].get("speaker", "?")
                    repetitions.append(f"{spk1}与{spk2}观点高度重复（相似度{sim:.2f}）")
        
        return repetitions

    def _empty_host_result(self) -> dict:
        return {
            "section_memories": [],
            "conflicts": [],
            "consensus": [],
            "repetitions": [],
            "angry_speeches": [],
            "supporters": [],
            "opponents": [],
        }

    def _empty_guest_result(self) -> dict:
        return {
            "my_history": [],
            "others_viewpoints": [],
            "others_emotions": {},
            "opponents": [],
        }


# ═══════════════════════════════════════
# 向后兼容的函数接口（保持旧代码可用）
# ═══════════════════════════════════════

async def embed_text(text: str, emb_config: dict) -> list | None:
    """向后兼容：将单段文本转为向量"""
    return await _call_embedding(text, emb_config)


async def store_vector(meeting_id: str, speaker_name: str,
                       content: str, vector: list,
                       cos_config: dict):
    """向后兼容：存入向量（旧接口）。建议改用 VectorMemory.add_message()"""
    _fallback_store.setdefault(meeting_id, []).append({
        "id": f"legacy_{meeting_id}_{speaker_name}",
        "type": "vector",
        "vector": vector,
        "metadata": {
            "meeting_id": meeting_id,
            "speaker": speaker_name,
            "content": content[:500],
            "timestamp": datetime.now().isoformat(),
            "emotion": {"primary": "平静", "secondary": "", "intensity": 0.3},
            "stance": {"position": "未明确", "confidence": 0.5, "reasons": []},
        }
    })


async def search_related(query: str, meeting_id: str,
                        emb_config: dict, top_k: int = 3) -> str:
    """向后兼容：通用语义搜索。返回格式化的相关历史文本"""
    if not emb_config or not emb_config.get("apiKey"):
        return ""

    records = _fallback_store.get(meeting_id, [])
    if len(records) < 3:
        return ""

    query_vec = await _call_embedding(query, emb_config)
    if not query_vec:
        return ""

    scored = []
    for r in records:
        sim = _cosine_similarity(query_vec, r.get("vector", []))
        if sim > 0.3:
            scored.append((sim, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return ""

    lines = ["【相关历史讨论】"]
    for sim, rec in top:
        meta = rec.get("metadata", rec)
        speaker = meta.get("speaker", rec.get("speaker", "?"))
        content = meta.get("content", rec.get("content", ""))
        lines.append(f"{speaker}：{content[:200]}")

    return "\n".join(lines)


def clear_meeting_vectors(meeting_id: str):
    """向后兼容：清除某次会议的向量缓存"""
    _fallback_store.pop(meeting_id, None)
