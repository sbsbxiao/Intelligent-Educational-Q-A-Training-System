from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.conversation_memory import ConversationMemoryStore


class FakeShortMemoryModel:
    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        records = json.loads(user_prompt)
        return json.dumps(
            [
                {
                    "question": f"短问{item.get('turn_index')}: {item.get('question', '')[:20]}",
                    "answer": f"短答{item.get('turn_index')}: {item.get('answer', '')[:30]}",
                }
                for item in records[-5:]
            ],
            ensure_ascii=False,
        )


class FakeLongMemoryModel:
    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        record = json.loads(user_prompt)
        return json.dumps(
            {"memory": f"长期关键信息{record.get('turn_index')}: {record.get('question', '')[:30]}"},
            ensure_ascii=False,
        )


async def main() -> None:
    temp_dir = Path(mkdtemp(prefix="agenthub-memory-demo-"))
    try:
        store = ConversationMemoryStore(
            file_path=str(temp_dir / "conversation_history.json"),
            records_file_path=str(temp_dir / "conversation_records.json"),
            short_memory_file_path=str(temp_dir / "short_memory.json"),
            long_memory_file_path=str(temp_dir / "long_memory.json"),
        )
        store._short_memory_model = FakeShortMemoryModel()
        store._long_memory_model = FakeLongMemoryModel()

        for index in range(1, 31):
            agent = "qa" if index % 2 else "education"
            metadata = {"intent": "factoid"} if agent == "qa" else {"skill": "study_plan", "tools_used": ["course_material_search"], "sources": []}
            store.append_record(
                question=f"问题 {index}",
                answer=f"回答 {index}",
                agent=agent,
                metadata=metadata,
            )
            await store.refresh_short_memory()
            await store.refresh_long_memory()

        records = store.load_records()
        short_memories = store.load_short_memories()
        long_memories = store.load_long_memories()
        prompt_context = store.format_history_with_short_memory()

        print("records_count:", len(records))
        print("records_first_turn:", records[0]["turn_index"])
        print("records_last_turn:", records[-1]["turn_index"])
        print("short_memory_count:", len(short_memories))
        print("long_memory_count:", len(long_memories))
        print("has_short_label:", "短时记忆" in prompt_context)
        print("has_long_label:", "长时记忆" in prompt_context)
        print("latest_short_memory:", short_memories[-1] if short_memories else {})
        print("latest_long_memory:", long_memories[-1] if long_memories else {})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
