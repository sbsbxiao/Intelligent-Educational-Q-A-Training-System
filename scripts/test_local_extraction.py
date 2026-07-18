from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings

TEST_TEXT = """
Python 基础课程包含函数章节和面向对象章节。函数章节讲解参数传递、返回值和作用域。
参数传递是期末考试的常见考点，题目解析需要结合函数调用过程进行说明。
""".strip()

BASE_PROMPT = """
你是一个专业的知识抽取引擎。给定一段文本，请抽取其中的实体、关系和事件。
只返回 JSON，不要返回解释文字。
JSON 格式：{"entities": [], "relations": [], "events": []}
""".strip()

EDU_STRONG_PROMPT = """
你是教育培训知识图谱抽取器。你的任务是从课程资料中抽取知识图谱数据。

必须遵守：
1. 课程名、章节名、知识点、考点、题目类型、学习任务都要当作实体。
2. 不要轻易返回空数组；只要文本里有课程、章节、知识点、考点，就必须抽取。
3. 关系类型只能用英文小写下划线：contains, explains, tests, related_to, part_of。
4. 只返回 JSON，不要解释。

JSON 格式：
{
  "entities": [
    {"name": "实体名称", "type": "Course|Chapter|Concept|Question|ExamPoint", "description": "简短描述"}
  ],
  "relations": [
    {"head": "头实体", "relation": "contains|explains|tests|related_to|part_of", "tail": "尾实体", "confidence": 0.9}
  ],
  "events": []
}
""".strip()

FEW_SHOT_PROMPT = """
你是教育培训知识图谱抽取器。请从文本中抽取实体和关系。

示例：
文本：Python 基础课程包含函数章节。函数章节讲解参数传递。
输出：
{
  "entities": [
    {"name": "Python 基础课程", "type": "Course", "description": "课程"},
    {"name": "函数章节", "type": "Chapter", "description": "章节"},
    {"name": "参数传递", "type": "Concept", "description": "知识点"}
  ],
  "relations": [
    {"head": "Python 基础课程", "relation": "contains", "tail": "函数章节", "confidence": 0.95},
    {"head": "函数章节", "relation": "explains", "tail": "参数传递", "confidence": 0.95}
  ],
  "events": []
}

现在处理新的文本。要求：
- 只返回 JSON
- 不要返回 Markdown 代码块
- 尽量抽取课程、章节、知识点、考点之间的关系
""".strip()

PROMPTS = [
    ("base", BASE_PROMPT),
    ("education_strong", EDU_STRONG_PROMPT),
    ("few_shot", FEW_SHOT_PROMPT),
]

MAX_NEW_TOKENS = 160
MAX_TIME_SECONDS = 20


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(settings.local_text_generation_path)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Project root: {ROOT_DIR}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}, max time: {MAX_TIME_SECONDS}s")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def generate(tokenizer: Any, model: Any, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"文本：\n{user_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        max_time=MAX_TIME_SECONDS,
        do_sample=False,
    )
    outputs = outputs[:, inputs.input_ids.shape[-1]:]
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def clean_json_text(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def parse_result(raw: str) -> dict[str, Any]:
    try:
        return json.loads(clean_json_text(raw))
    except Exception as exc:
        return {"parse_error": str(exc), "raw": raw}


def main() -> None:
    tokenizer, model = load_model()

    for name, prompt in PROMPTS:
        print("\n" + "=" * 80)
        print(f"Prompt case: {name}")
        started = time.perf_counter()
        raw = generate(tokenizer, model, prompt, TEST_TEXT)
        elapsed = time.perf_counter() - started
        print(f"Elapsed: {elapsed:.2f}s")
        print("\n--- Raw output ---")
        print(raw)
        print("\n--- Parsed output ---")
        print(json.dumps(parse_result(raw), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
