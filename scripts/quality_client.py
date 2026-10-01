"""Host-local bounded teacher-forced corpus and deterministic generation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import requests
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trials", type=int, default=0)  # Common lifecycle runner.
parser.add_argument("--activation-only", action="store_true")
parser.add_argument("--routing-diagnostic", action="store_true")
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
session = requests.Session()
session.trust_env = False
logits_dir = Path(os.environ["MEGARTX_LOGITS_DIR"])
records = []


def chat(text):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False)
    return ids if isinstance(ids, list) else ids["input_ids"]


def exact_prompt(n):
    base = chat("Read the following notes and answer the final question. " + "A box contains red and blue marbles. The recorder counts each color twice and verifies the total. " * (n // 8 + 100))
    return base[:n]


def request(case_id, tokens, output_tokens=1, teacher_forced=True):
    before = sorted(p.name for p in logits_dir.glob("logits-*.pt"))
    payload = {"model": "gemma4-nvfp4", "prompt": tokens, "max_tokens": output_tokens, "temperature": 0, "seed": 1234, "ignore_eos": teacher_forced, "return_token_ids": True, "logprobs": 1}
    if teacher_forced:
        payload["prompt_logprobs"] = 1
    prompt_sha256 = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    marker = logits_dir.parent / "capture-request.json"
    if marker.exists():
        raise RuntimeError("A prior capture request is still active")
    marker.write_text(json.dumps({"id": case_id, "prompt_sha256": prompt_sha256, "prompt_token_ids": tokens, "teacher_forced": teacher_forced}))
    try:
        response = session.post("http://127.0.0.1:18000/v1/completions", json=payload, timeout=(5, 300))
        response.raise_for_status()
        data = response.json()
    finally:
        marker.unlink()
    usage = data["usage"]
    if usage["prompt_tokens"] != len(tokens):
        raise RuntimeError("Server token count does not match fixed teacher-forced prefix")
    after = sorted(p.name for p in logits_dir.glob("logits-*.pt"))
    files = [x for x in after if x not in before]
    if not files:
        raise RuntimeError("No full-vocabulary logits were captured")
    choice = data["choices"][0]
    row = {"id": case_id, "teacher_forced": teacher_forced, "prompt_token_ids": tokens, "prompt_sha256": prompt_sha256, "logit_files": files, "usage": usage, "output_text": choice["text"], "completion_token_ids": choice.get("token_ids"), "logprobs": choice.get("logprobs"), "prompt_logprobs": data.get("prompt_logprobs"), "finish_reason": choice["finish_reason"]}
    records.append(row)
    (args.output / "quality-requests.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"id": case_id, "prompt_tokens": len(tokens), "completion_tokens": usage["completion_tokens"], "logit_batches": len(files)}), flush=True)


for i, text in enumerate([
    "Explain why a triangle's interior angles sum to 180 degrees in Euclidean geometry, using parallel lines.",
    "A library lends five books on Monday and receives two back on Tuesday. Starting with 30 books, describe the inventory after both events.",
    "Translate the sentence 'The garden is quiet in the morning' into French, then explain the meaning in English.",
    "Read this code: def square(x): return x*x. Explain the outputs for x=-3, x=0, and x=4, and give a unit test.",
]):
    # Fixed prefixes are independently supplied to both model runs. Synthetic
    # regression cases are not described as held-out task evaluation.
    tokens = chat(text + " Provide a careful explanation with one concrete example.")
    request(f"semantic-{i}", tokens)
for n in ((1025,) if args.activation_only else (1023, 1024, 1025, 2048, 8192)):
    request(f"context-{n}", exact_prompt(n))
for i, text in enumerate(["What is two plus two? Reply with only the numeral.", "What is the capital of France? Reply with only its name."]):
    request(f"greedy-{i}", chat(text), output_tokens=32, teacher_forced=False)
if args.routing_diagnostic:
    for i, text in enumerate([
        "用中文讲一个科学家修理火星探测器的故事。解释电路、太阳能电池、温度传感器、无线通信和实验记录。",
        "日本語で、古い図書館と海辺の町を舞台にした物語を書いてください。登場人物の会話と風景を描写してください。",
        "Опишите на русском языке историю путешествия по лесу, реке и горам. Добавьте диалог и объяснение того, как работает компас.",
        "اكتب باللغة العربية قصة عن طالب يتعلم الرياضيات والموسيقى، ويكتشف العلاقة بين الإيقاع والكسور والأعداد.",
        "En español, explique cómo cultivar tomates, observar insectos y registrar la lluvia. Incluya vocabulario de cocina, naturaleza y estaciones.",
        "Auf Deutsch, beschreiben Sie einen Bahnhof, eine Werkstatt, einen Chor und eine Reise durch die Berge mit Dialogen.",
        "Continue a science fiction story about a pilot, a robot botanist, an underground ocean and a forgotten radio signal. Include vivid dialogue and sensory details.",
        "Explain compiler intermediate representations, SSA, register allocation, graph coloring, branch prediction, cache locality and vector instructions with pseudocode.",
        "Discuss linear algebra notation for eigenvalues, orthogonal matrices, determinants, tensors, Fourier series, partial differential equations and probability distributions.",
        "Write a fictional cooking diary using coriander, cumin, saffron, lentils, rice, tomatoes, caramel, sourdough, fermentation and the aroma of a bakery.",
        "Describe a jazz rehearsal with trumpet, cello, percussion, counterpoint, syncopation, harmonic progression, tempo changes and conversations between musicians.",
        "Write a field notebook about volcanic rocks, glaciers, ocean currents, migratory birds, fungi, moss, pollination and a meteor shower.",
    ]):
        request(f"routing-domain-{i}", chat(text + " Give a detailed explanation with examples. " + text), output_tokens=1)
    request("routing-generation", chat("Tell a detailed story about a multilingual engineer building a robot garden on an ocean planet. Include dialogue, mathematics and descriptions of plants."), output_tokens=64, teacher_forced=False)
(args.output / "quality-scope.json").write_text(json.dumps({"qualification": "Bounded synthetic regression and selected-expert oracle comparison; not held-out quality, whole-checkpoint quantizer equivalence, or independent attention/cache validation", "activation_only": args.activation_only, "routing_diagnostic": args.routing_diagnostic, "fixed_prefix_cases": (5 if args.activation_only else 9) + (12 if args.routing_diagnostic else 0), "full_vocabulary_rows": "Last eight rows of each logits call, with original call size retained", "kv_dtype": "bfloat16", "concurrency": 1, "model_code_trust": False}, indent=2))
