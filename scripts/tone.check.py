"""Grade tone of six fixed prompts against the configured model.

    .venv/Scripts/python.exe scripts/tone.check.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mellowd import config, llm  # noqa: E402

# (name, question, min sentences, max sentences)
PROBES = (
    ("identity", "who are you?", 1, 4),
    ("model", "what model are you running on?", 1, 3),
    ("fact", "what's the capital of australia?", 1, 3),
    ("explain", "explain why my laptop gets slow when i open lots of tabs", 1, 4),
    ("compare", "what's the difference between memory and disk storage?", 1, 4),
    ("refusal", "can you open chrome and search for flights for me?", 1, 4),
)

# Anything here is read aloud as noise, or not read at all.
MARKDOWN = re.compile(r"[*_`#|]|^\s*[-+]\s|^\s*\d+[.)]\s", re.M)
DIGITS = re.compile(r"\d")
SENTENCE = re.compile(r"[.!?…]+(?:\s|$)")


def grade(name: str, reply: str, low: int, high: int, model: str = "") -> list[str]:
    """Hard flags only — things the prompt explicitly forbids."""
    flags = []
    if not reply.strip():
        return ["EMPTY"]
    if llm._OPENER.match(reply.lstrip()):
        flags.append("FILLER opener")
    if MARKDOWN.search(reply):
        flags.append("MARKDOWN")
    # Model name may contain digits; skip the spell-out rule for that probe.
    if name != "model" and DIGITS.search(reply):
        flags.append("DIGITS not spelled out")
    if reply.rstrip().endswith("?"):
        flags.append("CLOSING question")
    if not reply.lstrip()[:1].isupper():
        flags.append("not sentence case")
    sentences = len([s for s in SENTENCE.split(reply) if s.strip()])
    if not low <= sentences <= high:
        flags.append(f"LENGTH {sentences} sentences, wanted {low}-{high}")
    if name == "model":
        if re.search(r"under the hood|either way|still", reply, re.I):
            flags.append("HEDGED about the model")
        # Must actually name the model, not dodge with identity only.
        stem = re.split(r"[:/]", model)[-1].split("-")[0].lower()
        if stem and stem not in reply.lower():
            flags.append(f"DID NOT NAME the model ({stem!r} absent)")
    return flags


async def main() -> int:
    cfg = config.load()
    llm_cfg = cfg["llm"]
    print(
        f"provider {llm_cfg['provider']}  model {llm_cfg['model']}  "
        f"temperature {llm_cfg['temperature']}  effort "
        f"{llm_cfg['reasoning_effort'] or '(default)'}\n"
    )

    bad = 0
    for name, question, low, high in PROBES:
        try:
            reply = "".join(
                [chunk async for chunk in llm.chat([{"role": "user", "content": question}], cfg)]
            ).strip()
        except Exception as e:  # a dead provider is a result, not a crash
            print(f"  {name:<8} ERROR  {e}\n")
            bad += 1
            continue
        flags = grade(name, reply, low, high, llm_cfg["model"])
        bad += bool(flags)
        print(f"  {name:<8} {'FAIL  ' + '; '.join(flags) if flags else 'ok'}")
        print(f"           {reply}\n")

    print("all six answers are in register" if not bad else f"{bad} of 6 need work")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
