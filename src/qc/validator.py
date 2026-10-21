"""
Comprehensive Quality Control (QC) & Validation Engine for Synthetic Datasets.
Rule: Valid JSON does not mean valid training data.
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set

REFUSAL_PHRASES = [
    "as an ai",
    "as a large language model",
    "i cannot fulfill",
    "i am sorry, but",
    "i'm sorry, but",
    "i cannot assist",
    "i don't have access to",
    "i cannot provide",
    "my apologies, but",
    "as an artificial intelligence"
]

VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}

def extract_python_code_blocks(text: str) -> List[str]:
    """Extracts all code from ```python ... ``` blocks."""
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]

def check_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validates that a python code string parses into a valid Abstract Syntax Tree (AST).
    Protects against hallucinated, incomplete, or broken syntax.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError on line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Code parse error: {str(e)}"

def check_repetition(text: str, max_repeated_lines: int = 2) -> bool:
    """
    Detects degenerative looping where the model repeats the exact same line.
    """
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 15]
    if not lines:
        return True
    
    seen = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1
        if seen[line] > max_repeated_lines:
            return False
    return True

def calculate_token_jaccard(s1: str, s2: str) -> float:
    """Calculates word-level Jaccard similarity to detect near-duplicate instructions."""
    tokens1 = set(re.findall(r"\w+", s1.lower()))
    tokens2 = set(re.findall(r"\w+", s2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

class QualityControlValidator:
    def __init__(self, rejection_log_path: str = "data/rejections.jsonl"):
        self.rejection_log_path = Path(rejection_log_path)
        self.seen_exact_keys: Set[Tuple[str, str]] = set()
        self.seen_instructions: List[str] = []

    def load_existing_dataset(self, dataset_path: str):
        """Pre-populates deduplication cache from an existing dataset."""
        path = Path(dataset_path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    inst = item.get("instruction", "").strip().lower()
                    inp = item.get("input", "").strip().lower()
                    self.seen_exact_keys.add((inst, inp))
                    if inst:
                        self.seen_instructions.append(inst)
                except Exception:
                    continue

    def log_rejection(self, raw_output: str, reason: str, check_type: str, topic: str = "Unknown"):
        """Logs rejected samples for failure diagnosis."""
        self.rejection_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "check_type": check_type,
            "reason": reason,
            "raw_output": raw_output[:1000]
        }
        with open(self.rejection_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def validate(
        self,
        sample: Dict[str, Any],
        raw_output: str,
        topic: str = "Unknown",
        similarity_threshold: float = 0.85
    ) -> Tuple[bool, Optional[str]]:
        """
        Runs comprehensive multi-tier quality checks:
        1. Schema & Type Validation
        2. Content Depth & Length Heuristics
        3. Refusal / Canned Response Check
        4. Repetition / Degeneration Check
        5. Exact & Near-Duplicate Check
        6. Task-Specific Code Syntax Check (ast.parse)
        """
        if not isinstance(sample, dict):
            reason = "Sample is not a valid JSON dictionary"
            self.log_rejection(raw_output, reason, "schema_validation", topic)
            return False, reason

        # --- 1. Schema & Required Fields ---
        required_fields = ["instruction", "output"]
        for field in required_fields:
            if field not in sample or not isinstance(sample[field], str):
                reason = f"Missing or non-string required field: '{field}'"
                self.log_rejection(raw_output, reason, "schema_validation", topic)
                return False, reason

        instruction = sample["instruction"].strip()
        input_text = sample.get("input", "").strip() if isinstance(sample.get("input"), str) else ""
        output = sample["output"].strip()
        sample["input"] = input_text

        difficulty = sample.get("difficulty", "intermediate").lower().strip()
        if difficulty not in VALID_DIFFICULTIES:
            difficulty = "intermediate"
        sample["difficulty"] = difficulty

        if "category" not in sample or not isinstance(sample["category"], str) or len(sample["category"].strip()) < 3:
            sample["category"] = topic

        # --- 2. Length & Substance Heuristics ---
        if len(instruction.split()) < 5:
            reason = f"Instruction is too short/trivial ({len(instruction.split())} words)"
            self.log_rejection(raw_output, reason, "length_check", topic)
            return False, reason

        if len(output.split()) < 15:
            reason = f"Output is too short or stubbed ({len(output.split())} words)"
            self.log_rejection(raw_output, reason, "length_check", topic)
            return False, reason

        # --- 3. Refusal / Non-compliance Detection ---
        output_lower = output.lower()
        for phrase in REFUSAL_PHRASES:
            if phrase in output_lower[:150]:
                reason = f"Model refusal detected: contains '{phrase}'"
                self.log_rejection(raw_output, reason, "refusal_check", topic)
                return False, reason

        # --- 4. Repetition / Degeneration Check ---
        if not check_repetition(output):
            reason = "Degenerative repeating loops detected in output text"
            self.log_rejection(raw_output, reason, "repetition_check", topic)
            return False, reason

        # --- 5. Deduplication (Exact + Fuzzy/Semantic) ---
        exact_key = (instruction.lower(), input_text.lower())
        if exact_key in self.seen_exact_keys:
            reason = "Exact duplicate of an existing (instruction, input) pair"
            self.log_rejection(raw_output, reason, "exact_deduplication", topic)
            return False, reason

        for prev_inst in self.seen_instructions[-100:]:
            similarity = calculate_token_jaccard(instruction, prev_inst)
            if similarity >= similarity_threshold:
                reason = f"Near-duplicate instruction detected ({similarity:.0%} token similarity with existing sample)"
                self.log_rejection(raw_output, reason, "fuzzy_deduplication", topic)
                return False, reason

        # --- 6. Task-Specific Validation (Python Code AST check) ---
        code_blocks = extract_python_code_blocks(output)
        if input_text:
            code_blocks.extend(extract_python_code_blocks(input_text))

        for code in code_blocks:
            is_valid, error_msg = check_python_syntax(code)
            if not is_valid:
                reason = f"Task-specific validation failed: {error_msg}"
                self.log_rejection(raw_output, reason, "code_syntax_validation", topic)
                return False, reason

        # Passed all checks!
        self.seen_exact_keys.add(exact_key)
        self.seen_instructions.append(instruction.lower())
        return True, None
