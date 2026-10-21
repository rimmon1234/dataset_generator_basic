import json
import re
from typing import Optional, Dict, Any

def strip_thinking(text: str) -> str:
    """
    Strips internal reasoning tokens/tags (<think>...</think>) from thinking models.
    """
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

def strip_conversational_filler(text: str) -> str:
    """
    Removes chatty opening preambles like 'Here is the implementation...',
    'Sure! Below is...', 'Certainly!', etc., so responses start directly with code or substance.
    """
    cleaned = re.sub(
        r"^(?:Here (?:is|are|’s|'s)|Sure!?|Certainly!?|Below is|The following is)[^:\n]*:?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )
    return cleaned.strip()

def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses a JSON object from raw LLM output,
    handling nested code blocks, unescaped control characters, and markdown wrappers.
    """
    text = strip_thinking(raw_text.strip())
    
    # Locate the outermost JSON object bounds
    start = text.find("{")
    end = text.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        return None
        
    candidate = text[start:end + 1].strip()
    data = None
    
    # 1. Try strict=False parsing (handles multiline strings and unescaped newlines)
    try:
        parsed = json.loads(candidate, strict=False)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        pass

    # 2. Heuristic repair: check if final closing quote before '}' was omitted
    if not data and not candidate.rstrip("}\n\r\t ").endswith('"'):
        last_brace = candidate.rfind("}")
        repaired = candidate[:last_brace].rstrip() + '"\n}'
        try:
            parsed = json.loads(repaired, strict=False)
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            pass

    if data:
        # Strip conversational filler from the output response
        if isinstance(data.get("output"), str):
            data["output"] = strip_conversational_filler(data["output"])
        return data

    return None

def validate_alpaca_sample(sample: Dict[str, Any]) -> bool:
    """
    Validates that the dictionary adheres to the Alpaca schema:
    - Must contain 'instruction' and 'output' (non-empty strings)
    - 'input' must be present (can be empty string)
    """
    if not isinstance(sample, dict):
        return False
        
    instruction = sample.get("instruction")
    input_text = sample.get("input")
    output = sample.get("output")
    
    if not isinstance(instruction, str) or len(instruction.strip()) < 10:
        return False
        
    if not isinstance(output, str) or len(output.strip()) < 20:
        return False
        
    if input_text is None or not isinstance(input_text, str):
        sample["input"] = ""
        
    return True
