import re
from typing import Optional


JSON_SINGLE_BLOCK = re.compile(r'(?s)([\{\[].*[\}\]])')

def extract_single_json_block(text: str) -> Optional[str]:
    m = JSON_SINGLE_BLOCK.search(text)
    if not m:
        return None
    return m.group(1)