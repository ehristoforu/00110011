import re
from typing import List, Optional

DEFAULT_VISION_PATTERNS = [
    r"visual\.",
    r"vision_tower\.",
    r"vision_model\.",
    r"image_newline",
    r"multi_modal_projector\.",
    r"mm_projector\.",
    r"vlm\.",
    r"clip\.",
    r"vit\.",
]

class TensorFilter:
    def __init__(
        self,
        ignore_vision: bool = True,
        custom_ignore_patterns: Optional[List[str]] = None,
        custom_keep_patterns: Optional[List[str]] = None,
        skip_patterns: Optional[List[str]] = None,
    ):
        self.ignore_vision = ignore_vision
        patterns = []
        if ignore_vision:
            patterns.extend(DEFAULT_VISION_PATTERNS)
        if custom_ignore_patterns:
            patterns.extend(custom_ignore_patterns)
        self.ignore_regex = [re.compile(p, re.IGNORECASE) for p in patterns] if patterns else []
        self.keep_regex = [re.compile(p, re.IGNORECASE) for p in custom_keep_patterns] if custom_keep_patterns else []
        self.skip_regex = [re.compile(p, re.IGNORECASE) for p in skip_patterns] if skip_patterns else []

    def should_ignore(self, tensor_name: str) -> bool:
        if self.keep_regex and any(r.search(tensor_name) for r in self.keep_regex):
            return False
        return any(r.search(tensor_name) for r in self.ignore_regex)

    def should_skip_merge(self, tensor_name: str) -> bool:
        return any(r.search(tensor_name) for r in self.skip_regex)
