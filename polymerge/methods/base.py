from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch

class BaseMergeMethod(ABC):
    @abstractmethod
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        pass
