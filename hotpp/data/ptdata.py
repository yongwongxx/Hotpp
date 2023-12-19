from .base import AtomsData
from typing import List, Optional
import torch
import os


class PtData(AtomsData):
    def __init__(self,
                 name   : str,
                 device : str="cpu",
                 ) -> None:
        super().__init__()
        self.data, self.slices = torch.load(name, map_location=device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"
