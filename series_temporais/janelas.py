"""Enjanelamento vetorizado dentro de segmentos contínuos."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class Janela:
    tamanho: int
    sobreposicao: float = 0.5

    def __post_init__(self):
        if self.tamanho < 2 or not 0 <= self.sobreposicao < 1:
            raise ValueError("Use tamanho >= 2 e sobreposição no intervalo [0, 1).")

    @property
    def passo(self):
        return self.tamanho - int(self.tamanho * self.sobreposicao)

    def inicios(self, quebras):
        limites = np.r_[0, np.flatnonzero(quebras[1:]) + 1, len(quebras)]
        return np.concatenate([
            np.arange(inicio, fim - self.tamanho + 1, self.passo, dtype=int)
            for inicio, fim in pairwise(limites)
        ])

    def valores(self, matriz, inicios):
        indices = inicios[:, None] + np.arange(self.tamanho)
        return matriz[indices]
