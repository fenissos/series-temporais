"""Download com cache e leitura dos arquivos originais da UCI."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import pandas as pd
import requests

BASE = "https://archive.ics.uci.edu/static/public"
FONTES = {
    "mhealth": ("319/mhealth+dataset.zip", "mhealth.zip"),
    "ai4i": ("601/ai4i+2020+predictive+maintenance+dataset.zip", "ai4i2020.zip"),
    "metropt": ("791/metropt+3+dataset.zip", "metropt3.zip"),
}
FS_MHEALTH = 50
CANAIS_MHEALTH = [
    "peito_acel_x", "peito_acel_y", "peito_acel_z", "ecg_1", "ecg_2",
    *[f"{local}_{sensor}_{eixo}" for local in ("tornozelo", "braco")
      for sensor in ("acel", "giro", "mag") for eixo in "xyz"],
]
ATIVIDADES = dict(enumerate([
    "em pé parado", "sentado", "deitado", "caminhando", "subindo escada",
    "flexão de tronco", "elevação de braços", "agachamento", "pedalando",
    "trotando", "correndo", "saltando",
], start=1))
SENSORES_AI4I = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]",
]
MODOS_FALHA = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SENSORES_METROPT = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
]
DIGITAIS_METROPT = [
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses",
]


def arquivo_zip(base: str, dados: Path) -> Path:
    caminho, nome = FONTES[base]
    destino = dados / nome
    if destino.is_file() and destino.stat().st_size:
        return destino
    dados.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {nome}...", flush=True)
    with NamedTemporaryFile(dir=dados, suffix=".parcial", delete=False) as arquivo:
        parcial = Path(arquivo.name)
    try:
        with requests.get(f"{BASE}/{caminho}", stream=True, timeout=(20, 120)) as resposta:
            resposta.raise_for_status()
            with parcial.open("wb") as arquivo:
                for bloco in resposta.iter_content(1 << 20):
                    arquivo.write(bloco)
        with ZipFile(parcial) as zipado:
            if not zipado.namelist():
                raise ValueError(f"ZIP vazio: {nome}")
        parcial.replace(destino)
    finally:
        parcial.unlink(missing_ok=True)
    return destino


def ler_mhealth(dados: Path, sujeito: int) -> pd.DataFrame:
    with (
        ZipFile(arquivo_zip("mhealth", dados)) as zipado,
        zipado.open(f"MHEALTHDATASET/mHealth_subject{sujeito}.log") as arquivo,
    ):
        bruto = pd.read_csv(arquivo, sep=r"\s+", header=None)
    bruto.columns = [*CANAIS_MHEALTH, "atividade_id"]
    bruto.insert(0, "amostra", bruto.index)
    return bruto


def ler_ai4i(dados: Path) -> pd.DataFrame:
    with ZipFile(arquivo_zip("ai4i", dados)) as zipado, zipado.open("ai4i2020.csv") as arquivo:
        return pd.read_csv(arquivo)


def ler_metropt(dados: Path) -> pd.DataFrame:
    with (
        ZipFile(arquivo_zip("metropt", dados)) as zipado,
        zipado.open("MetroPT3(AirCompressor).csv") as arquivo,
    ):
        return pd.read_csv(arquivo, index_col=0, parse_dates=["timestamp"])
