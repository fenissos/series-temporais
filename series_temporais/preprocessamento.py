"""Limpeza, construção de alvos e definição de partições."""

import numpy as np
import pandas as pd

CORTES_METROPT = (pd.Timestamp("2020-06-05"), pd.Timestamp("2020-07-01"))
FALHAS_METROPT = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


def linhas_validas(df, sensores):
    validas = np.isfinite(df[sensores].to_numpy(dtype=float)).all(axis=1)
    return df.loc[validas].copy()


def particao_sujeito(sujeito):
    if sujeito <= 7:
        return "treino"
    return "validacao" if sujeito == 8 else "teste"


def dividir_ai4i(df):
    df = df.sort_values("UDI").reset_index(drop=True)
    treino, validacao = int(.70 * len(df)), int(.85 * len(df))
    return df.iloc[:treino], df.iloc[treino:validacao], df.iloc[validacao:]


def rotular_metropt(df):
    falha = np.zeros(len(df), dtype="int8")
    for inicio, fim in FALHAS_METROPT:
        falha[df["timestamp"].between(pd.Timestamp(inicio), pd.Timestamp(fim))] = 1
    return df.assign(falha=falha)


def dividir_metropt(df):
    tempo = df["timestamp"]
    primeiro, segundo = CORTES_METROPT
    return df[tempo < primeiro], df[(tempo >= primeiro) & (tempo < segundo)], df[tempo >= segundo]
