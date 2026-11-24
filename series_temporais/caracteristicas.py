"""Descritores temporais, espectrais e operacionais."""

import numpy as np
import pandas as pd


def sinais(valores, canais, *, inerciais=(), fs=None):
    minimo = valores.min(axis=1)
    maximo = valores.max(axis=1)
    medidas = {
        "media": valores.mean(axis=1),
        "desvio": valores.std(axis=1),
        "min": minimo,
        "max": maximo,
        "amplitude": maximo - minimo,
        "rms": np.sqrt(np.mean(valores ** 2, axis=1)),
        "delta": valores[:, -1] - valores[:, 0],
    }
    resultado = {
        f"{canal}__{nome}": medida[:, indice]
        for indice, canal in enumerate(canais)
        for nome, medida in medidas.items()
    }
    if inerciais:
        indices = [canais.index(canal) for canal in inerciais]
        sinal = valores[:, :, indices]
        centrado = sinal - sinal.mean(axis=1, keepdims=True)
        taper = np.hanning(sinal.shape[1]) if sinal.shape[1] > 2 else np.ones(2)
        espectro = np.abs(np.fft.rfft(centrado * taper[None, :, None], axis=1))[:, 1:]
        frequencias = np.fft.rfftfreq(sinal.shape[1], d=1 / fs)[1:]
        picos = np.where(np.ptp(sinal, axis=1) > 0, frequencias[espectro.argmax(axis=1)], 0)
        resultado.update({f"{canal}__pico_hz": picos[:, i] for i, canal in enumerate(inerciais)})
    return pd.DataFrame(resultado)


def digitais(valores, canais):
    fracoes = valores.mean(axis=1)
    mudancas = np.count_nonzero(np.diff(valores, axis=1), axis=1)
    return pd.DataFrame({
        f"{canal}__{nome}": medida[:, indice]
        for indice, canal in enumerate(canais)
        for nome, medida in (("fracao_ativa", fracoes), ("transicoes", mudancas))
    })


def operacionais(df, sensores):
    dados = df[sensores].reset_index(drop=True)
    tipos = pd.get_dummies(pd.Categorical(df["Type"], categories=["L", "M", "H"]),
                           prefix="tipo", dtype=int)
    torque = dados["Torque [Nm]"]
    return pd.concat([dados.assign(
        diferencial_termico_k=dados["Process temperature [K]"] - dados["Air temperature [K]"],
        potencia_w=torque * dados["Rotational speed [rpm]"] * (2 * np.pi / 60),
        desgaste_x_torque=dados["Tool wear [min]"] * torque,
    ), tipos], axis=1)
