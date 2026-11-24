"""Geração das tabelas de machine learning e seus relatórios."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import caracteristicas as car
from . import connectors as con
from . import preprocessamento as prep
from .janelas import Janela

PARTICOES = ("treino", "validacao", "teste")


def mhealth(dados, janela):
    tabelas = []
    for sujeito in range(1, 11):
        df = con.ler_mhealth(dados, sujeito)
        df = prep.linhas_validas(df[df["atividade_id"].isin(con.ATIVIDADES)], con.CANAIS_MHEALTH)
        amostras = df["amostra"].to_numpy()
        alvos = df["atividade_id"].to_numpy()
        inicio = janela.inicios(np.r_[True, (np.diff(amostras) != 1) | (np.diff(alvos) != 0)])
        valores = janela.valores(df[con.CANAIS_MHEALTH].to_numpy(), inicio)
        atributos = car.sinais(valores, con.CANAIS_MHEALTH, fs=con.FS_MHEALTH,
                               inerciais=[c for c in con.CANAIS_MHEALTH if not c.startswith("ecg")])
        meta = pd.DataFrame({
            "particao": prep.particao_sujeito(sujeito), "sujeito": sujeito,
            "amostra_inicio": amostras[inicio], "amostra_fim": amostras[inicio + janela.tamanho - 1],
            "alvo": alvos[inicio],
        })
        tabelas.append(pd.concat([meta, atributos], axis=1))
    return pd.concat(tabelas, ignore_index=True), atributos.columns.tolist()


def ai4i(dados):
    tabelas = []
    for nome, bruto in zip(PARTICOES, prep.dividir_ai4i(con.ler_ai4i(dados))):
        df = prep.linhas_validas(bruto[bruto["Type"].isin(["L", "M", "H"])], con.SENSORES_AI4I)
        atributos = car.operacionais(df, con.SENSORES_AI4I)
        meta = pd.DataFrame({"particao": nome, "UDI": df["UDI"].to_numpy(),
                             "alvo": df["Machine failure"].to_numpy()})
        tabelas.append(pd.concat([meta, atributos], axis=1))
    return pd.concat(tabelas, ignore_index=True), atributos.columns.tolist()


def metropt(dados, janela):
    canais = con.SENSORES_METROPT + con.DIGITAIS_METROPT
    bruto = con.ler_metropt(dados).dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    bruto = prep.rotular_metropt(bruto.drop_duplicates("timestamp"))
    tabelas = []
    for nome, parte in zip(PARTICOES, prep.dividir_metropt(bruto)):
        parte = parte.copy()
        parte["_posicao"] = np.arange(len(parte))
        df = prep.linhas_validas(parte, canais)
        dt = df["timestamp"].diff().dt.total_seconds()
        quebras = ~dt.between(8, 12) | df["_posicao"].diff().ne(1)
        inicios = janela.inicios(quebras.to_numpy())
        analogicos = df[con.SENSORES_METROPT].to_numpy()
        digitais = df[con.DIGITAIS_METROPT].to_numpy()
        alvos = df["falha"].to_numpy()
        tempo = df["timestamp"].to_numpy()
        # Limita o volume de janelas materializadas ao ajustar o tamanho ou a sobreposição.
        for inicio in np.array_split(inicios, max(1, (len(inicios) + 1023) // 1024)):
            analog = car.sinais(janela.valores(analogicos, inicio), con.SENSORES_METROPT)
            digital = car.digitais(janela.valores(digitais, inicio), con.DIGITAIS_METROPT)
            meta = pd.DataFrame({
                "particao": nome, "inicio": tempo[inicio], "fim": tempo[inicio + janela.tamanho - 1],
                "alvo": janela.valores(alvos, inicio).max(axis=1),
            })
            tabelas.append(pd.concat([meta, analog, digital], axis=1))
    return pd.concat(tabelas, ignore_index=True), [*analog.columns, *digital.columns]


def exportar(nome, tabela, preditores, saida, janela=None):
    pasta = saida / nome
    pasta.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(pasta / "dataset.csv", index=False)
    tabela.head(10).to_csv(pasta / "previa.csv", index=False)
    por_classe = tabela.groupby(["particao", "alvo"]).size().unstack(fill_value=0)
    classes = list(con.ATIVIDADES) if nome == "mhealth" else [0, 1]
    por_classe = por_classe.reindex(index=PARTICOES, columns=classes, fill_value=0)
    distribuicao = por_classe.rename_axis(columns="classe").stack().rename("exemplos").reset_index()
    totais = distribuicao["particao"].map(por_classe.sum(axis=1))
    distribuicao["percentual"] = 100 * distribuicao["exemplos"].div(totais.where(totais > 0))
    if nome == "mhealth":
        distribuicao["atividade"] = distribuicao["classe"].map(con.ATIVIDADES)
    distribuicao.to_csv(pasta / "classes.csv", index=False)
    estatisticas = pd.concat({
        particao: tabela.loc[tabela["particao"].eq(particao), preditores].describe().T
        for particao in PARTICOES
    }, names=["particao", "caracteristica"])
    estatisticas.to_csv(pasta / "estatisticas.csv")
    resumo = {
        "linhas": len(tabela), "colunas": len(tabela.columns), "preditores": preditores,
        "linhas_por_particao": por_classe.sum(axis=1).to_dict(),
    }
    if janela is not None:
        resumo["janela"] = {
            "tamanho": janela.tamanho, "sobreposicao": janela.sobreposicao, "passo": janela.passo,
        }
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"{nome}: {len(tabela):,} exemplos, {len(preditores)} características")
    print(por_classe.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=("todas", *con.FONTES), default="todas")
    parser.add_argument("--dados", type=Path, default=Path("dados"))
    parser.add_argument("--saida", type=Path, default=Path("resultados"))
    for nome, tamanho in (("mhealth", 128), ("metropt", 60)):
        parser.add_argument(f"--{nome}-tamanho", type=int, default=tamanho)
        parser.add_argument(f"--{nome}-sobreposicao", type=float, default=0.5)
    args = parser.parse_args()
    for nome, gerar in (("ai4i", ai4i), ("mhealth", mhealth), ("metropt", metropt)):
        if args.base not in ("todas", nome):
            continue
        janela = None
        if nome != "ai4i":
            try:
                janela = Janela(getattr(args, f"{nome}_tamanho"), getattr(args, f"{nome}_sobreposicao"))
            except ValueError as erro:
                parser.error(str(erro))
        tabela, preditores = gerar(args.dados) if janela is None else gerar(args.dados, janela)
        exportar(nome, tabela, preditores, args.saida, janela)


if __name__ == "__main__":
    main()
