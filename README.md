# Séries temporais para machine learning

O projeto transforma as três bases da [aula 1](aula_01_fundamentos.ipynb) em tabelas supervisionadas. A [especificação SDD](specs/001-pipeline.md) registra as regras de particionamento, os alvos e os critérios de aceitação.

## Gerar os dados

Na raiz do projeto, instale as dependências e gere os resultados.

```powershell
python -m pip install -r requirements.txt
python -m series_temporais.pipeline
```

O pipeline baixa os ZIPs da UCI para `dados/` na primeira execução e reutiliza os arquivos nas seguintes. Use `--dados` para apontar para um cache existente e `--saida` para escolher outro destino. O notebook usa `DADOS_CURSO` como caminho opcional para o mesmo cache.

Para alterar os parâmetros, use o tamanho em amostras e a sobreposição como fração. Por exemplo, 256 amostras com sobreposição de 25% avançam 192 amostras a cada janela.

```powershell
python -m series_temporais.pipeline --base mhealth --mhealth-tamanho 256 --mhealth-sobreposicao 0.25
python -m series_temporais.pipeline --base metropt --metropt-tamanho 60 --metropt-sobreposicao 0.5
```

## Organização

| Arquivo | Responsabilidade |
|---|---|
| `series_temporais/connectors.py` | Download, cache e leitura dos arquivos da UCI |
| `series_temporais/preprocessamento.py` | Validação de sensores, rótulos e cortes |
| `series_temporais/janelas.py` | Configuração e geração de janelas completas |
| `series_temporais/caracteristicas.py` | Cálculo dos preditores |
| `series_temporais/pipeline.py` | Execução por base e exportação dos relatórios |

## Consultar os resultados

A última seção do notebook exibe uma prévia, as classes e uma amostra das estatísticas. Em `resultados/<base>/` ficam:

- `dataset.csv` com todos os exemplos.
- `previa.csv` com as primeiras dez linhas.
- `classes.csv` com contagens e percentuais por partição, incluindo classes com zero exemplos.
- `estatisticas.csv` com contagem, média, desvio-padrão, mínimo, quartis e máximo para cada preditor por partição.
- `resumo.json` com dimensões, configuração de janelas e nomes dos preditores. Ao treinar um modelo, use essa lista para montar `X` e a coluna `alvo` para montar `y`.

No MetroPT-3, a classe positiva indica sobreposição com um evento documentado. Ela serve para diagnóstico durante o evento. Os campos `validacao` e `teste` correspondem aos testes público e privado da aula.
