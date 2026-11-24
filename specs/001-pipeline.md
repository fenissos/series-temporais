# SDD 001 — Preparação de dados da aula 1

## Especificação

O pipeline recebe os arquivos ZIP da UCI usados na aula e gera uma tabela supervisionada por base. Cada linha inclui a partição, o alvo e as características. As tabelas preservam identificadores temporais para auditoria. A execução informa quais colunas podem entrar no modelo.

| Base | Unidade padrão | Partições | Alvo |
|---|---|---|---|
| MHEALTH | 128 amostras a 50 Hz, passo 64 | Treino sujeitos 1–7, validação 8, teste 9–10 | Atividade constante em toda a janela |
| AI4I | Uma operação, em ordem de UDI | Treino, validação e teste cronológicos de 70%, 15% e 15% | Falha da operação |
| MetroPT-3 | 60 leituras, aproximadamente 10 min, passo 30 | Cortes da aula em 2020-06-05 e 2020-07-01 | Pelo menos uma leitura em um intervalo de falha da aula |

O MHEALTH usa os 23 canais, com descritores temporais e frequência dominante nos canais inerciais. No AI4I, preservam-se os cinco sensores e o tipo codificado, com diferencial térmico, potência mecânica e interação desgaste × torque. O MetroPT-3 usa estatísticas dos sete sensores analógicos e fração ativa e transições dos oito canais digitais. As características são calculadas por exemplo e dispensam ajuste global.

## Critérios de aceitação

- Configurar tamanho em amostras e sobreposição como fração, com passo igual ao tamanho menos a sobreposição arredondada para baixo em amostras. Manter somente janelas completas.
- Impedir janelas entre partições, sujeitos, atividades ou lacunas. No MetroPT-3, separar trechos quando o intervalo sair da faixa de 8 a 12 segundos. Remover linhas com sensores ausentes ou não finitos, preservando a lacuna correspondente.
- Excluir os cinco indicadores de modo de falha e identificadores dos preditores do AI4I. Usar intervalos de falha apenas para construir o alvo do MetroPT-3.
- Exportar características, lista de preditores, distribuição de classes, estatísticas descritivas por partição e prévia de dados. Permitir consultar os resultados no notebook.

## Plano

1. Modularizar conectores e reaproveitá-los na aula.
2. Preparar partições, construir janelas e extrair características.
3. Exportar os resultados, testar as fronteiras e executar com as três bases.
