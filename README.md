# Aplicacao de K-Means e Regras de Associacao na Analise de Queimadas no Brasil

## Autores

Thaigo Plancke  
Mateus Linardi  
Kaue Lima

Projeto desenvolvido para a disciplina de Inteligencia Artificial e Mineracao de Dados.

## Resumo

Este trabalho apresenta uma analise de focos de queimadas no Brasil utilizando tecnicas de mineracao de dados e aprendizado de maquina. Foram utilizados dados publicos da Base dos Dados, extraidos da tabela de microdados de queimadas do INPE, referentes ao periodo de 2023 a 2025. Os algoritmos aplicados incluem K-Means para clusterizacao espacial e Apriori para mineracao de regras de associacao entre variaveis climaticas, biomas, estacoes do ano e risco de fogo. Os resultados mostraram a formacao de agrupamentos geograficos de focos de queimadas e regras fortes associando baixa precipitacao, seca alta e primavera ao risco alto de fogo. O estudo contribui para a compreensao de padroes ambientais relacionados a queimadas e demonstra como tecnicas de IA podem apoiar analises de dados climaticos e territoriais.

## Palavras-chave

Mineracao de Dados; Machine Learning; Clusterizacao; Regras de Associacao; Queimadas; Analise Ambiental.

## 1. Introducao

As queimadas representam um problema ambiental relevante no Brasil, afetando biomas, municipios, qualidade do ar, biodiversidade e atividades humanas. A recorrencia desses eventos esta relacionada a fatores climaticos, territoriais e sazonais, como periodos prolongados sem chuva, baixa precipitacao, estacao do ano e caracteristicas de cada bioma.

Esse problema e importante porque a identificacao de padroes em grandes volumes de dados pode auxiliar na compreensao das regioes mais afetadas e das condicoes mais associadas ao risco de fogo. Trabalhos de analise ambiental frequentemente utilizam estatisticas descritivas, mapas, modelos de agrupamento e tecnicas de mineracao de regras para interpretar fenomenos complexos.

Entretanto, analisar apenas a quantidade de focos por localidade pode limitar a interpretacao do problema. Por isso, este projeto combina clusterizacao espacial com regras de associacao, buscando relacionar a distribuicao geografica das queimadas com variaveis climaticas e ambientais.

O objetivo deste trabalho e aplicar tecnicas de mineracao de dados para analisar focos de queimadas no Brasil entre 2023 e 2025, identificando agrupamentos espaciais e padroes frequentes associados ao risco alto de fogo.

## 2. Metodologia

### 2.1 Base de Dados

Os dados foram obtidos da Base dos Dados, a partir da tabela `br_inpe_queimadas.microdados`, que disponibiliza registros de focos de queimadas do INPE. A consulta SQL utilizada no projeto seleciona dados dos anos de 2023, 2024 e 2025, enriquecidos com nomes de estados e municipios por meio de tabelas de diretorios brasileiros.

A base local em formato Parquet possui 12.903.132 registros e 14 atributos antes do processamento. Apos a limpeza e normalizacao, foram mantidos 12.247.168 registros e 17 atributos.

Distribuicao dos registros limpos por ano:

| Ano | Registros |
| --- | ---: |
| 2023 | 4.326.247 |
| 2024 | 7.850.298 |
| 2025 | 70.623 |

Variaveis utilizadas no projeto:

| Variavel | Descricao |
| --- | --- |
| `ano`, `mes` | Periodo do registro |
| `bioma` | Bioma associado ao foco de queimada |
| `sigla_uf`, `sigla_uf_nome` | Estado do registro |
| `id_municipio`, `id_municipio_nome` | Municipio do registro |
| `latitude`, `longitude` | Localizacao geografica |
| `satelite` | Satelite responsavel pela deteccao |
| `dias_sem_chuva` | Quantidade de dias sem chuva |
| `precipitacao` | Nivel de precipitacao |
| `risco_fogo` | Indicador de risco de fogo |
| `potencia_radiativa_fogo` | Potencia radiativa do foco de fogo |

### 2.2 Pre-processamento

O pre-processamento foi realizado no arquivo `pipeline.py` e incluiu as seguintes etapas:

1. Leitura do cache local `queimadas_2023_2025.parquet` ou download via BigQuery caso o arquivo nao exista.
2. Conversao das colunas numericas para tipo `float32`.
3. Remocao de valores ausentes, infinitos e registros inconsistentes.
4. Filtragem de coordenadas validas para o territorio brasileiro.
5. Remocao de outliers nas variaveis `precipitacao`, `risco_fogo` e `dias_sem_chuva`, usando os percentis 1% e 99%.
6. Normalizacao Min-Max das variaveis `risco_fogo`, `dias_sem_chuva` e `precipitacao`.

As variaveis normalizadas foram usadas no processo de clusterizacao para reduzir o impacto de escalas diferentes entre os atributos.

### 2.3 Algoritmos Utilizados

#### 2.3.1 Clusterizacao com K-Means

O algoritmo K-Means foi utilizado para identificar grupos semelhantes de focos de queimadas. As variaveis usadas na clusterizacao foram:

- `latitude`
- `longitude`
- `risco_fogo_normalizado`
- `dias_sem_chuva_normalizado`
- `precipitacao_normalizado`

Para melhorar o desempenho e manter representatividade geografica, o projeto utiliza uma amostragem por estado, limitada a 500 focos por UF em cada ano. O numero ideal de clusters foi escolhido pela Regra do Cotovelo, avaliando a inercia para diferentes valores de `k`.

Na execucao validada do projeto, o melhor valor encontrado foi:

| Ano | Melhor k |
| --- | ---: |
| 2023 | 2 |
| 2024 | 2 |
| 2025 | 2 |

Os resultados da clusterizacao foram exportados em mapas interativos Folium:

- `mapa_kmeans_2023.html`
- `mapa_kmeans_2024.html`
- `mapa_kmeans_2025.html`

#### 2.3.2 Regras de Associacao com Apriori

O algoritmo Apriori foi aplicado para identificar padroes frequentes entre variaveis climaticas, ambientais e sazonais. Para isso, variaveis numericas foram categorizadas:

| Variavel original | Categorias criadas |
| --- | --- |
| `precipitacao` | `chuva_baixa`, `chuva_media`, `chuva_alta` |
| `dias_sem_chuva` | `seca_baixa`, `seca_media`, `seca_alta` |
| `risco_fogo` | `risco_baixo`, `risco_medio`, `risco_alto` |
| `potencia_radiativa_fogo` | `potencia_baixa`, `potencia_media`, `potencia_alta` |
| `mes` | `verao`, `outono`, `inverno`, `primavera` |

As transacoes foram formadas por itens como chuva, seca, bioma, potencia, estacao e risco. Os parametros utilizados no Apriori foram:

| Parametro | Valor |
| --- | ---: |
| Suporte minimo | 0,015 |
| Confianca minima | 0,70 |
| Lift minimo | 1,30 |
| Amostra maxima | 50.000 registros |

O projeto filtra regras cujo consequente seja `risco=risco_alto`, priorizando relacoes diretamente associadas ao risco elevado de fogo.

## 3. Resultados e Discussao

### 3.1 Escolha de Parametros

Para a clusterizacao, foram testados diferentes valores de `k` por meio da Regra do Cotovelo. A inercia foi calculada para cada quantidade de clusters, e o projeto selecionou o valor com maior reducao relativa. Para os anos analisados, o valor escolhido foi `k=2`.

No Apriori, os limites de suporte, confianca e lift foram definidos para manter apenas regras com frequencia minima, alta confiabilidade e associacao positiva relevante.

### 3.2 Resultados dos Clusters

Os mapas gerados mostram a separacao dos focos de queimadas em dois grupos principais por ano. Como as variaveis usadas incluem localizacao, risco de fogo, dias sem chuva e precipitacao, os clusters representam combinacoes de proximidade geografica e condicoes climaticas.

Essa divisao permite observar regioes com caracteristicas semelhantes de queimadas, apoiando a interpretacao espacial dos focos ao longo do periodo analisado. Os centroides dos clusters tambem foram adicionados aos mapas, facilitando a visualizacao do centro aproximado de cada agrupamento.

### 3.3 Visualizacoes

O projeto gera graficos de barras com a quantidade de focos de queimada por estado para cada ano. A escala logaritmica foi utilizada para melhorar a leitura, pois existe grande diferenca entre estados com muitos registros e estados com menor quantidade de focos.

Alem dos graficos estatisticos, os mapas interativos Folium permitem explorar os focos por localizacao, cluster, municipio, estado e quantidade de dias sem chuva.

### 3.4 Regras Encontradas

O Apriori encontrou 31 regras associadas ao risco alto de fogo. A regra mais forte encontrada foi:

| Antecedente | Consequente | Suporte | Confianca | Lift |
| --- | --- | ---: | ---: | ---: |
| chuva=chuva_baixa + estacao=primavera + seca=seca_alta | risco=risco_alto | 0,2039 | 0,9987 | 1,3789 |

Essa regra indica que registros com baixa chuva, seca alta e ocorrencia na primavera apresentam probabilidade muito elevada de estarem associados ao risco alto de fogo.

Outras regras relevantes tambem associam `seca_alta`, `primavera`, `Caatinga`, `Cerrado`, `Pantanal` e baixa precipitacao ao risco elevado. Isso sugere que o periodo seco e a sazonalidade possuem papel importante na ocorrencia de condicoes favoraveis ao fogo.

Principais regras exportadas em `regras_apriori.csv`:

| Antecedente | Consequente | Suporte | Confianca | Lift |
| --- | --- | ---: | ---: | ---: |
| chuva=chuva_baixa + estacao=primavera + seca=seca_alta | risco=risco_alto | 0,2039 | 0,9987 | 1,3789 |
| bioma=Caatinga + estacao=primavera + seca=seca_alta | risco=risco_alto | 0,0233 | 0,9957 | 1,3748 |
| bioma=Caatinga + chuva=chuva_baixa + seca=seca_alta | risco=risco_alto | 0,0301 | 0,9954 | 1,3743 |
| bioma=Cerrado + estacao=primavera + seca=seca_alta | risco=risco_alto | 0,0954 | 0,9940 | 1,3723 |
| estacao=primavera + potencia=potencia_media + seca=seca_alta | risco=risco_alto | 0,0516 | 0,9912 | 1,3685 |

## 4. Conclusao

O trabalho demonstrou que tecnicas de aprendizado de maquina e mineracao de dados podem auxiliar na analise de queimadas no Brasil. A clusterizacao com K-Means permitiu identificar agrupamentos espaciais de focos de queimadas, enquanto o Apriori revelou padroes frequentes relacionados ao risco alto de fogo.

Os resultados mostram que baixa precipitacao, seca alta e primavera aparecem fortemente associadas ao risco elevado. Biomas como Caatinga, Cerrado, Amazonia e Pantanal tambem apareceram em regras relevantes, indicando a importancia de considerar fatores ambientais e sazonais na interpretacao das queimadas.

Como limitacoes, o projeto utiliza amostragem para melhorar o desempenho dos algoritmos e trabalha com os dados disponiveis no periodo de 2023 a 2025. Como trabalhos futuros, recomenda-se testar outros algoritmos de clusterizacao, incluir metricas como Silhouette Score, comparar modelos preditivos supervisionados e expandir a analise para series temporais mais longas.

## Como Executar o Projeto

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar acesso ao BigQuery, se necessario

Caso o arquivo `queimadas_2023_2025.parquet` nao exista, o projeto tentara baixar os dados da Base dos Dados via BigQuery. Para isso, crie um arquivo `.env` com o ID do projeto Google Cloud:

```env
MEU_PROJETO_ID=seu_projeto_google_cloud
```

### 3. Executar pelo notebook

Abra o arquivo `main.ipynb` e execute as celulas em ordem. O notebook realiza:

- importacao do pipeline;
- carregamento e limpeza dos dados;
- geracao dos mapas K-Means;
- visualizacao estatistica;
- preparacao dos dados para Apriori;
- mineracao e exportacao das regras de associacao.

### 4. Principais arquivos do projeto

| Arquivo | Funcao |
| --- | --- |
| `pipeline.py` | Contem as funcoes de carregamento, limpeza, clusterizacao, graficos e Apriori |
| `config.py` | Contem a query SQL, parametros e cores dos clusters |
| `main.ipynb` | Notebook principal de execucao |
| `queimadas_2023_2025.parquet` | Cache local dos dados |
| `mapa_kmeans_2023.html` | Mapa interativo de clusters de 2023 |
| `mapa_kmeans_2024.html` | Mapa interativo de clusters de 2024 |
| `mapa_kmeans_2025.html` | Mapa interativo de clusters de 2025 |
| `regras_apriori.csv` | Regras de associacao mineradas |

## Referencias

BASE DOS DADOS. Base dos Dados: repositorio publico de bases brasileiras. Disponivel em: <https://basedosdados.org/>. Acesso em: 28 maio 2026.

INSTITUTO NACIONAL DE PESQUISAS ESPACIAIS (INPE). Programa Queimadas. Disponivel em: <https://terrabrasilis.dpi.inpe.br/queimadas/>. Acesso em: 28 maio 2026.

AGRAWAL, Rakesh; SRIKANT, Ramakrishnan. Fast algorithms for mining association rules. In: INTERNATIONAL CONFERENCE ON VERY LARGE DATA BASES, 20., 1994. Proceedings. Santiago: VLDB, 1994.

MACQUEEN, J. Some methods for classification and analysis of multivariate observations. In: BERKELEY SYMPOSIUM ON MATHEMATICAL STATISTICS AND PROBABILITY, 5., 1967. Proceedings. Berkeley: University of California Press, 1967.
