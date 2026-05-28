import os
from dotenv import load_dotenv

import basedosdados as bd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import folium

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from apyori import apriori

import config

load_dotenv()

# =========================================================
# CONFIGURAÇÕES
# =========================================================

COLUNAS_NUMERICAS = [
    "latitude",
    "longitude",
    "risco_fogo",
    "precipitacao",
    "dias_sem_chuva"
]

FEATURES_CLUSTER = [
    "latitude",
    "longitude",
    "risco_fogo_normalizado",
    "dias_sem_chuva_normalizado",
    "precipitacao_normalizado"
]


# =========================================================
# CARREGAMENTO E LIMPEZA
# =========================================================

def obter_dados():

    # =====================================================
    # CACHE
    # =====================================================

    if os.path.exists(config.ARQUIVO_PARQUET):

        print(f"Lendo cache local: {config.ARQUIVO_PARQUET}")

        df = pd.read_parquet(config.ARQUIVO_PARQUET)

    else:

        print("Baixando dados do BigQuery...")

        df = bd.read_sql(
            config.QUERY_BASE,
            billing_project_id=os.getenv("MEU_PROJETO_ID")
        )

        print("Salvando cache local...")

        df.to_parquet(
            config.ARQUIVO_PARQUET,
            compression="snappy",
            index=False
        )

    # =====================================================
    # CONVERSÃO NUMÉRICA (vetorizada)
    # =====================================================

    df[COLUNAS_NUMERICAS] = (
        df[COLUNAS_NUMERICAS]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float32")
    )

    # =====================================================
    # REMOVE NaN E INFINITOS
    # =====================================================

    df = (
        df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=COLUNAS_NUMERICAS)
    )

    # =====================================================
    # FILTROS ÚNICOS
    # =====================================================

    filtro = (

        # coordenadas válidas
        (df["latitude"].between(-35, 10)) &
        (df["longitude"].between(-75, -30)) &

        # limites lógicos
        (df["risco_fogo"] >= 0) &
        (df["dias_sem_chuva"] >= 0) &
        (df["precipitacao"] >= 0)
    )

    df = df[filtro]

    # =====================================================
    # OUTLIERS (máscara única)
    # =====================================================

    mascara_outlier = pd.Series(True, index=df.index)

    for coluna in [
        "precipitacao",
        "risco_fogo",
        "dias_sem_chuva"
    ]:

        q1 = df[coluna].quantile(0.01)
        q99 = df[coluna].quantile(0.99)

        mascara_outlier &= df[coluna].between(q1, q99)

    df = df[mascara_outlier]

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================

    scaler = MinMaxScaler()

    colunas_normalizadas = [
        "risco_fogo",
        "dias_sem_chuva",
        "precipitacao"
    ]

    dados_normalizados = scaler.fit_transform(
        df[colunas_normalizadas]
    )

    for i, coluna in enumerate(colunas_normalizadas):

        df[f"{coluna}_normalizado"] = (
            dados_normalizados[:, i].astype("float32")
        )

    # =====================================================
    # RESET INDEX
    # =====================================================

    df = df.reset_index(drop=True)

    print(f"Dados limpos: {len(df)}")

    return df


# =========================================================
# REGRA DO COTOVELO
# =========================================================

def encontrar_k_ideal(X, max_clusters=10):

    if len(X) < 2:

        return 2

    # =====================================================
    # AMOSTRA PARA PERFORMANCE
    # =====================================================

    if len(X) > 5000:

        X = X.sample(
            n=5000,
            random_state=42
        )

    inercias = []

    max_clusters = min(max_clusters, len(X))

    for k in range(1, max_clusters + 1):

        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=5
        )

        modelo.fit(X)

        inercias.append(modelo.inertia_)

    # =====================================================
    # MELHOR K
    # =====================================================

    diferencas = np.diff(inercias) * -1

    melhor_k = int(np.argmax(diferencas) + 2)

    # =====================================================
    # GRÁFICO
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, max_clusters + 1),
        inercias,
        marker="o"
    )

    plt.xlabel("Número de Clusters")
    plt.ylabel("Inércia")
    plt.title("Regra do Cotovelo")

    plt.grid(True)

    plt.show()

    print(f"Melhor K encontrado: {melhor_k}")

    return melhor_k


# =========================================================
# KMEANS + MAPA
# =========================================================

def gerar_mapas_kmeans(df):

    for ano_alvo in sorted(df["ano"].unique()):

        print(f"\nProcessando ano {ano_alvo}...")

        df_ano = df[df["ano"] == ano_alvo]

        # =====================================================
        # AMOSTRAGEM POR ESTADO
        # =====================================================

        df_amostra = (
                df_ano
                .groupby("sigla_uf", group_keys=False)
                .sample(
                    frac=1,
                    random_state=42
                )
                .groupby("sigla_uf", group_keys=False)
                .head(config.FOCOS_POR_ESTADO)
            )

        # =====================================================
        # FEATURES
        # =====================================================

        X = (
            df_amostra[FEATURES_CLUSTER]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(X) < 2:

            print("Poucos dados válidos.")

            continue

        # =====================================================
        # K IDEAL
        # =====================================================

        melhor_k = encontrar_k_ideal(X)

        # =====================================================
        # KMEANS
        # =====================================================

        modelo = KMeans(
            n_clusters=melhor_k,
            random_state=42,
            n_init=10
        )

        clusters = modelo.fit_predict(X)

        df_amostra = df_amostra.loc[X.index]

        df_amostra["cluster"] = clusters

        centroides = modelo.cluster_centers_

        # =====================================================
        # MAPA
        # =====================================================

        mapa = folium.Map(
            location=[-14.2350, -51.9253],
            zoom_start=4,
            tiles="cartodbpositron"
        )

        # itertuples é MUITO mais rápido
        for linha in df_amostra.itertuples():

            cluster = int(linha.cluster)

            cor = config.CORES_CLUSTERS[
                cluster % len(config.CORES_CLUSTERS)
            ]

            popup = (
                f"<b>{linha.id_municipio_nome} - "
                f"{linha.sigla_uf}</b><br>"
                f"Cluster: {cluster}<br>"
                f"Dias sem chuva: {linha.dias_sem_chuva}"
            )

            folium.CircleMarker(
                location=[
                    linha.latitude,
                    linha.longitude
                ],
                radius=2,
                popup=popup,
                color=cor,
                fill=True,
                fill_color=cor,
                fill_opacity=0.5,
                weight=0
            ).add_to(mapa)

        # =====================================================
        # CENTRÓIDES
        # =====================================================

        for idx, centro in enumerate(centroides):

            folium.Marker(
                location=[centro[0], centro[1]],
                popup=f"Centro Cluster {idx}",
                icon=folium.Icon(
                    color="red",
                    icon="fire",
                    prefix="fa"
                )
            ).add_to(mapa)

        nome_arquivo = f"mapa_kmeans_{ano_alvo}.html"

        mapa.save(nome_arquivo)

        print(f"Mapa salvo: {nome_arquivo}")


# =========================================================
# GRÁFICOS
# =========================================================

def plotar_graficos(df):

    sns.set_theme(style="whitegrid")

    def formatar_y(x, pos):

        if x >= 1_000_000:
            return f"{x / 1_000_000:.1f} Mi"

        if x >= 1_000:
            return f"{x / 1_000:.1f} Mil"

        return str(int(x))

    for ano in sorted(df["ano"].unique()):

        df_ano = df[df["ano"] == ano]

        contagem = (
            df_ano["sigla_uf"]
            .value_counts()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(15, 6))

        ax = plt.bar(
            contagem.index,
            contagem.values
        )

        plt.yscale("log")

        plt.gca().yaxis.set_major_formatter(
            ticker.FuncFormatter(formatar_y)
        )

        plt.title(f"Focos de Queimada por Estado ({ano})")

        plt.xlabel("Estado")
        plt.ylabel("Quantidade")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()


# =========================================================
# PREPARAÇÃO APRIORI
# =========================================================

def preparar_dados_apriori(df):

    df = df.copy()

    # =====================================================
    # CHUVA
    # =====================================================

    df["categoria_chuva"] = pd.cut(
        df["precipitacao"],
        bins=[-1, 1, 10, np.inf],
        labels=[
            "chuva_baixa",
            "chuva_media",
            "chuva_alta"
        ]
    )

    # =====================================================
    # SECA
    # =====================================================

    df["categoria_seca"] = pd.cut(
        df["dias_sem_chuva"],
        bins=[-1, 7, 15, np.inf],
        labels=[
            "seca_baixa",
            "seca_media",
            "seca_alta"
        ]
    )

    # =====================================================
    # RISCO
    # =====================================================

    df["categoria_risco"] = pd.cut(
        df["risco_fogo"],
        bins=[-1, 0.3, 0.7, np.inf],
        labels=[
            "risco_baixo",
            "risco_medio",
            "risco_alto"
        ]
    )

    # =====================================================
    # POTÊNCIA DO FOGO
    # =====================================================

    df["categoria_potencia"] = pd.cut(
        df["potencia_radiativa_fogo"],
        bins=[-1, 10, 50, np.inf],
        labels=[
            "potencia_baixa",
            "potencia_media",
            "potencia_alta"
        ]
    )

    # =====================================================
    # ESTAÇÃO DO ANO
    # =====================================================

    mapa_estacoes = {

        12: "verao",
        1: "verao",
        2: "verao",

        3: "outono",
        4: "outono",
        5: "outono",

        6: "inverno",
        7: "inverno",
        8: "inverno",

        9: "primavera",
        10: "primavera",
        11: "primavera"
    }

    df["estacao"] = df["mes"].map(
        mapa_estacoes
    )

    # =====================================================
    # REMOVE NULOS
    # =====================================================

    return df.dropna(
        subset=[
            "categoria_chuva",
            "categoria_seca",
            "categoria_risco",
            "categoria_potencia",
            "estacao"
        ]
    )


# =========================================================
# APRIORI
# =========================================================

def minerar_regras(df):

    print("Preparando Apriori...")

    df = preparar_dados_apriori(df)

    # =====================================================
    # LIMITA AMOSTRA
    # =====================================================

    if len(df) > 50000:

        df = df.sample(
            n=50000,
            random_state=42
        )

    # =====================================================
    # TRANSAÇÕES
    # =====================================================

    transacoes = [

        [

            f"chuva={linha.categoria_chuva}",

            f"seca={linha.categoria_seca}",

            f"bioma={linha.bioma}",

            f"potencia={linha.categoria_potencia}",

            f"estacao={linha.estacao}",

            f"risco={linha.categoria_risco}"

        ]

        for linha in df.itertuples()
    ]

    # =====================================================
    # APRIORI
    # =====================================================

    regras = apriori(

        transacoes,

        min_support=0.015,

        min_confidence=0.7,

        min_lift=1.3
    )

    resultados = []

    regras_unicas = set()

    # =====================================================
    # FILTRA REGRAS
    # =====================================================

    for regra in regras:

        for estatistica in regra.ordered_statistics:

            antecedente = list(
                estatistica.items_base
            )

            consequente = list(
                estatistica.items_add
            )

            # =============================================
            # REGRAS VAZIAS
            # =============================================

            if len(antecedente) == 0:
                continue

            # =============================================
            # QUEREMOS MAIS CONTEXTO
            # =============================================

            if len(antecedente) < 2:
                continue

            # máximo 3 antecedentes
            if len(antecedente) > 3:
                continue

            # apenas 1 consequente
            if len(consequente) != 1:
                continue

            # =============================================
            # CONSEQUENTE
            # =============================================

            if consequente[0] != "risco=risco_alto":
                continue

            # =============================================
            # EVITA DUPLICADAS
            # =============================================

            antecedente = sorted(
                antecedente
            )

            chave_regra = (
                tuple(antecedente),
                consequente[0]
            )

            if chave_regra in regras_unicas:
                continue

            regras_unicas.add(
                chave_regra
            )

            resultados.append({

                "antecedente": " + ".join(
                    antecedente
                ),

                "consequente": consequente[0],

                "suporte": round(
                    regra.support,
                    4
                ),

                "confianca": round(
                    estatistica.confidence,
                    4
                ),

                "lift": round(
                    estatistica.lift,
                    4
                ),

                "tamanho_regra": len(
                    antecedente
                )
            })

    # =====================================================
    # DATAFRAME
    # =====================================================

    df_regras = pd.DataFrame(
        resultados
    )

    if df_regras.empty:

        print("Nenhuma regra encontrada.")

        return df_regras

    # =====================================================
    # PRIORIZAÇÃO
    # =====================================================

    df_regras = df_regras.sort_values(

        by=[
            "confianca",
            "lift",
            "suporte"
        ],

        ascending=[
            False,
            False,
            False
        ]
    )

    # remove auxiliar
    df_regras = df_regras.drop(
        columns=["tamanho_regra"]
    )

    # =====================================================
    # TOP REGRAS
    # =====================================================

    print("\nTOP 10 REGRAS CLIMÁTICAS:\n")

    for linha in df_regras.head(10).itertuples():

        print(f"SE: {linha.antecedente}")

        print(f"ENTÃO: {linha.consequente}")

        print(
            f"Confiança: {linha.confianca} | "
            f"Lift: {linha.lift} | "
            f"Suporte: {linha.suporte}"
        )

        print("-" * 60)

    return df_regras