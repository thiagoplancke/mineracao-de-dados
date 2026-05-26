import os
from dotenv import load_dotenv

import basedosdados as bd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import folium

from sklearn.cluster import KMeans
from apyori import apriori

import config

load_dotenv()


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================

def obter_dados():

    # =====================================================
    # CARREGA DADOS
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
    # LIMPEZA
    # =====================================================

    colunas_importantes = [
        "latitude",
        "longitude",
        "risco_fogo",
        "precipitacao",
        "dias_sem_chuva"
    ]

    df = df.dropna(subset=colunas_importantes)

    # garante numérico
    for coluna in colunas_importantes:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    # remove infinito
    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # remove NaN novamente
    df = df.dropna(subset=colunas_importantes)

    # =====================================================
    # COORDENADAS VÁLIDAS
    # =====================================================

    df = df[
        (df["latitude"] >= -35) &
        (df["latitude"] <= 10)
    ]

    df = df[
        (df["longitude"] >= -75) &
        (df["longitude"] <= -30)
    ]

    # =====================================================
    # LIMITES LÓGICOS
    # =====================================================

    df = df[df["risco_fogo"] >= 0]
    df = df[df["dias_sem_chuva"] >= 0]
    df = df[df["precipitacao"] >= 0]

    # =====================================================
    # REMOVER OUTLIERS
    # =====================================================

    colunas_numericas = [
        "precipitacao",
        "risco_fogo",
        "dias_sem_chuva"
    ]

    for coluna in colunas_numericas:

        q1 = df[coluna].quantile(0.01)
        q99 = df[coluna].quantile(0.99)

        df = df[
            (df[coluna] >= q1) &
            (df[coluna] <= q99)
        ]

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================

    for coluna in colunas_numericas:

        minimo = df[coluna].min()
        maximo = df[coluna].max()

        df[f"{coluna}_normalizado"] = (
            (df[coluna] - minimo)
            / (maximo - minimo)
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

def encontrar_k_ideal(df, max_clusters=10):
    """
    Usa regra do cotovelo para descobrir o melhor K.
    """

    # ==========================================
    # FEATURES DO CLUSTER
    # ==========================================

    X = df[
        [
            "latitude",
            "longitude",
            "risco_fogo_normalizado",
            "dias_sem_chuva_normalizado",
            "precipitacao_normalizado"
        ]
    ].copy()

    # ==========================================
    # GARANTE NUMÉRICO
    # ==========================================

    for coluna in X.columns:

        X[coluna] = pd.to_numeric(
            X[coluna],
            errors="coerce"
        )

    # ==========================================
    # REMOVE NaN E INFINITO
    # ==========================================

    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X = X.dropna()

    # ==========================================
    # SEGURANÇA
    # ==========================================

    if len(X) < 2:

        print("Poucos dados válidos para KMeans.")

        return 2

    # ==========================================
    # REGRA DO COTOVELO
    # ==========================================

    inercias = []

    max_clusters = min(max_clusters, len(X))

    for k in range(1, max_clusters + 1):

        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        modelo.fit(X)

        inercias.append(modelo.inertia_)

    # ==========================================
    # MELHOR K
    # ==========================================

    diferencas = []

    for i in range(1, len(inercias)):

        diferencas.append(
            inercias[i - 1] - inercias[i]
        )

    melhor_k = (
        diferencas.index(max(diferencas)) + 2
    )

    # ==========================================
    # GRÁFICO
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, max_clusters + 1),
        inercias,
        marker="o"
    )

    plt.xlabel("Número de Clusters (K)")
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
    """
    Aplica KMeans usando automaticamente
    o K encontrado pela regra do cotovelo.
    """

    for ano_alvo in sorted(df["ano"].unique()):

        print(f"\nProcessando ano {ano_alvo}...")

        df_ano = df[df["ano"] == ano_alvo].copy()

        # ==========================================
        # GARANTE NUMÉRICO
        # ==========================================

        df_ano["latitude"] = pd.to_numeric(
            df_ano["latitude"],
            errors="coerce"
        )

        df_ano["longitude"] = pd.to_numeric(
            df_ano["longitude"],
            errors="coerce"
        )

        # ==========================================
        # REMOVE NAN
        # ==========================================

        df_ano = df_ano.dropna(
            subset=["latitude", "longitude"]
        )

        # ==========================================
        # REMOVE INFINITO
        # ==========================================

        df_ano = df_ano[
            ~df_ano["latitude"].isin([float("inf"), float("-inf")])
        ]

        df_ano = df_ano[
            ~df_ano["longitude"].isin([float("inf"), float("-inf")])
        ]

        # ==========================================
        # AMOSTRAGEM
        # ==========================================

        df_amostra = (
            df_ano.groupby("sigla_uf", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), config.FOCOS_POR_ESTADO),
                    random_state=42
                )
            )
        )

        # ==========================================
        # MATRIZ KMEANS
        # ==========================================

        X = df_amostra[
            [
                "latitude",
                "longitude",
                "risco_fogo_normalizado",
                "dias_sem_chuva_normalizado",
                "precipitacao_normalizado"
            ]
        ].copy()

        # garante numérico
        for coluna in X.columns:

            X[coluna] = pd.to_numeric(
                X[coluna],
                errors="coerce"
            )

        # remove infinito
        X = X.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )

        # remove NaN
        X = X.dropna()

        if len(X) < 2:

            print("Poucos dados válidos.")

            continue

        # ==========================================
        # REGRA DO COTOVELO
        # ==========================================

        melhor_k = encontrar_k_ideal(df_amostra)

        print(f"K ideal encontrado: {melhor_k}")

        # ==========================================
        # KMEANS
        # ==========================================

        modelo = KMeans(
            n_clusters=melhor_k,
            random_state=42,
            n_init=10
        )

        clusters = modelo.fit_predict(X)

        df_amostra = df_amostra.loc[X.index].copy()

        df_amostra["cluster"] = clusters

        centroides = modelo.cluster_centers_

        # ==========================================
        # MAPA
        # ==========================================

        mapa = folium.Map(
            location=[-14.2350, -51.9253],
            zoom_start=4,
            tiles="cartodbpositron"
        )

        for _, linha in df_amostra.iterrows():

            cluster = int(linha["cluster"])

            cor = config.CORES_CLUSTERS[
                cluster % len(config.CORES_CLUSTERS)
            ]

            popup = (
                f"<b>{linha['id_municipio_nome']} - "
                f"{linha['sigla_uf']}</b><br>"
                f"Cluster: {cluster}<br>"
                f"Dias sem chuva: {linha['dias_sem_chuva']}"
            )

            folium.CircleMarker(
                location=[
                    linha["latitude"],
                    linha["longitude"]
                ],
                radius=2.5,
                popup=popup,
                color=cor,
                fill=True,
                fill_color=cor,
                fill_opacity=0.6,
                weight=0
            ).add_to(mapa)

        # ==========================================
        # CENTRÓIDES
        # ==========================================

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

        plt.figure(figsize=(15, 6))

        ordem = df_ano["sigla_uf"].value_counts().index

        ax = sns.countplot(
            data=df_ano,
            x="sigla_uf",
            order=ordem,
            color="#D32F2F"
        )

        ax.set_yscale("log")

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(formatar_y)
        )

        for p in ax.patches:

            valor = int(p.get_height())

            if valor <= 0:
                continue

            ax.text(
                p.get_x() + p.get_width() / 2,
                valor,
                f"{valor:,}".replace(",", "."),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
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
    """
    Cria categorias climáticas simples e interpretáveis.
    """

    df = df.copy()

    # Precipitação
    df["categoria_chuva"] = pd.cut(
        df["precipitacao"],
        bins=[-1, 1, 10, float("inf")],
        labels=[
            "chuva_baixa",
            "chuva_media",
            "chuva_alta"
        ]
    )

    # Dias sem chuva
    df["categoria_seca"] = pd.cut(
        df["dias_sem_chuva"],
        bins=[-1, 7, 15, float("inf")],
        labels=[
            "seca_baixa",
            "seca_media",
            "seca_alta"
        ]
    )

    # Risco de fogo
    df["categoria_risco"] = pd.cut(
        df["risco_fogo"],
        bins=[-1, 0.3, 0.7, float("inf")],
        labels=[
            "risco_baixo",
            "risco_medio",
            "risco_alto"
        ]
    )

    return df.dropna(
        subset=[
            "categoria_chuva",
            "categoria_seca",
            "categoria_risco"
        ]
    )


# =========================================================
# APRIORI
# =========================================================

def minerar_regras(df):

    """
    Descobre padrões climáticos relacionados
    ao risco alto de fogo.
    """

    df = preparar_dados_apriori(df)

    transacoes = []

    for _, linha in df.iterrows():

        transacao = [

            f"chuva={linha['categoria_chuva']}",

            f"seca={linha['categoria_seca']}",

            f"bioma={linha['bioma']}",

            f"risco={linha['categoria_risco']}"
        ]

        transacoes.append(transacao)

    regras = apriori(
        transacoes,
        min_support=0.05,
        min_confidence=0.6,
        min_lift=1.2
    )

    resultados = []

    for regra in regras:

        for estatistica in regra.ordered_statistics:

            antecedente = list(estatistica.items_base)

            consequente = list(estatistica.items_add)

            if not consequente:
                continue

            # Só queremos regras que levam a risco alto
            if "risco=risco_alto" not in consequente:
                continue

            resultados.append({

                "antecedente": ", ".join(antecedente),

                "consequente": ", ".join(consequente),

                "suporte": round(regra.support, 4),

                "confianca": round(
                    estatistica.confidence,
                    4
                ),

                "lift": round(
                    estatistica.lift,
                    4
                )
            })

    df_regras = pd.DataFrame(resultados)

    if df_regras.empty:

        print("Nenhuma regra encontrada.")

        return df_regras

    df_regras = df_regras.sort_values(
        by=["lift", "confianca"],
        ascending=False
    )

    print("\nTOP 10 REGRAS CLIMÁTICAS:\n")

    for _, linha in df_regras.head(10).iterrows():

        print(f"SE: {linha['antecedente']}")

        print(f"ENTÃO: {linha['consequente']}")

        print(
            f"Suporte={linha['suporte']} | "
            f"Confiança={linha['confianca']} | "
            f"Lift={linha['lift']}"
        )

        print("-" * 60)

    return df_regras