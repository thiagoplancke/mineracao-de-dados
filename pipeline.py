import os
from dotenv import load_dotenv
import basedosdados as bd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import folium
from sklearn.cluster import KMeans
import config
from apyori import apriori

load_dotenv()  

def obter_dados():
    """Verifica se há dados locais em Parquet. Se não, busca na nuvem, limpa e salva."""
    
    # 1. Verifica se o Cache Local (Parquet) existe
    if os.path.exists(config.ARQUIVO_PARQUET):
        print(f"Lendo dados ultrarrápidos do arquivo local '{config.ARQUIVO_PARQUET}'...")
        df = pd.read_parquet(config.ARQUIVO_PARQUET)
        print(f"Sucesso! {len(df)} registros carregados da memória RAM.")
        return df
    
    # 2. Caso não exista, vai até a nuvem (BigQuery)
    print("Arquivo local não encontrado. Extraindo matriz do BigQuery... Aguarde.")
    df = bd.read_sql(config.QUERY_BASE, billing_project_id=os.getenv('MEU_PROJETO_ID'))
    
    # Limpeza obrigatória para o algoritmo K-Means
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
    
    # 3. Salva os dados em disco comprimidos para as próximas sessões
    print(f"Comprimindo e salvando cache local como '{config.ARQUIVO_PARQUET}'...")
    df.to_parquet(config.ARQUIVO_PARQUET, compression="snappy", index=False)
    
    print(f"Sucesso! {len(df)} registros prontos para análise.")
    return df

def gerar_mapas_kmeans(df):
    """Aplica Machine Learning espacial e exporta mapas interativos."""
    for ano_alvo in sorted(df['ano'].unique()):
        df_ano = df[df['ano'] == ano_alvo]
        
        df_amostra = (
            df_ano.groupby('sigla_uf', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), config.FOCOS_POR_ESTADO), random_state=42))
        ).copy()
        
        X = df_amostra[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42, n_init=10)
        df_amostra['cluster'] = kmeans.fit_predict(X)
        centroides = kmeans.cluster_centers_
        
        mapa = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles="cartodbpositron")
        
        for _, linha in df_amostra.iterrows():
            cluster_id = int(linha['cluster'])
            popup_texto = f"<b>{linha['id_municipio_nome']}-{linha['sigla_uf']}</b><br>Cluster: {cluster_id + 1}<br>Dias s/ chuva: {linha['dias_sem_chuva']}"
            
            folium.CircleMarker(
                location=[linha['latitude'], linha['longitude']], radius=2.5,
                popup=folium.Popup(popup_texto, max_width=200),
                color=config.CORES_CLUSTERS[cluster_id], fill=True,
                fill_color=config.CORES_CLUSTERS[cluster_id], fill_opacity=0.6, weight=0
            ).add_to(mapa)
            
        for idx, centro in enumerate(centroides):
            folium.Marker(
                location=[centro[0], centro[1]],
                popup=folium.Popup(f"<b>EPICENTRO DO CLUSTER {idx + 1} ({ano_alvo})</b>", max_width=250),
                icon=folium.Icon(color='red', icon='fire', prefix='fa')
            ).add_to(mapa)
            
        nome_arquivo = f"mapa_kmeans_{ano_alvo}.html"
        mapa.save(nome_arquivo)
        print(f"-> Mapa interativo de {ano_alvo} gerado: {nome_arquivo}")

def plotar_graficos(df):
    """Gera gráficos de ranking anual por estado com colunas vermelhas
    e os números de incêndios registrados exibidos no corpo de cada barra."""
    
    # Define o estilo visual de fundo
    sns.set_theme(style="whitegrid")
    
    # Formatador do eixo Y para manter as marcações de apoio limpas
    def formata_valores_dinamicos(x, pos):
        if x == 0: return '0'
        if x >= 1_000_000: return f"{x / 1_000_000:.1f}".replace('.', ',').rstrip(',0') + " Mi"
        if x >= 1_000: return f"{x / 1_000:.1f}".replace('.', ',').rstrip(',0') + " Mil"
        return str(int(x))

    # Loop para gerar um gráfico para cada ano
    for ano_alvo in sorted(df['ano'].unique()):
        df_ano = df[df['ano'] == ano_alvo]
        
        if df_ano.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(15, 6))
        ordem_estados = df_ano['sigla_uf'].value_counts().index
        
        # --- ALTERAÇÃO DE COR: Forçamos todas as colunas para um vermelho sólido (Crimson) ---
        sns.countplot(
            data=df_ano, 
            x='sigla_uf', 
            order=ordem_estados, 
            color='#D32F2F',  # Vermelho vivo profissional
            ax=ax
        )
        
        # Configurações da escala logarítmica
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)  # Garante que a base do gráfico comece em 1 para o cálculo do centro
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(formata_valores_dinamicos))
        
        # --- ADICIONANDO OS NÚMEROS NO CORPO DAS COLUNAS ---
        for p in ax.patches:
            height = p.get_height()
            if height > 1:
                # Formata o número com pontos de milhar padrão BR (ex: 145.230)
                valor_formatado = f"{int(height):,}".replace(",", ".")
                
                # Alinhamento horizontal: exatamente no meio da largura da coluna
                x_pos = p.get_x() + p.get_width() / 2
                
                # Controle de segurança para estados com pouquíssimas queimadas (barras muito baixas)
                if height < 150:
                    # Se a barra for minúscula, o texto não cabe dentro. Colocamos ACIMA dela em preto.
                    y_pos = height * 1.5
                    cor_texto = 'black'
                    peso_fonte = 'normal'
                else:
                    # Se a barra for grande, o texto vai DENTRO do corpo em branco.
                    # O segredo: height ** 0.5 acha o centro visual exato na escala LOG
                    y_pos = height ** 0.5
                    cor_texto = 'white'
                    peso_fonte = 'bold'
                
                # Desenha o texto rotacionado em 90 graus para caber perfeitamente na coluna
                ax.text(
                    x_pos, y_pos, valor_formatado,
                    ha='center', va='center',
                    rotation=90, fontsize=9,
                    color=cor_texto, fontweight=peso_fonte
                )

        # Títulos e formatações estéticas
        plt.title(f'Focos de Incêndio por Estado — Ano {ano_alvo}', fontsize=14, fontweight='bold')
        plt.xlabel('Estado (UF)', fontsize=12)
        plt.ylabel('Quantidade de Focos (Escala Logarítmica)', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def minerar_regras_clima_bioma(
    df,
    amostra=50000,
    suporte_minimo=0.05,
    confianca_minima=0.6,
    lift_minimo=1.0,
    # === VARIÁVEIS CONFIGURÁVEIS PARA DISCRETIZAÇÃO ===
    colunas_categoricas=None,  # Ex: ['bioma', 'mes', 'satelite']
    colunas_numericas=None,    # Ex: ['precipitacao', 'risco_fogo', 'dias_sem_chuva']
    bins_discretizacao=None,   # Ex: {'precipitacao': [[-1, 0.5], [0.5, 10.0], [10.0, float('inf')]], ...}
    labels_categorias=None,    # Ex: {'precipitacao': ['Sem Chuva', 'Chuva Leve', 'Chuva Forte'], ...}
    filtros_adicionais=None,   # Ex: {'ano': [2023, 2024, 2025], 'bioma': ['Cerrado', 'Floresta Amazônica']}
    verbose=True
):

    
    # ========== VALORES PADRÃO (CONFIGURÁVEIS) ==========
    if colunas_categoricas is None:
        colunas_categoricas = ['bioma', 'mes', 'satelite', 'sigla_uf']
    
    if colunas_numericas is None:
        colunas_numericas = ['precipitacao', 'risco_fogo', 'dias_sem_chuva']
    
    # Bins padrão para discretização
    if bins_discretizacao is None:
        bins_discretizacao = {
            'precipitacao': [[-1, 0.5], [0.5, 10.0], [10.0, float('inf')]],
            'risco_fogo': [[-1, 0.3], [0.3, 0.7], [0.7, float('inf')]],
            'dias_sem_chuva': [[-1, 7], [7, 14], [14, float('inf')]]
        }
    
    # Labels padrão para categorias
    if labels_categorias is None:
        labels_categorias = {
            'precipitacao': ['Sem Chuva', 'Chuva Leve', 'Chuva Forte'],
            'risco_fogo': ['Risco Baixo', 'Risco Médio', 'Risco Alto'],
            'dias_sem_chuva': ['Dias Poucos', 'Dias Moderados', 'Dias Muitos']
        }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ALGORITMO APRIORI - MINERAÇÃO DE REGRAS CLIMÁTICAS")
        print(f"{'='*70}")
    
    # ========== VALIDAÇÃO INICIAL ==========
    colunas_necessarias = colunas_categoricas + colunas_numericas
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
    
    if colunas_faltantes:
        print(f"Erro: Colunas não encontradas no DataFrame: {colunas_faltantes}")
        print(f"Colunas disponíveis: {sorted(df.columns.tolist())}")
        return pd.DataFrame()
    
    # ========== PRÉ-PROCESSAMENTO ==========
    df_trabalho = df.copy()
    
    # Aplicar filtros adicionais
    if filtros_adicionais:
        if verbose:
            print(f"\nAplicando filtros adicionais...")
        for col, valores in filtros_adicionais.items():
            if col in df_trabalho.columns:
                df_trabalho = df_trabalho[df_trabalho[col].isin(valores)]
                if verbose:
                    print(f"   - {col}: {valores}")
    
    # Selecionar apenas as colunas necessárias
    df_trabalho = df_trabalho[colunas_necessarias].copy().dropna()
    
    if df_trabalho.empty:
        print("Erro: Nenhum registro após aplicar filtros e remover valores nulos.")
        return pd.DataFrame()
    
    if verbose:
        print(f"\nDados carregados: {len(df_trabalho)} registros")
    
    # ========== DISCRETIZAÇÃO DE VARIÁVEIS NUMÉRICAS ==========
    if verbose:
        print(f"\nDiscretizando variáveis numéricas...")
    
    for col in colunas_numericas:
        if col in df_trabalho.columns:
            bins = bins_discretizacao.get(col, [[-1, float('inf')]])
            labels = labels_categorias.get(col, None)

            # Normaliza bins: aceita tanto formato [[a,b], [b,c]] quanto [a, b, c]
            try:
                if isinstance(bins, (list, tuple)) and bins and all(isinstance(b, (list, tuple)) for b in bins):
                    # bins fornecido como lista de intervalos -> converter para bordas
                    if all(len(b) == 2 for b in bins):
                        edges = [bins[0][0]] + [b[1] for b in bins]
                    else:
                        raise ValueError("bins como intervalos devem ter pares [inicio, fim]")
                else:
                    # Assume já é uma sequência de bordas
                    edges = list(bins)

                # Garante que labels tenham comprimento correto
                if labels is None:
                    labels = [f"{col}_cat_{i+1}" for i in range(len(edges) - 1)]

                df_trabalho[f'{col}_cat'] = pd.cut(
                    df_trabalho[col],
                    bins=edges,
                    labels=labels,
                    include_lowest=True
                )

                if verbose:
                    n_cats = df_trabalho[f'{col}_cat'].nunique(dropna=True)
                    print(f"   ✓ {col}: {n_cats} categorias criadas")
            except Exception as e:
                print(f"   ⚠️  Erro ao discretizar '{col}': {str(e)}")
    
    # ========== PREPARAÇÃO PARA APYORI ==========
    if verbose:
        print(f"\nPreparando transações para Apyori...")
    
    # Monta a lista de colunas finais (categorias + numéricas discretizadas)
    colunas_finais = colunas_categoricas.copy()
    for col in colunas_numericas:
        if f'{col}_cat' in df_trabalho.columns:
            colunas_finais.append(f'{col}_cat')
    
    df_cat = df_trabalho[colunas_finais].astype(str)
    
    # Amostragem
    n_amostra = min(amostra, len(df_cat))
    df_sample = df_cat.sample(n=n_amostra, random_state=42)
    
    if verbose:
        print(f"Amostra utilizada: {n_amostra} registros")
    
    # Converter para lista de transações
    transacoes = df_sample.values.tolist()
    
    # ========== MINERAÇÃO COM APYORI ==========
    if verbose:
        print(f"\n⛏️  Minerando regras com Apyori...")
        print(f"   - Suporte mínimo: {suporte_minimo}")
        print(f"   - Confiança mínima: {confianca_minima}")
        print(f"   - Lift mínimo: {lift_minimo}")
    
    regras_brutas = list(apriori(
        transacoes,
        min_support=suporte_minimo,
        min_confidence=confianca_minima,
        min_lift=lift_minimo
    ))

    if verbose:
        print(f"   ✓ {len(regras_brutas)} itemsets frequentes encontrados")
    
    # ========== FORMATAÇÃO DOS RESULTADOS ==========
    if verbose:
        print(f"\n📊 Formatando resultados...")
    
    linhas_regras = []
    
    for registro in regras_brutas:
        for estatistica in registro.ordered_statistics:
            if len(estatistica.items_base) == 0:
                continue
            
            linhas_regras.append({
                'antecedentes': ', '.join(sorted(list(estatistica.items_base))),
                'consequentes': ', '.join(sorted(list(estatistica.items_add))),
                'suporte': round(registro.support, 4),
                'confianca': round(estatistica.confidence, 4),
                'lift': round(estatistica.lift, 4),
                'items_antecedentes': list(estatistica.items_base),
                'items_consequentes': list(estatistica.items_add)
            })
    
    if not linhas_regras:
        print("⚠️  Nenhuma regra encontrada com estes parâmetros.")
        return pd.DataFrame()
    
    df_regras = pd.DataFrame(linhas_regras)
    df_regras = df_regras.sort_values(by='lift', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\n✅ {len(df_regras)} regras extraídas com sucesso!")
        print(f"\n{'='*70}")
        print("TOP 5 REGRAS (Ordenadas por LIFT):")
        print(f"{'='*70}")
        for idx, row in df_regras.head(5).iterrows():
            print(f"\n[Regra #{idx + 1}]")
            print(f"  Se:     {row['antecedentes']}")
            print(f"  Então:  {row['consequentes']}")
            print(f"  Métrica: Suporte={row['suporte']}, Confiança={row['confianca']}, Lift={row['lift']}")
    
    return df_regras