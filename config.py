
# Nome do arquivo local de alta performance
ARQUIVO_PARQUET = 'queimadas_2023_2025.parquet'

# Query SQL Otimizada (2023-2025)
QUERY_BASE = """
SELECT
    dados.ano, dados.mes, dados.bioma,
    dados.sigla_uf, diretorio_sigla_uf.nome AS sigla_uf_nome,
    dados.id_municipio, diretorio_id_municipio.nome AS id_municipio_nome,
    dados.latitude, dados.longitude, dados.satelite,
    dados.dias_sem_chuva, dados.precipitacao,
    dados.risco_fogo, dados.potencia_radiativa_fogo
FROM `basedosdados.br_inpe_queimadas.microdados` AS dados
LEFT JOIN (SELECT DISTINCT sigla, nome FROM `basedosdados.br_bd_diretorios_brasil.uf`) AS diretorio_sigla_uf
    ON dados.sigla_uf = diretorio_sigla_uf.sigla
LEFT JOIN (SELECT DISTINCT id_municipio, nome FROM `basedosdados.br_bd_diretorios_brasil.municipio`) AS diretorio_id_municipio
    ON dados.id_municipio = diretorio_id_municipio.id_municipio
WHERE dados.ano BETWEEN 2023 AND 2025
"""

# Parâmetros de Mineração
FOCOS_POR_ESTADO = 500
N_CLUSTERS = 7
CORES_CLUSTERS = ['blue', 'green', 'purple', 'orange', 'darkred', 'black', 'magenta']