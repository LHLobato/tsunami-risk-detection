import pandas as pd
import numpy as np
import warnings

# Ignorar avisos de chained assignment (não são relevantes aqui)
pd.options.mode.chained_assignment = None 

# --- Ajuste seus critérios aqui ---
MAG_THRESHOLD_HIGH = 6.0        # O que você considera "magnitude alta"
MAG_THRESHOLD_MEDIUM = 5.0      # O que você considera "magnitude média"
DEPTH_THRESHOLD_DEEP = 100      # Profundidade (em km) para considerar "profundo"
# ------------------------------------

print("Carregando 'earthquakes_filtred.csv'...")
try:
    df = pd.read_csv("earthquakes_filtred.csv")
except FileNotFoundError:
    print("Erro: 'earthquakes_filtred.csv' não encontrado.")
    print("Por favor, rode o script de processamento de dados primeiro.")
    exit()

print(f"Total de amostras no dataset: {len(df)}")
print(f"Distribuição do alvo (properties.tsunami):\n{df['properties.tsunami'].value_counts(normalize=True)}\n")

# --- Filtro Base ---
# Queremos apenas amostras que NÃO SÃO tsunamis (Tsunami == 0)
# mas que PARECEM ser.
base_filter = (df['properties.tsunami'] == 0)

# --- 1. Caçar Terremotos Fortes (Mag >= 6.0) mas Profundos (>= 100km) ---
filter_deep = (
    base_filter &
    (df['properties.mag'] >= MAG_THRESHOLD_HIGH) &
    (df['profundidade'] >= DEPTH_THRESHOLD_DEEP) &
    (df['properties.type_earthquake'] == 1) # Garantir que é um terremoto
)
amostras_profundas = df[filter_deep]
indices_profundos = amostras_profundas.index

print(f"--- 1. Pegadinhas 'Profundas' (Mag >= {MAG_THRESHOLD_HIGH}, Prof >= {DEPTH_THRESHOLD_DEEP}km) ---")
if amostras_profundas.empty:
    print("Nenhuma amostra encontrada.")
else:
    print(f"Encontradas {len(amostras_profundas)} amostras.")
    print(amostras_profundas[['properties.mag', 'profundidade', 'latitude', 'longitude', 'is_land']].head())
print("-" * 40)


# --- 2. Caçar Terremotos Fortes (Mag >= 6.0) mas Em Terra (is_land == 1) ---
filter_land = (
    base_filter &
    (df['properties.mag'] >= MAG_THRESHOLD_HIGH) &
    (df['is_land'] == 1) &
    (df['properties.type_earthquake'] == 1) # Garantir que é um terremoto
)
amostras_em_terra = df[filter_land]
indices_em_terra = amostras_em_terra.index

print(f"--- 2. Pegadinhas 'Em Terra' (Mag >= {MAG_THRESHOLD_HIGH}, is_land == 1) ---")
if amostras_em_terra.empty:
    print("Nenhuma amostra encontrada.")
else:
    print(f"Encontradas {len(amostras_em_terra)} amostras.")
    print(amostras_em_terra[['properties.mag', 'profundidade', 'latitude', 'longitude', 'is_land']].head())
print("-" * 40)


# --- 3. Caçar Eventos Fortes (Mag >= 5.0) mas de Tipo Errado (Não-terremoto) ---
filter_wrong_type = (
    base_filter &
    (df['properties.mag'] >= MAG_THRESHOLD_MEDIUM) &
    (df['properties.type_earthquake'] == 0) # NÃO é um terremoto
)
amostras_tipo_errado = df[filter_wrong_type]
indices_tipo_errado = amostras_tipo_errado.index

print(f"--- 3. Pegadinhas 'Tipo Errado' (Mag >= {MAG_THRESHOLD_MEDIUM}, type_earthquake == 0) ---")
if amostras_tipo_errado.empty:
    print("Nenhuma amostra encontrada.")
else:
    print(f"Encontradas {len(amostras_tipo_errado)} amostras.")
    # Achar colunas de tipo para mostrar
    type_cols = [col for col in df.columns if col.startswith('properties.type_') and col != 'properties.type_earthquake']
    print(amostras_tipo_errado[['properties.mag', 'profundidade'] + type_cols].head())
print("-" * 40)


# --- AÇÃO FINAL: Salvar os índices ---
all_difficult_indices = indices_profundos.union(indices_em_terra).union(indices_tipo_errado)

if len(all_difficult_indices) > 0:
    print(f"\n✅ SUCESSO! Total de {len(all_difficult_indices)} amostras 'pegadinha' únicas encontradas.")
    print("Essas são as amostras que seu modelo precisa ver para aprender as exceções.")
    
    # Salvar os índices em um arquivo .npy
    np.save('difficult_negative_idx.npy', all_difficult_indices.to_numpy())
    print("\nArquivo 'difficult_negative_idx.npy' salvo com os índices dessas amostras.")
    print("\nPróximo passo: Ao criar seu 'bad_idx.npy', garanta que esses índices estejam incluídos!")
else:
    print("\n⚠️ Nenhuma amostra 'pegadinha' encontrada com esses critérios.")
    print("Isso pode significar que seu dataset NÃO TEM os exemplos que você precisa.")