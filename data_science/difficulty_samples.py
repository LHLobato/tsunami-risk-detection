import pandas as pd
import numpy as np
import warnings


pd.options.mode.chained_assignment = None 


MAG_THRESHOLD_HIGH = 6.0        
MAG_THRESHOLD_MEDIUM = 5.0      
DEPTH_THRESHOLD_DEEP = 100      

print("Carregando 'earthquakes_filtred.csv'...")
try:
    df = pd.read_csv("earthquakes_filtred.csv")
except FileNotFoundError:
    print("Erro: 'earthquakes_filtred.csv' não encontrado.")
    print("Por favor, rode o script de processamento de dados primeiro.")
    exit()

print(f"Total de amostras no dataset: {len(df)}")
print(f"Distribuição do alvo (properties.tsunami):\n{df['properties.tsunami'].value_counts(normalize=True)}\n")


base_filter = (df['properties.tsunami'] == 0)


filter_deep = (
    base_filter &
    (df['properties.mag'] >= MAG_THRESHOLD_HIGH) &
    (df['profundidade'] >= DEPTH_THRESHOLD_DEEP) &
    (df['properties.type_earthquake'] == 1)
amostras_profundas = df[filter_deep]
indices_profundos = amostras_profundas.index

print(f"--- 1. Pegadinhas 'Profundas' (Mag >= {MAG_THRESHOLD_HIGH}, Prof >= {DEPTH_THRESHOLD_DEEP}km) ---")
if amostras_profundas.empty:
    print("Nenhuma amostra encontrada.")
else:
    print(f"Encontradas {len(amostras_profundas)} amostras.")
    print(amostras_profundas[['properties.mag', 'profundidade', 'latitude', 'longitude', 'is_land']].head())
print("-" * 40)



filter_land = (
    base_filter &
    (df['properties.mag'] >= MAG_THRESHOLD_HIGH) &
    (df['is_land'] == 1) &
    (df['properties.type_earthquake'] == 1) 
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


filter_wrong_type = (
    base_filter &
    (df['properties.mag'] >= MAG_THRESHOLD_MEDIUM) &
    (df['properties.type_earthquake'] == 0) 
)
amostras_tipo_errado = df[filter_wrong_type]
indices_tipo_errado = amostras_tipo_errado.index

print(f"--- 3. Pegadinhas 'Tipo Errado' (Mag >= {MAG_THRESHOLD_MEDIUM}, type_earthquake == 0) ---")
if amostras_tipo_errado.empty:
    print("Nenhuma amostra encontrada.")
else:
    print(f"Encontradas {len(amostras_tipo_errado)} amostras.")

    type_cols = [col for col in df.columns if col.startswith('properties.type_') and col != 'properties.type_earthquake']
    print(amostras_tipo_errado[['properties.mag', 'profundidade'] + type_cols].head())
print("-" * 40)


all_difficult_indices = indices_profundos.union(indices_em_terra).union(indices_tipo_errado)

if len(all_difficult_indices) > 0:
    print(f"\n Total de {len(all_difficult_indices)} amostras únicas encontradas.")

    np.save('difficult_negative_idx.npy', all_difficult_indices.to_numpy())
    print("\nArquivo 'difficult_negative_idx.npy' salvo com os índices dessas amostras.")
else:
    print("\n Nenhuma amostra encontrada com esses critérios.")
