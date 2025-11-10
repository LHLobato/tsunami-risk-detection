import pandas as pd
import ast 
from global_land_mask import globe  # <-- NOVO 1: Importar a biblioteca

df = pd.read_csv('earthquakes_anual_completo.csv')
print(df["properties.tsunami"].value_counts())

features = ["properties.mag", "properties.sig", "properties.magType",
            "properties.type", "geometry.coordinates", "properties.tsunami"]

clean_df = df[features]
lista_de_coordenadas = clean_df['geometry.coordinates'].apply(ast.literal_eval)

clean_df['profundidade'] = lista_de_coordenadas.str[2]
clean_df['longitude'] = lista_de_coordenadas.str[0]
clean_df['latitude'] = lista_de_coordenadas.str[1]


print("Calculando a feature 'is_land' (pode levar um minuto)...")

clean_df['is_land'] = clean_df.apply(
    lambda row: globe.is_land(row['latitude'], row['longitude']), 
    axis=1
)
clean_df['is_land'] = clean_df['is_land'].astype(int) 
print("Cálculo de 'is_land' concluído.")


clean_df = clean_df.drop(columns=['geometry.coordinates'])
colunas_categoricas = ['properties.magType', 'properties.type']
clean_df = pd.get_dummies(clean_df, columns=colunas_categoricas, dtype=int)

print(clean_df[['longitude', 'latitude', 'profundidade', 'is_land']].head()) 
print(clean_df.info())

clean_df['profundidade_segura'] = clean_df['profundidade'] + 1 

# Nova feature: Mag / Profundidade
clean_df['risco_mag_prof'] = clean_df['properties.mag'] / clean_df['profundidade_segura']

# Outra feature: Risco em terra
clean_df['risco_terra'] = clean_df['properties.mag'] * (1 - clean_df['is_land'])

clean_df.to_csv("earthquakes_filtred.csv", index=False)