import pandas as pd
import ast 
df = pd.read_csv('earthquakes_anual_completo.csv')
print(df["properties.tsunami"].value_counts())
features = ["properties.mag", "properties.sig", "properties.magType","properties.type", "geometry.coordinates", "properties.tsunami"]

clean_df = df[features]
lista_de_coordenadas = clean_df['geometry.coordinates'].apply(ast.literal_eval)

#print(lista_de_coordenadas)
clean_df['profundidade'] = lista_de_coordenadas.str[2]

# (Bônus) Vamos pegar a longitude e latitude também
clean_df['longitude'] = lista_de_coordenadas.str[0]
clean_df['latitude'] = lista_de_coordenadas.str[1]

# 4. Agora podemos nos livrar da coluna de string original
clean_df = clean_df.drop(columns=['geometry.coordinates'])
colunas_categoricas = ['properties.magType', 'properties.type']
clean_df = pd.get_dummies(clean_df, columns=colunas_categoricas, dtype=int)

print(clean_df[['longitude', 'latitude', 'profundidade']].head())
print(clean_df.info())
clean_df.to_csv("earthquakes_filtred.csv")