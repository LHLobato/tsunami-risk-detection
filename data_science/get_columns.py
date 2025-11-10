import pandas as pd

df = pd.read_csv("earthquakes_anual_completo.csv")
print(df.columns)

df = df[df.columns[1:]]
#df.to_csv("earthquakes_filtred.csv", index=False)

print(df['properties.magType'].unique())