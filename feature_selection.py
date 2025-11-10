import pandas as pd

df = pd.read_csv('earthquakes.csv')

features = ["properties.mag", "properties.sig", "properties.magType","properties.type", "geometry.coordinates", "properties.tsunami"]

clean_df = df[features]
clean_df.to_csv("clean_earth.csv", index=False)