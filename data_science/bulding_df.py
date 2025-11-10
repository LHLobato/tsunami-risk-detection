import requests
import pandas as pd
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta 
import time

data_fim_total = datetime.now() 
data_inicio_total = data_fim_total - relativedelta(years=1)

print(f"Iniciando coleta em lotes de {data_inicio_total.date()} até {data_fim_total.date()}...")


lista_dataframes_mes = []
data_loop = data_inicio_total

while data_loop < data_fim_total:
    

    data_inicio_lote = data_loop
    data_fim_lote = data_loop + relativedelta(months=1)
    

    inicio_str = data_inicio_lote.strftime('%Y-%m-%d')
    fim_str = data_fim_lote.strftime('%Y-%m-%d')

    print(f"Baixando lote: {inicio_str} a {fim_str} ...", end='')

    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        "format=geojson"
        f"&starttime={inicio_str}"
        f"&endtime={fim_str}"
        "&minmagnitude=1"
    )

    try:
        request = requests.get(url, timeout=60)
        request.raise_for_status() 
        data = request.json()

        df_lote = pd.json_normalize(data['features'])
        
        lista_dataframes_mes.append(df_lote)
        
        print(f" OK ({len(df_lote)} eventos)")

    except requests.exceptions.HTTPError as err:
        print(f" FALHA NO LOTE (Erro HTTP): {err}")
    except requests.exceptions.JSONDecodeError:
        print(" FALHA NO LOTE (Não recebeu JSON)")
    
    data_loop = data_fim_lote

    time.sleep(2) 


print("\nDownload de todos os lotes concluído.")
print("Juntando os arquivos...")

df_final_anual = pd.concat(lista_dataframes_mes, ignore_index=True)

df_final_anual.to_csv('earthquakes_anual_completo.csv', index=False)

print(f"CSV final salvo com sucesso!")
print(f"Total de eventos coletados: {len(df_final_anual)}")