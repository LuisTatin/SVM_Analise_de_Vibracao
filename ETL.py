# Bibliotecas
import numpy as np
import scipy.io
from pathlib import Path
import pandas as pd

#%%
#Funções

def load_mat_file(file_path):
    """
    Carrega um arquivo .mat e retorna seu conteúdo.
    :param file_path: Caminho para o arquivo .mat
    :return: Dicionário com os dados do arquivo
    """
    try:
        data = scipy.io.loadmat(file_path)
        return data
    except Exception as e:
        print(f"Erro ao carregar o arquivo .mat: {e}")
        return None

def arquivos(folder_path):
    pasta = Path(folder_path)
    lista_arquivos = []
    for i in pasta.iterdir():
        if i.is_file():
            lista_arquivos.append(i.name)
    return lista_arquivos

#%%
#Faulty bd
pasta_faulty = 'archive/Faulty'
lista_arquivos = arquivos(pasta_faulty)

content = []
for i in lista_arquivos:
    file_path = i  # Substitua pelo caminho do seu arquivo
    mat_data = load_mat_file('archive/Faulty/' + file_path)

    for x in mat_data['H']:
        content.append(x)


bd = pd.DataFrame(content)
bd.columns = ['X', 'Y', 'Z']
bd.to_csv('Faulty.csv', columns=['X', 'Y', 'Z'], index=False)

#%%
#Healthy bd
pasta_healthy = 'archive/Healthy'
lista_arquivos = arquivos(pasta_healthy)

content = []
for i in lista_arquivos:
    file_path = i  # Substitua pelo caminho do seu arquivo
    mat_data = load_mat_file('archive/Healthy/' + file_path)

    for x in mat_data['H']:
        content.append(x)


bd = pd.DataFrame(content)
bd.columns = ['X', 'Y', 'Z']
bd.to_csv('Healthy.csv', columns=['X', 'Y', 'Z'], index=False)

#%%
#Adicionar 0 no final do Healthy
bd_healthy = pd.read_csv('Healthy.csv')

lista_1 = [0 for x in range(len(bd_healthy))]
bd_healthy['Failure'] = lista_1

#print(bd_healthy.head())

#%%
#Adicionar 1 no final do Faulty

bd_Faulty = pd.read_csv('Faulty.csv')

lista_1 = [1 for x in range(len(bd_Faulty))]
bd_Faulty['Failure'] = lista_1

#print(bd_Faulty.head())

#%%
#Juntar os 2
bd_final = pd.concat([bd_healthy, bd_Faulty], axis=0)
#print(bd_final.head())
bd_final.to_csv('Vibration.csv', columns=['X', 'Y', 'Z', 'Failure'], index=False)