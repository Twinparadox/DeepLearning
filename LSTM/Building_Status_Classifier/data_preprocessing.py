import pandas as pd
import numpy as np

df = pd.read_excel('data/BuildingTag_Napa.xlsx', encoding='utf-8')
df.rename(columns = {'Orignial Placard':'Placard'}, inplace=True)
df = df.loc[:,['Placard','Updated Placard','Description']]

# Remove Missing Value
df['Placard'] = df['Placard'].replace(str('TAPED OFF'), np.nan)

df['Updated Placard'] = df['Updated Placard'].replace([np.nan, str(' ')], 1)
df = df[df['Updated Placard'] == 1]
df['Description'] = df['Description'].replace(str(' '), np.nan)
df = df.dropna(subset=['Placard', 'Description'])
df.drop('Updated Placard', axis=1, inplace=True)

# Capitalize for Labeling
df['Placard'] = df['Placard'].str.capitalize()

# Labelling,
# 0: Green
# 1: Yellow
# 2: Red
df['Placard'] = df['Placard'].replace(['Green','Yellow','Red'],[0,1,2])

df.to_csv('data/BuildingTag_Napa.csv',
          index=False,
          encoding='utf-8')