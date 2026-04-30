import pandas as pd
import os
import json
from pathlib import Path
from PIL import Image
pokemon_data=[]
counter=0
base_dir = Path(__file__).parent
sprites=base_dir/"split_sprites"
pokeapi_data=base_dir/"pokeapi_data"

pokemon_folders = sorted([f for f in pokeapi_data.iterdir()])

while counter<1025:
    curr_folder = pokemon_folders[counter]
    for file in curr_folder.iterdir():
        if file.name!="species.json":
            with open(file, 'r') as f:
                data = json.load(f)
                print(f"Processing {file.name}...")

                pokemon_types = [item['type']['name'] for item in data.get('types')]

                # Save the filename (without .json) and the list of types
                pokemon_data.append({
                    'name': file.stem,
                    'types': pokemon_types,
                    "ID": curr_folder.stem.split("_")[0],
                })
    counter+=1

df = pd.DataFrame(pokemon_data)

types=all_possible_types = ['normal', 'fire', 'water', 'grass', 'electric', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'steel', 'dark', 'fairy']

for type in types:
    df[type] = df['types'].apply(lambda x: 1 if type in x else 0)

df_final = df.drop(columns=['types'])

sprite_folder=sorted([f for f in sprites.iterdir()])
counter=0

sprite_data=[]
while counter<len(sprite_folder):
    curr_folder = sprite_folder[counter]
    for file in curr_folder.iterdir():
       sprite=str(file)
       ID=curr_folder.name.split("-")[0]

       sprite_data.append({"Sprite": sprite, "ID": ID, "Sprite_ID": str(ID)+"_"+str(file.stem)})
    counter+=1


df2 = pd.DataFrame(sprite_data)


df_final=pd.merge(df2, df_final, left_on="ID", right_on="ID", how="left")



df_final = df_final.drop(columns=['ID'])

#GETTING RID OF ALL OUTSIDE THE BASE 1025
df_final = df_final.dropna()











