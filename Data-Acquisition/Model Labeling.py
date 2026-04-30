import pandas as pd
import os
import json
from pathlib import Path
from PIL import Image

#Setting up paths and counter
pokemon_data=[]
counter=0
base_dir = Path(__file__).parent
sprites=base_dir/"split_sprites"
pokeapi_data=base_dir/"pokeapi_data"

#folders of pokeapi data
pokemon_folders = sorted([f for f in pokeapi_data.iterdir()])


#loop through each folder
while counter<1025:
    curr_folder = pokemon_folders[counter]

    #for every file not species, open and get types
    for file in curr_folder.iterdir():
        if file.name!="species.json":
            with open(file, 'r') as f:
                data = json.load(f)
                print(f"Processing {file.name}...")

                pokemon_types = [item['type']['name'] for item in data.get('types')]

                # Save the filename and the list of types
                pokemon_data.append({
                    'name': file.stem,
                    'types': pokemon_types,
                    "ID": curr_folder.stem.split("_")[0],
                })
    counter+=1

#create df with data
df = pd.DataFrame(pokemon_data)

#all possible types
types=all_possible_types = ['normal', 'fire', 'water', 'grass', 'electric', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'steel', 'dark', 'fairy']

#label 1 or 0 (true or false) for each type
for type in types:
    df[type] = df['types'].apply(lambda x: 1 if type in x else 0)

#drop old types column
df_final = df.drop(columns=['types'])

#setup sprite folder
sprite_folder=sorted([f for f in sprites.iterdir()])
counter=0


#loop through each sprite png and gather data
sprite_data=[]
while counter<len(sprite_folder):
    curr_folder = sprite_folder[counter]
    for file in curr_folder.iterdir():
       sprite=str(file)
       ID=curr_folder.name.split("-")[0]

#this has sprite location, base ID, and Sprite ID
       sprite_data.append({"Sprite": sprite, "ID": ID, "Sprite_ID": str(ID)+"_"+str(file.stem)})
    counter+=1

#create a 2nd dataframe
df2 = pd.DataFrame(sprite_data)

#merge together according to sprite ID to have data for individual sprites
df_final=pd.merge(df2, df_final, left_on="ID", right_on="ID", how="left")


#drop ID column as Sprite_ID is more specific
df_final = df_final.drop(columns=['ID'])

#GETTING RID OF ALL OUTSIDE THE BASE 1025
df_final = df_final.dropna()











