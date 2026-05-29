import requests
import os
import json
from tqdm import tqdm

#pulls all pokemon's data and puts into a folder called pokeapi_data
#each pokemon has its own folder with its data
def pull_pokemon_data_by_index(
    species_url: str = "https://pokeapi.co/api/v2/pokemon-species/", 
    data_folder: str = "pokeapi_data",
    start_index: int = 1,
    end_index: int = 1025):
    
    """
    Args:
        species_url (str): The URL for the Pokémon species API.
        data_folder (str, optional): The folder to save the data to.
        start_index (int): The starting index for the data pull.
        end_index (int): The ending index for the data pull.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created directory: {data_folder}")

    print(f"Starting data pull for species {start_index} to {end_index}...")

    # Using tqdm for a progress bar
    for i in tqdm(range(start_index, end_index + 1), desc="Downloading Pokémon Data"):
        try:
            # Construct species URL (ignoring the default arg if it ends in /1/)
            curr_species_url = f"{species_url.rstrip('/')}/{i}/"
            species_response = requests.get(curr_species_url)
            species_response.raise_for_status()
            species_data = species_response.json()
            
            species_name = species_data['name']
            species_dir = os.path.join(data_folder, f"{i}_{species_name}")
            
            if not os.path.exists(species_dir):
                os.makedirs(species_dir)
            
            # Save species data
            with open(os.path.join(species_dir, "species.json"), "w") as f:
                json.dump(species_data, f, indent=4)
                
            # Pull data for each variety (default, megas, regionals, etc.)
            for variety in species_data['varieties']:
                variety_name = variety['pokemon']['name']
                variety_url = variety['pokemon']['url']
                
                variety_response = requests.get(variety_url)
                variety_response.raise_for_status()
                variety_data = variety_response.json()
                
                # Save variety data
                variety_filename = f"{variety_name}.json"
                with open(os.path.join(species_dir, variety_filename), "w") as f:
                    json.dump(variety_data, f, indent=4)

        except Exception as e:
            # Using tqdm.write to avoid messing up the progress bar
            tqdm.write(f"Failed to fetch data for Pokémon species {i}: {e}")

    print("Data pull complete.")





if __name__ == "__main__":
    pull_pokemon_data_by_index()