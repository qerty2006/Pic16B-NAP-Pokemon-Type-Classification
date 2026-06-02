import requests
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_complete(species_dir, species_data):
    expected = {v['pokemon']['name'] + '.json' for v in species_data['varieties']}
    existing = {f for f in os.listdir(species_dir) if f.endswith('.json') and f != 'species.json'}
    return expected == existing


def fetch_pokemon(i, species_url, data_folder):
    try:
        species_dir_glob = [
            d for d in os.listdir(data_folder)
            if d.startswith(f"{i}_") and os.path.isdir(os.path.join(data_folder, d))
        ]

        # load cached species.json if available
        species_data = None
        species_dir = None
        if species_dir_glob:
            species_dir = os.path.join(data_folder, species_dir_glob[0])
            species_json = os.path.join(species_dir, "species.json")
            if os.path.exists(species_json):
                with open(species_json) as f:
                    species_data = json.load(f)

        # fetch species if not cached
        if species_data is None:
            response = requests.get(f"{species_url.rstrip('/')}/{i}/")
            response.raise_for_status()
            species_data = response.json()
            species_dir = os.path.join(data_folder, f"{i}_{species_data['name']}")
            os.makedirs(species_dir, exist_ok=True)
            with open(os.path.join(species_dir, "species.json"), "w") as f:
                json.dump(species_data, f, indent=4)

        if is_complete(species_dir, species_data):
            return

        # fetch only missing varieties
        existing = set(os.listdir(species_dir))
        for variety in species_data['varieties']:
            variety_name = variety['pokemon']['name']
            if f"{variety_name}.json" in existing:
                continue
            variety_response = requests.get(variety['pokemon']['url'])
            variety_response.raise_for_status()
            with open(os.path.join(species_dir, f"{variety_name}.json"), "w") as f:
                json.dump(variety_response.json(), f, indent=4)

    except Exception as e:
        tqdm.write(f"Failed to fetch data for Pokemon species {i}: {e}")


def pull_pokemon_data_by_index(
    species_url: str = "https://pokeapi.co/api/v2/pokemon-species/",
    data_folder: str = "Classification/pokeapi_data",
    start_index: int = 1,
    end_index: int = 1025,
    workers: int = 20):

    os.makedirs(data_folder, exist_ok=True)
    print(f"Starting data pull for species {start_index} to {end_index} ({workers} threads)...")

    indices = range(start_index, end_index + 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_pokemon, i, species_url, data_folder): i for i in indices}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading Pokemon Data"):
            pass

    print("Data pull complete.")


if __name__ == "__main__":
    pull_pokemon_data_by_index()