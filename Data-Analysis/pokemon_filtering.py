import os
import json

# Mapping from version group names to generation numbers
# Reused from pokeapi_visualizers.py
GAME_TO_GEN = {
    "red-blue": 1, "yellow": 1, "red-green-japan": 1, "blue-japan": 1,
    "gold-silver": 2, "crystal": 2,
    "ruby-sapphire": 3, "emerald": 3, "firered-leafgreen": 3, "colosseum": 3, "xd": 3,
    "diamond-pearl": 4, "platinum": 4, "heartgold-soulsilver": 4,
    "black-white": 5, "black-2-white-2": 5,
    "x-y": 6, "omega-ruby-alpha-sapphire": 6,
    "sun-moon": 7, "ultra-sun-ultra-moon": 7, "lets-go-pikachu-lets-go-eevee": 7,
    "sword-shield": 8, "brilliant-diamond-shining-pearl": 8, "legends-arceus": 8, "the-isle-of-armor": 8, "the-tundra-beat": 8,
    "scarlet-violet": 9, "the-teal-mask": 9, "the-indigo-disk": 9
}

import pandas as pd

def get_effective_generation(variety_data, species_gen):
    """
    Determine the 'effective generation' of a Pokemon variety.
    Reused logic from pokeapi_visualizers.py.
    """
    variety_name = variety_data.get("name", "").lower()
    effective_gen = species_gen
    
    if not variety_data.get("is_default", True):
        # Suffix Overrides
        if any(s in variety_name for s in ["-alola"]): effective_gen = 7
        elif any(s in variety_name for s in ["-galar", "-hisui", "-gmax"]): effective_gen = 8
        elif any(s in variety_name for s in ["-paldea"]): effective_gen = 9
        elif any(s in variety_name for s in ["-mega", "-primal"]): effective_gen = 6
        else:
            # Fallback: Detect debut from move data
            move_details = variety_data.get("moves", [])
            detected_gens = []
            for move in move_details:
                for detail in move.get("version_group_details", []):
                    vg_name = detail.get("version_group", {}).get("name", "")
                    if vg_name in GAME_TO_GEN:
                        detected_gens.append(GAME_TO_GEN[vg_name])
            
            if detected_gens:
                effective_gen = min(detected_gens)
    
    return effective_gen

def identify_pokemon_by_criteria(generations, typing, order_exact=False, loose=False, data_dir=None):
    """
    Identify which pokemon satisfy the given generation and typing criteria.
    
    Args:
        generations (list): Array of generation numbers (e.g., [1, 2]).
        typing (tuple): Tuple of size 1 or 2 (e.g., ('Water',) or ('Water', 'Flying')).
        order_exact (bool): If True, order of types in dual-typing must match exactly.
        loose (bool): If True and a mono-type is passed, includes dual-type pokemon containing that type.
        data_dir (str): Path to the pokeapi_data directory.
        
    Returns:
        tuple: (count (int), dataframe (pd.DataFrame))
    """
    if data_dir is None:
        # Try to find pokeapi_data relative to this script
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_path, "..", "pokeapi_data")
        
        # If not found, try current directory
        if not os.path.exists(data_dir):
            data_dir = "pokeapi_data"
            
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    results = []
    gen_filter = set(generations) if generations else set()
    
    # Normalize typing to lowercase for comparison
    is_dual_target = False
    target_types = None
    if typing:
        target_types = tuple(t.lower() for t in typing)
        is_dual_target = len(target_types) == 2

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        species_file = os.path.join(folder_path, "species.json")
        if not os.path.exists(species_file):
            continue
            
        with open(species_file, "r") as f:
            try:
                species_data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        species_id = species_data.get("id")
        gen_url = species_data.get("generation", {}).get("url", "")
        try:
            species_gen = int(gen_url.strip("/").split("/")[-1])
        except (ValueError, IndexError):
            species_gen = 0
        
        for variety_file in sorted(os.listdir(folder_path)):
            if variety_file == "species.json" or not variety_file.endswith(".json"):
                continue
                
            variety_path = os.path.join(folder_path, variety_file)
            with open(variety_path, "r") as f:
                try:
                    variety_data = json.load(f)
                except json.JSONDecodeError:
                    continue
            
            # Filter by generation
            eff_gen = get_effective_generation(variety_data, species_gen)
            if gen_filter and eff_gen not in gen_filter:
                continue
                
            # Extract types
            t_list = variety_data.get("types", [])
            t_list = sorted(t_list, key=lambda x: x['slot'])
            current_types = tuple(t['type']['name'].lower() for t in t_list)
            
            # Check typing criteria
            match = False
            
            if target_types is None:
                # No typing filter, include all
                match = True
            elif loose and not is_dual_target:
                # Loose mono-type: matches if the type is present at all
                if target_types[0] in current_types:
                    match = True
            else:
                # Strict matching (or dual-type query)
                if len(current_types) == len(target_types):
                    if not is_dual_target:
                        if current_types[0] == target_types[0]:
                            match = True
                    else:
                        if order_exact:
                            if current_types == target_types:
                                match = True
                        else:
                            if set(current_types) == set(target_types):
                                match = True
            
            if match:
                results.append({
                    "pokedex_number": species_id,
                    "name": variety_data["name"].capitalize(),
                    "type1": current_types[0].capitalize(),
                    "type2": current_types[1].capitalize() if len(current_types) > 1 else "None",
                    "generation": eff_gen
                })
                        
    df = pd.DataFrame(results)
    return len(df), df

if __name__ == "__main__":
    # Example usage:
    # All generations, Water/Flying types
    gens = []
    types = ('Water', 'Flying')
    
    count, df = identify_pokemon_by_criteria(gens, types, order_exact=False)
    print(f"\n--- All Generations, Types {types}, order_exact=False ---")
    print(f"Total count: {count}")
    print(df.head().to_string(index=False)) # Show first few
    
    # Mono-type test (Strict)
    types_mono = ('Fire',)
    count_mono, df_mono = identify_pokemon_by_criteria([1], types_mono, loose=False)
    print(f"\n--- Generation 1, Types {types_mono} (Strict) ---")
    print(f"Total count: {count_mono}")
    print(df_mono.to_string(index=False))
    
    # Mono-type test (Loose)
    count_loose, df_loose = identify_pokemon_by_criteria([1], types_mono, loose=True)
    print(f"\n--- Generation 1, Types {types_mono} (Loose) ---")
    print(f"Total count: {count_loose}")
    print(df_loose.to_string(index=False))
