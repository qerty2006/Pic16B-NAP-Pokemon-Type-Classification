#!/usr/bin/env python3
"""
Pokemon HTML Viewer Generator
This script takes in a range of Pokemon indices or a pandas DataFrame (such as those generated
by pokemon_filtering.py) and outputs a premium, interactive, highly-styled HTML visualizer.

THIS WAS MADE WITH GEMINI. Nish made this to look at various pokemon sprites more easily, not for general purposes.
I thought it would be interesting to show.
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd

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

def get_effective_generation(variety_data, species_gen):
    """
    Determine the 'effective generation' of a Pokemon variety.
    Self-contained implementation.
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


# Color system for Pokemon types
TYPE_COLORS = {
    "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0", "Electric": "#F7D02C",
    "Grass": "#7AC74C", "Ice": "#96D9D6", "Fighting": "#C22E28", "Poison": "#A33EA1",
    "Ground": "#E2BF65", "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
    "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC", "Steel": "#B7B7CE",
    "Fairy": "#D685AD", "Dark": "#705746", "None": "#68A090"
}

TYPE_COLORS_RGBA = {
    "Normal": "rgba(168, 167, 122, 0.2)", "Fire": "rgba(238, 129, 48, 0.2)", "Water": "rgba(99, 144, 240, 0.2)",
    "Electric": "rgba(247, 208, 44, 0.2)", "Grass": "rgba(122, 199, 76, 0.2)", "Ice": "rgba(150, 217, 214, 0.2)",
    "Fighting": "rgba(194, 46, 40, 0.2)", "Poison": "rgba(163, 62, 161, 0.2)", "Ground": "rgba(226, 191, 101, 0.2)",
    "Flying": "rgba(169, 143, 243, 0.2)", "Psychic": "rgba(249, 85, 135, 0.2)", "Bug": "rgba(166, 185, 26, 0.2)",
    "Rock": "rgba(182, 161, 54, 0.2)", "Ghost": "rgba(115, 87, 151, 0.2)", "Dragon": "rgba(111, 53, 252, 0.2)",
    "Steel": "rgba(183, 183, 206, 0.2)", "Fairy": "rgba(214, 133, 173, 0.2)", "Dark": "rgba(112, 87, 70, 0.2)",
    "None": "rgba(104, 160, 144, 0.2)"
}

def get_pokeapi_data_dir():
    """Helper to locate pokeapi_data directory absolutely."""
    # Try finding it relative to the script location first (script is in Data-Analysis)
    script_dir = Path(__file__).resolve().parent
    relative_path = script_dir.parent / "Classification" / "pokeapi_data"
    if relative_path.exists() and relative_path.is_dir():
        return relative_path
        
    # Fallback to absolute workspace path
    absolute_workspace = Path("/Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Classification/pokeapi_data")
    if absolute_workspace.exists() and absolute_workspace.is_dir():
        return absolute_workspace
        
    raise FileNotFoundError("Could not locate pokeapi_data directory.")


def fetch_pokemon_details(pokedex_number, name_or_variety=None, data_dir=None):
    """
    Fetch comprehensive details for a Pokemon from its JSON folders.
    Enriches basic DataFrame entries with stats, high-res official artwork, weight, height.
    """
    if data_dir is None:
        data_dir = get_pokeapi_data_dir()
        
    data_dir = Path(data_dir)
    
    # 1. Locate the species folder (starts with "{pokedex_number}_")
    folder_path = None
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith(f"{pokedex_number}_"):
            folder_path = item
            break
            
    if not folder_path:
        return None
        
    # 2. Find the variety json file
    variety_json = None
    species_name = folder_path.name.split("_", 1)[1]
    
    # If a variety name is provided, try that first
    if name_or_variety:
        # Variety name might have different casing/suffixes
        target_name = name_or_variety.lower().replace(" ", "-")
        candidate = folder_path / f"{target_name}.json"
        if candidate.exists():
            variety_json = candidate
            
    # Fallback/Default search
    if not variety_json or not variety_json.exists():
        # Find any json ending with variety name or not species.json
        jsons = [f for f in folder_path.glob("*.json") if f.name != "species.json"]
        if jsons:
            # Prefer the default variety if it matches the folder/species name
            default_json = folder_path / f"{species_name}.json"
            variety_json = default_json if default_json.exists() else jsons[0]
            
    if not variety_json or not variety_json.exists():
        return None
        
    # 3. Parse JSON files
    try:
        with open(variety_json, "r") as f:
            data = json.load(f)
            
        species_file = folder_path / "species.json"
        species_data = {}
        if species_file.exists():
            with open(species_file, "r") as f:
                species_data = json.load(f)
    except Exception as e:
        print(f"Error parsing JSON for #{pokedex_number}: {e}")
        return None
        
    # Extract types
    types = sorted(data.get("types", []), key=lambda x: x["slot"])
    type1 = types[0]["type"]["name"].capitalize() if len(types) > 0 else "None"
    type2 = types[1]["type"]["name"].capitalize() if len(types) > 1 else "None"
    
    # Extract sprites (prefer official artwork, fall back to front_default)
    sprites = data.get("sprites", {})
    official_artwork = sprites.get("other", {}).get("official-artwork", {}).get("front_default")
    sprite_url = official_artwork or sprites.get("front_default") or ""
    
    # Extract stats
    stats = {s["stat"]["name"]: s["base_stat"] for s in data.get("stats", [])}
    
    # Extract effective generation
    species_gen_url = species_data.get("generation", {}).get("url", "")
    try:
        species_gen = int(species_gen_url.strip("/").split("/")[-1])
    except (ValueError, IndexError):
        species_gen = 0
    eff_gen = get_effective_generation(data, species_gen)
    
    # 4. Find all split sprite directories associated with this ID
    split_dir = Path(data_dir).parent / "split_sprites"
    pokerogue_sprites = []
    if split_dir.exists() and split_dir.is_dir():
        for item in split_dir.iterdir():
            if item.is_dir():
                name_part = item.name
                if name_part == str(pokedex_number) or name_part.startswith(f"{pokedex_number}-"):
                    if name_part == str(pokedex_number):
                        label = "Default"
                    else:
                        label = name_part.split("-", 1)[1].replace("-", " ").capitalize()
                    
                    # Gather the individual split PNG frames (e.g. 0001.png, 0002.png) sorted
                    frames = sorted([f.name for f in item.glob("*.png")])
                    
                    if frames:
                        pokerogue_sprites.append({
                            "label": label,
                            "folder_name": item.name,
                            "preview_url": f"Classification/split_sprites/{item.name}/{frames[0]}",
                            "frames": [f"Classification/split_sprites/{item.name}/{f}" for f in frames]
                        })
    
    # Sort pokerogue_sprites so Default comes first
    pokerogue_sprites.sort(key=lambda x: 0 if x["label"] == "Default" else 1)


    return {
        "id": pokedex_number,
        "name": data.get("name", "").capitalize(),
        "type1": type1,
        "type2": type2,
        "sprite_url": sprite_url,
        "generation": eff_gen,
        "height": data.get("height", 0) / 10.0, # Convert to meters
        "weight": data.get("weight", 0) / 10.0, # Convert to kg
        "stats": {
            "hp": stats.get("hp", 0),
            "attack": stats.get("attack", 0),
            "defense": stats.get("defense", 0),
            "sp_atk": stats.get("special-attack", 0),
            "sp_def": stats.get("special-defense", 0),
            "speed": stats.get("speed", 0),
        },
        "pokerogue_sprites": pokerogue_sprites
    }


def generate_html(indices_or_df, output_file="pokemon_viewer.html", data_dir=None):
    """
    Generate an ultra-premium HTML page visualizing the Pokemon.
    
    Args:
        indices_or_df: Either a range/list of pokedex numbers, or a pandas DataFrame.
        output_file: Target HTML output filepath.
        data_dir: Source pokeapi_data directory.
    """
    if data_dir is None:
        data_dir = get_pokeapi_data_dir()
        
    pokemon_list = []
    
    if isinstance(indices_or_df, pd.DataFrame):
        print(f"Processing DataFrame with {len(indices_or_df)} rows...")
        for _, row in indices_or_df.iterrows():
            # Check for pokedex number column names (could be pokedex_number or id)
            idx = row.get("pokedex_number") or row.get("id") or row.get("pokedex_id")
            name = row.get("name")
            if pd.notna(idx):
                details = fetch_pokemon_details(int(idx), name, data_dir)
                if details:
                    # Keep customized type values if DataFrame is pre-filtered/modified
                    if "type1" in row: details["type1"] = row["type1"].capitalize()
                    if "type2" in row: details["type2"] = row["type2"].capitalize() if pd.notna(row["type2"]) else "None"
                    if "generation" in row: details["generation"] = int(row["generation"])
                    pokemon_list.append(details)
    else:
        # Assumed to be list, range, or iterable of integers
        indices = list(indices_or_df)
        print(f"Processing range of {len(indices)} Pokemon indices...")
        for idx in indices:
            details = fetch_pokemon_details(idx, data_dir=data_dir)
            if details:
                pokemon_list.append(details)
                
    if not pokemon_list:
        print("No Pokemon data found. The generated HTML will be empty.")
        
    # Sort by ID
    pokemon_list.sort(key=lambda x: x["id"])
    
    # Render premium HTML
    html_content = build_html_template(pokemon_list)
    
    output_path = Path(output_file)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Successfully generated visualizer HTML: {output_path.resolve()}")
    return output_path

def build_html_template(pokemon_list):
    """Build the premium, gorgeous visualizer HTML string with dynamic JS filters and type analysis charts."""
    pokemon_json = json.dumps(pokemon_list, indent=2)
    type_colors_json = json.dumps(TYPE_COLORS)
    type_colors_rgba_json = json.dumps(TYPE_COLORS_RGBA)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pokédex Interactive Analysis & Visualizer</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Custom Modern Styling -->
    <style>
        :root {{
            --bg-dark: #0a0e17;
            --bg-card: rgba(18, 26, 44, 0.6);
            --border-light: rgba(255, 255, 255, 0.08);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-glow: rgba(99, 144, 240, 0.35);
            --transition-smooth: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        html {{
            overflow-y: auto;
        }}

        body {{
            background-color: var(--bg-dark);
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            padding: 2rem;
            overflow-x: hidden;
            overflow-y: auto;
            background-image: 
                radial-gradient(at 10% 20%, rgba(99, 144, 240, 0.08) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(247, 208, 44, 0.05) 0px, transparent 50%);
        }}
        
        header {{
            max-width: 1400px;
            margin: 0 auto 3rem auto;
            text-align: center;
            position: relative;
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #fff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 4px 20px rgba(165, 180, 252, 0.15);
        }}
        
        header p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        /* Stats Dashboard Panel */
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto 3rem auto;
            background: var(--bg-card);
            border: 1px solid var(--border-light);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }}
        
        .dash-card {{
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        
        .dash-card h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }}
        
        .dash-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: #fff;
        }}
        
        .type-pill-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        /* Filters & Search Control Panel */
        .control-panel {{
            max-width: 1400px;
            margin: 0 auto 2rem auto;
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            justify-content: space-between;
            align-items: center;
        }}
        
        .search-box {{
            flex: 1;
            min-width: 320px;
            position: relative;
        }}
        
        .search-box input {{
            width: 100%;
            background: rgba(18, 26, 44, 0.8);
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.1rem 1.5rem 1.1rem 3rem;
            color: #fff;
            font-size: 1rem;
            font-family: inherit;
            outline: none;
            transition: var(--transition-smooth);
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        }}
        
        .search-box input:focus {{
            border-color: rgba(99, 144, 240, 0.6);
            box-shadow: 0 0 20px rgba(99, 144, 240, 0.25);
        }}
        
        .search-box::before {{
            content: "🔍";
            position: absolute;
            left: 1.2rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.1rem;
            opacity: 0.6;
        }}
        
        .filter-group {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}
        
        .custom-select {{
            background: rgba(18, 26, 44, 0.8);
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.1rem 2.5rem 1.1rem 1.5rem;
            color: #fff;
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            cursor: pointer;
            appearance: none;
            transition: var(--transition-smooth);
            background-image: url("data:image/svg+xml;utf8,<svg fill='white' height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'><path d='M7 10l5 5 5-5z'/><path d='M0 0h24v24H0z' fill='none'/></svg>");
            background-repeat: no-repeat;
            background-position: right 0.75rem center;
            background-size: 1.25rem;
            min-width: 160px;
        }}
        
        .custom-select:focus {{
            border-color: rgba(99, 144, 240, 0.6);
        }}
        
        /* Pokemon Grid Layout */
        .pokemon-grid {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 2rem;
        }}
        
        /* Pokemon Card Component */
        .pokemon-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-light);
            border-radius: 24px;
            padding: 2rem;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            backdrop-filter: blur(10px);
            transition: var(--transition-smooth);
            cursor: pointer;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        
        .pokemon-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 140px;
            background: var(--card-glow-bg, rgba(99, 144, 240, 0.05));
            mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%);
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%);
            z-index: 0;
            transition: var(--transition-smooth);
        }}
        
        .pokemon-card:hover {{
            transform: translateY(-8px);
            border-color: var(--card-glow-border, rgba(99, 144, 240, 0.3));
            box-shadow: 0 20px 40px var(--card-glow-shadow, rgba(99, 144, 240, 0.15));
        }}
        
        .pokemon-id {{
            position: absolute;
            top: 1.25rem;
            right: 1.5rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            opacity: 0.25;
            letter-spacing: 0.05em;
        }}
        
        .pokemon-gen {{
            position: absolute;
            top: 1.25rem;
            left: 1.5rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(255,255,255,0.06);
            padding: 0.3rem 0.6rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        
        .pokemon-sprite-container {{
            position: relative;
            z-index: 1;
            margin: 1.5rem 0 1rem 0;
            height: 160px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .pokemon-sprite {{
            width: 140px;
            height: 140px;
            object-fit: contain;
            transition: var(--transition-smooth);
            filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
        }}
        
        .pokemon-card:hover .pokemon-sprite {{
            transform: scale(1.12) translateY(-4px);
            filter: drop-shadow(0 15px 25px rgba(0,0,0,0.4));
        }}
        
        .pokemon-name {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #fff;
            letter-spacing: -0.02em;
            text-align: center;
            z-index: 1;
        }}
        
        .type-badge-container {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            z-index: 1;
        }}
        
        .type-badge {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.4rem 0.9rem;
            border-radius: 50px;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        /* Stats Panel (in Card) */
        .stats-panel {{
            width: 100%;
            margin-top: auto;
            border-top: 1px solid var(--border-light);
            padding-top: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            z-index: 1;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        
        .stat-bar-container {{
            flex: 1;
            height: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
            margin: 0 0.75rem;
            overflow: hidden;
        }}
        
        .stat-bar {{
            height: 100%;
            border-radius: 3px;
            background: var(--primary-type-color, var(--text-secondary));
            width: 0%;
            transition: width 1s ease-out;
        }}
        
        .stat-value {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            width: 24px;
            text-align: right;
        }}
        
        .dimensions-row {{
            display: flex;
            justify-content: space-around;
            width: 100%;
            border-top: 1px solid var(--border-light);
            padding-top: 0.9rem;
            margin-top: 0.9rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            z-index: 1;
        }}
        
        .dimension-item {{
            text-align: center;
        }}
        
        .dimension-val {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            color: #fff;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }}
        
        /* Modal for Details */
        .modal {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(5, 7, 12, 0.85);
            backdrop-filter: blur(20px);
            z-index: 100;
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            pointer-events: none;
            transition: var(--transition-smooth);
            padding: 1.5rem;
        }}
        
        .modal.active {{
            opacity: 1;
            pointer-events: auto;
        }}
        
        .modal-content {{
            background: #0f172a;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 32px;
            max-width: 650px;
            width: 100%;
            padding: 2.5rem;
            position: relative;
            transform: scale(0.9) translateY(20px);
            transition: var(--transition-smooth);
            box-shadow: 0 25px 60px rgba(0,0,0,0.5);
        }}
        
        .modal.active .modal-content {{
            transform: scale(1) translateY(0);
        }}
        
        .close-btn {{
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            background: rgba(255,255,255,0.06);
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            color: #fff;
            font-size: 1.25rem;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: var(--transition-smooth);
        }}
        
        .close-btn:hover {{
            background: rgba(255,255,255,0.15);
            transform: rotate(90deg);
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            h1 {{
                font-size: 2.5rem;
            }}
            .pokemon-grid {{
                grid-template-columns: 1fr;
            }}
            .modal-content {{
                padding: 1.5rem;
            }}
        }}
    </style>
</head>
<body>

    <header>
        <h1>Pokédex Analysis Visualizer</h1>
        <p>A premium interactive data visualizer for Pokemon species, types, and analytics.</p>
    </header>

    <!-- Interactive Dashboard Panel -->
    <section class="dashboard">
        <div class="dash-card">
            <h3>Total Pokemon</h3>
            <div class="value" id="total-count">0</div>
        </div>
        <div class="dash-card">
            <h3>Primary Types</h3>
            <div class="value" id="unique-types">0</div>
        </div>
        <div class="dash-card">
            <h3>PokeRogue Sprites Found</h3>
            <div class="value" id="total-sprites">0</div>
        </div>
    </section>

    <!-- Filters and Searches -->
    <section class="control-panel">
        <div class="search-box">
            <input type="text" id="search-input" placeholder="Search by name, ID or type...">
        </div>
        <div class="filter-group">
            <select class="custom-select" id="type-filter">
                <option value="All">Type 1: All</option>
            </select>
            <select class="custom-select" id="type2-filter">
                <option value="All">Type 2: All</option>
            </select>
            <select class="custom-select" id="gen-filter">
                <option value="All">All Gens</option>
            </select>
            <select class="custom-select" id="sort-filter">
                <option value="id-asc">Sort by ID (Asc)</option>
                <option value="id-desc">Sort by ID (Desc)</option>
                <option value="name-asc">Sort by Name (A-Z)</option>
                <option value="bst-desc">Sort by BST (High-Low)</option>
            </select>
        </div>
    </section>

    <!-- Pokemon Grid -->
    <main class="pokemon-grid" id="pokemon-grid"></main>

    <!-- Sentinel for Infinite Scroll -->
    <div id="infinite-scroll-sentinel" style="height: 40px; margin-top: 2rem;"></div>

    <!-- Premium Modal -->
    <div class="modal" id="details-modal">
        <div class="modal-content">
            <button class="close-btn" id="close-modal">✕</button>
            <div id="modal-body"></div>
        </div>
    </div>

    <!-- Data Injection & Interactive JS -->
    <script>
        const POKEMON_DATA = {pokemon_json};
        const TYPE_COLORS = {type_colors_json};
        const TYPE_COLORS_RGBA = {type_colors_rgba_json};

        // Render stats & filter choices
        function initDashboard() {{
            const types = new Set();
            let totalSprites = 0;
            
            POKEMON_DATA.forEach(p => {{
                if (p.type1) types.add(p.type1);
                if (p.type2 && p.type2 !== "None") types.add(p.type2);
                if (p.pokerogue_sprites) {{
                    totalSprites += p.pokerogue_sprites.length;
                }}
            }});

            document.getElementById("total-count").innerText = POKEMON_DATA.length;
            document.getElementById("unique-types").innerText = types.size;
            document.getElementById("total-sprites").innerText = totalSprites;

            // Populate filters
            const typeFilter = document.getElementById("type-filter");
            const typeFilter2 = document.getElementById("type2-filter");
            Array.from(types).sort().forEach(t => {{
                const opt = document.createElement("option");
                opt.value = t;
                opt.innerText = t;
                typeFilter.appendChild(opt);
                
                const opt2 = document.createElement("option");
                opt2.value = t;
                opt2.innerText = t;
                typeFilter2.appendChild(opt2);
            }});

            const genFilter = document.getElementById("gen-filter");
            const gens = Array.from(new Set(POKEMON_DATA.map(p => p.generation))).sort();
            gens.forEach(g => {{
                const opt = document.createElement("option");
                opt.value = g;
                opt.innerText = `Gen ${{g}}`;
                genFilter.appendChild(opt);
            }});
        }}

        function formatId(id) {{
            return `#${{String(id).padStart(4, '0')}}`;
        }}

        // Infinite Scroll & Lazy Rendering State
        let activeData = [];
        let currentIndex = 0;
        const PAGE_SIZE = 36;
        let scrollObserver = null;

        function loadNextChunk() {{
            const grid = document.getElementById("pokemon-grid");
            const nextChunk = activeData.slice(currentIndex, currentIndex + PAGE_SIZE);

            nextChunk.forEach(p => {{
                const primaryColor = TYPE_COLORS[p.type1] || "#68A090";
                const secondaryColor = p.type2 !== "None" ? TYPE_COLORS[p.type2] : primaryColor;
                
                const rgbaGlow = TYPE_COLORS_RGBA[p.type1] || "rgba(104, 160, 144, 0.15)";
                
                const card = document.createElement("div");
                card.className = "pokemon-card";
                card.style.setProperty('--card-glow-bg', `linear-gradient(135deg, ${{rgbaGlow}} 0%, transparent 100%)`);
                card.style.setProperty('--card-glow-border', primaryColor + "70");
                card.style.setProperty('--card-glow-shadow', primaryColor + "30");
                card.style.setProperty('--primary-type-color', primaryColor);
                card.onclick = () => showDetails(p);

                // Build HTML with native loading="lazy" for image performance
                card.innerHTML = `
                    <div class="pokemon-id">${{formatId(p.id)}}</div>
                    <div class="pokemon-gen">Gen ${{p.generation}}</div>
                    <div class="pokemon-sprite-container">
                        <img src="${{p.sprite_url}}" class="pokemon-sprite" alt="${{p.name}}" loading="lazy" onerror="this.src='https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${{p.id}}.png'">
                    </div>
                    <h2 class="pokemon-name">${{p.name}}</h2>
                    <div class="type-badge-container">
                        <span class="type-badge" style="background-color: ${{primaryColor}}">${{p.type1}}</span>
                        ${{p.type2 !== "None" ? `<span class="type-badge" style="background-color: ${{secondaryColor}}">${{p.type2}}</span>` : ''}}
                    </div>
                    
                    <div class="stats-panel">
                        <div style="display: flex; gap: 0.5rem; justify-content: center; align-items: center; background: rgba(0,0,0,0.18); border-radius: 12px; padding: 0.45rem; margin-bottom: 0.6rem; border: 1px solid rgba(255,255,255,0.05); width: 100%;">
                            <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.05em; margin-right: auto; padding-left: 0.4rem;">Rogue Sprites</span>
                            <div style="display: flex; gap: 0.35rem; align-items: center;">
                                ${{p.pokerogue_sprites && p.pokerogue_sprites.length > 0 ? p.pokerogue_sprites.map(s => `
                                    <img src="${{s.preview_url}}" style="width: 28px; height: 28px; object-fit: contain; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5)); image-rendering: pixelated;" title="${{s.label}}" loading="lazy" onerror="this.style.display='none'">
                                `).join('') : '<span style="font-size: 0.7rem; color: var(--text-secondary); opacity: 0.6; padding-right: 0.4rem;">None</span>'}}
                            </div>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">HP</span>
                            <div class="stat-bar-container">
                                <div class="stat-bar" style="width: ${{Math.min(100, (p.stats.hp / 255) * 100)}}%"></div>
                            </div>
                            <span class="stat-value">${{p.stats.hp}}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">ATK</span>
                            <div class="stat-bar-container">
                                <div class="stat-bar" style="width: ${{Math.min(100, (p.stats.attack / 190) * 100)}}%"></div>
                            </div>
                            <span class="stat-value">${{p.stats.attack}}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">DEF</span>
                            <div class="stat-bar-container">
                                <div class="stat-bar" style="width: ${{Math.min(100, (p.stats.defense / 230) * 100)}}%"></div>
                            </div>
                            <span class="stat-value">${{p.stats.defense}}</span>
                        </div>
                    </div>

                    <div class="dimensions-row">
                        <div class="dimension-item">
                            <div>Height</div>
                            <div class="dimension-val">${{p.height}} m</div>
                        </div>
                        <div class="dimension-item">
                            <div>Weight</div>
                            <div class="dimension-val">${{p.weight}} kg</div>
                        </div>
                    </div>
                `;
                
                grid.appendChild(card);
            }});

            currentIndex += PAGE_SIZE;

            // Disconnect observer if all loaded
            if (currentIndex >= activeData.length && scrollObserver) {{
                scrollObserver.disconnect();
            }}
        }}

        function renderGrid(data) {{
            activeData = data;
            currentIndex = 0;
            const grid = document.getElementById("pokemon-grid");
            grid.innerHTML = "";

            // Render first chunk absolutely instantly
            loadNextChunk();

            // Set up scroll observer
            if (scrollObserver) {{
                scrollObserver.disconnect();
            }}

            scrollObserver = new IntersectionObserver(entries => {{
                if (entries[0].isIntersecting && currentIndex < activeData.length) {{
                    loadNextChunk();
                }}
            }}, {{
                rootMargin: "400px" // Load ahead when scrolled within 400px of sentinel
            }});

            scrollObserver.observe(document.getElementById("infinite-scroll-sentinel"));
        }}

        // Dynamic Filtering System
        function applyFilters() {{
            const searchQuery = document.getElementById("search-input").value.toLowerCase();
            const selectedType = document.getElementById("type-filter").value;
            const selectedType2 = document.getElementById("type2-filter").value;
            const selectedGen = document.getElementById("gen-filter").value;
            const sortBy = document.getElementById("sort-filter").value;

            let filtered = POKEMON_DATA.filter(p => {{
                const matchesSearch = p.name.toLowerCase().includes(searchQuery) || 
                                      String(p.id).includes(searchQuery) ||
                                      p.type1.toLowerCase().includes(searchQuery) ||
                                      p.type2.toLowerCase().includes(searchQuery);
                                      
                let matchesType = true;
                if (selectedType !== "All" && selectedType2 !== "All") {{
                    matchesType = (p.type1 === selectedType && p.type2 === selectedType2) ||
                                  (p.type1 === selectedType2 && p.type2 === selectedType);
                }} else if (selectedType !== "All") {{
                    matchesType = p.type1 === selectedType || p.type2 === selectedType;
                }} else if (selectedType2 !== "All") {{
                    matchesType = p.type1 === selectedType2 || p.type2 === selectedType2;
                }}
                                    
                const matchesGen = selectedGen === "All" || 
                                   String(p.generation) === selectedGen;

                return matchesSearch && matchesType && matchesGen;
            }});

            // Sorting
            filtered.sort((a, b) => {{
                const bstA = Object.values(a.stats).reduce((x, y) => x + y, 0);
                const bstB = Object.values(b.stats).reduce((x, y) => x + y, 0);
                
                if (sortBy === "id-asc") return a.id - b.id;
                if (sortBy === "id-desc") return b.id - a.id;
                if (sortBy === "name-asc") return a.name.localeCompare(b.name);
                if (sortBy === "bst-desc") return bstB - bstA;
                return 0;
            }});

            renderGrid(filtered);
        }}

        // Show detailed modal on card click
        function showDetails(p) {{
            const modal = document.getElementById("details-modal");
            const modalBody = document.getElementById("modal-body");
            const primaryColor = TYPE_COLORS[p.type1] || "#68A090";
            const secondaryColor = p.type2 !== "None" ? TYPE_COLORS[p.type2] : primaryColor;
            modalBody.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
                    <span style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; opacity: 0.5;">${{formatId(p.id)}}</span>
                    <h2 style="font-family: 'Outfit', sans-serif; font-size: 2.5rem; font-weight: 800; color: #fff; margin-bottom: 0.5rem;">${{p.name}}</h2>
                    
                    <div class="type-badge-container" style="margin-bottom: 1.5rem;">
                        <span class="type-badge" style="background-color: ${{primaryColor}}">${{p.type1}}</span>
                        ${{p.type2 !== "None" ? `<span class="type-badge" style="background-color: ${{secondaryColor}}">${{p.type2}}</span>` : ''}}
                    </div>
                    
                    <img src="${{p.sprite_url}}" style="width: 200px; height: 200px; object-fit: contain; margin-bottom: 2rem; filter: drop-shadow(0 15px 30px rgba(0,0,0,0.4));" onerror="this.src='https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${{p.id}}.png'">
                    
                    <!-- PokeRogue Sprites Display Section -->
                    <div style="width: 100%; text-align: left; margin-bottom: 1.5rem;">
                        <h4 style="font-family: 'Outfit', sans-serif; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem;">PokeRogue Sprite Frames</h4>
                        <div style="display: flex; flex-direction: column; gap: 1.2rem; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.25rem; border-radius: 20px;">
                            ${{p.pokerogue_sprites && p.pokerogue_sprites.length > 0 ? p.pokerogue_sprites.map(s => `
                                <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.03);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                        <span style="font-size: 0.85rem; font-weight: 800; color: #fff; letter-spacing: 0.03em;">${{s.label}}</span>
                                        <span style="font-size: 0.7rem; color: var(--text-secondary); font-weight: 600; background: rgba(255,255,255,0.05); padding: 0.2rem 0.5rem; border-radius: 8px;">${{s.frames.length}} frames</span>
                                    </div>
                                    <div style="display: flex; gap: 1rem; align-items: center; background: rgba(0,0,0,0.15); padding: 0.75rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.02);">
                                        <!-- Magnified Detail Panel (Avoids clipping) -->
                                        <div style="flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; width: 110px; height: 110px; padding: 0.5rem; transition: background-color 0.25s;">
                                            <img id="zoom-preview-${{s.folder_name}}" src="${{s.preview_url}}" style="width: 80px; height: 80px; object-fit: contain; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5)); image-rendering: pixelated;">
                                            <span id="zoom-label-${{s.folder_name}}" style="font-size: 0.65rem; font-family: monospace; color: var(--text-secondary); font-weight: 700; margin-top: 0.25rem;">F001</span>
                                        </div>
                                        
                                        <!-- Scrollable Thumbnails Selector -->
                                        <div style="flex: 1; display: flex; gap: 0.5rem; overflow-x: auto; padding: 0.2rem; scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;">
                                            ${{s.frames.map((frame, idx) => `
                                                <div style="flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.03); padding: 0.35rem; border-radius: 8px; cursor: pointer; transition: background-color 0.2s, border-color 0.2s;"
                                                     onmouseover="
                                                        document.getElementById('zoom-preview-${{s.folder_name}}').src='${{frame}}';
                                                        document.getElementById('zoom-label-${{s.folder_name}}').innerText='F${{String(idx + 1).padStart(3, '0')}}';
                                                        this.style.background='rgba(255,255,255,0.08)';
                                                        this.style.borderColor='rgba(255,255,255,0.15)';
                                                     "
                                                     onmouseout="
                                                        this.style.background='rgba(255,255,255,0.02)';
                                                        this.style.borderColor='rgba(255,255,255,0.03)';
                                                     ">
                                                    <img src="${{frame}}" style="width: 44px; height: 44px; object-fit: contain; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2)); image-rendering: pixelated;">
                                                    <span style="font-size: 0.55rem; font-family: monospace; color: var(--text-secondary); font-weight: 600;">F${{String(idx + 1).padStart(3, '0')}}</span>
                                                </div>
                                            `).join('')}}
                                        </div>
                                    </div>
                                </div>
                            `).join('') : '<span style="font-size: 0.9rem; color: var(--text-secondary); padding: 0.5rem;">No PokeRogue split sprite frames found for this Pokemon.</span>'}}
                    </div>

                    <div style="width: 100%; display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; text-align: left; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 20px; margin-bottom: 1.5rem;">
                        <div>
                            <span style="color: var(--text-secondary); font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Height</span>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #fff; margin-top: 0.25rem;">${{p.height}} meters</div>
                        </div>
                        <div>
                            <span style="color: var(--text-secondary); font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Weight</span>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #fff; margin-top: 0.25rem;">${{p.weight}} kilograms</div>
                        </div>
                    </div>
                    
                    <div style="width: 100%; text-align: left;">
                        <h4 style="font-family: 'Outfit', sans-serif; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem;">Base Stats Profile</h4>
                        <div style="display: grid; gap: 0.8rem;">
                            ${{Object.entries(p.stats).map(([name, val]) => `
                                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem;">
                                    <span style="color: var(--text-secondary); text-transform: uppercase; font-weight: 700; width: 80px;">${{name.replace('_', ' ')}}</span>
                                    <div style="flex: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; margin: 0 1rem; overflow: hidden;">
                                        <div style="height: 100%; background: ${{primaryColor}}; width: ${{Math.min(100, (val / 255) * 100)}}%"></div>
                                    </div>
                                    <span style="font-family: 'Outfit', sans-serif; font-weight: 700; width: 30px; text-align: right;">${{val}}</span>
                                </div>
                            `).join('')}}
                        </div>
                    </div>
                </div>
            `;

            modal.classList.add("active");
        }}

        // Setup event listeners
        document.getElementById("search-input").addEventListener("input", applyFilters);
        document.getElementById("type-filter").addEventListener("change", applyFilters);
        document.getElementById("type2-filter").addEventListener("change", applyFilters);
        document.getElementById("gen-filter").addEventListener("change", applyFilters);
        document.getElementById("sort-filter").addEventListener("change", applyFilters);

        document.getElementById("close-modal").onclick = () => {{
            document.getElementById("details-modal").classList.remove("active");
        }};
        
        window.onclick = (e) => {{
            const modal = document.getElementById("details-modal");
            if (e.target === modal) {{
                modal.classList.remove("active");
            }}
        }};

        // Initialization
        initDashboard();
        renderGrid(POKEMON_DATA);
    </script>
</body>
</html>
""";

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Pokemon HTML Viewer")
    parser.add_argument("--start", type=int, default=1, help="Start Pokedex ID")
    parser.add_argument("--end", type=int, default=1025, help="End Pokedex ID")
    parser.add_argument("--out", type=str, default="Data-Analysis/pokemon_viewer.html", help="Output HTML file path")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to pokeapi_data directory")
    args = parser.parse_args()
    
    # Range of indices example:
    # generate_html(range(args.start, args.end + 1), output_file=args.out, data_dir=args.data_dir)
    
    # DataFrame example showing how to take in the dataframes made in pokemon_filtering.py:
    # ----------------------------------------------------------------------------------
    # try:
    #     from pokemon_filtering import identify_pokemon_by_criteria
    #     print("Loading DataFrame from pokemon_filtering.py...")
    #     # Example: Filter all Gen 1 & 2 Water/Flying Pokemon
    #     count, df = identify_pokemon_by_criteria([1, 2], ('Water', 'Flying'), data_dir=args.data_dir)
    #     generate_html(df, output_file="filtered_pokemon.html", data_dir=args.data_dir)
    # except ImportError:
    #     print("Note: pokemon_filtering.py could not be imported. Run generator with range of indices instead.")
    
    print(f"Generating viewer for Pokemon indices {args.start} to {args.end}...")
    generate_html(range(args.start, args.end + 1), output_file=args.out, data_dir=args.data_dir)

