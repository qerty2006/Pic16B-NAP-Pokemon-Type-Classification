import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from collections import Counter
from plotly.subplots import make_subplots

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

def _get_type_distribution_data(
    generation: int | list[int] = 0,
    data_dir: str = "pokeapi_data",
    fixed_types: list[str] | None = None
):
    """
    Helper function to aggregate type data and expand it for square pillar visualization.
    Returns (z_expanded, x_coords, y_coords, unique_types, customdata)
    """
    if not os.path.exists(data_dir):
        return None

    # Normalize generation filter to a list
    if isinstance(generation, int):
        gen_filter = [generation] if generation != 0 else []
    else:
        gen_filter = generation

    type_combos = []
    
    # Iterate through each species folder
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
            
        # Get species base generation
        gen_url = species_data.get("generation", {}).get("url", "")
        try:
            species_gen = int(gen_url.strip("/").split("/")[-1])
        except (ValueError, IndexError):
            species_gen = 0
            
        species_combos = set()
        for variety_file in os.listdir(folder_path):
            if variety_file == "species.json" or not variety_file.endswith(".json"):
                continue
                
            variety_path = os.path.join(folder_path, variety_file)
            with open(variety_path, "r") as f:
                try:
                    variety_data = json.load(f)
                except json.JSONDecodeError:
                    continue
            
            # Determine logic for "Effective Generation" of this variety
            variety_name = variety_data.get("name", "").lower()
            effective_gen = species_gen # Default
            
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

            # Filter by effective generation
            if gen_filter and effective_gen not in gen_filter:
                continue

            # Extract types
            t_list = variety_data.get("types", [])
            t_list = sorted(t_list, key=lambda x: x['slot'])
            if not t_list:
                continue
                
            type1 = t_list[0]['type']['name'].capitalize()
            type2 = t_list[1]['type']['name'].capitalize() if len(t_list) > 1 else type1
            
            t1, t2 = sorted([type1, type2])
            species_combos.add((t1, t2))
            
        for combo in species_combos:
            type_combos.append(combo)

    if not type_combos and fixed_types is None:
        return None

    combo_counts = Counter(type_combos)
    
    # If fixed_types is provided, use it for the axes. Otherwise, use all types found in this batch.
    if fixed_types:
        unique_types = fixed_types
    else:
        unique_types = sorted(list(set([t for combo in type_combos for t in combo])))
    
    N = len(unique_types)
    eps = 0.05
    width = 0.4
    
    x_coords = []
    y_coords = []
    for i in range(N):
        x_coords.extend([i - width - eps, i - width, i + width, i + width + eps])
        y_coords.extend([i - width - eps, i - width, i + width, i + width + eps])
        
    z_expanded = []
    customdata_expanded = []
    
    for i, t2 in enumerate(unique_types):
        row_0 = []
        row_val = []
        hover_row = []
        for j, t1 in enumerate(unique_types):
            val = combo_counts.get(tuple(sorted((t1, t2))), 0)
            row_0.extend([0, 0, 0, 0])
            row_val.extend([0, val, val, 0])
            hover_text = f"Type 1: {t1}<br>Type 2: {t2}<br>Count: {val}"
            hover_row.extend([hover_text, hover_text, hover_text, hover_text])
            
        z_expanded.append(row_0)
        z_expanded.append(row_val)
        z_expanded.append(row_val)
        z_expanded.append(row_0)
        for _ in range(4):
            customdata_expanded.append(hover_row)
            
    return z_expanded, x_coords, y_coords, unique_types, customdata_expanded

def type_distribution(
    generation: int | list[int] = 0,
    data_dir: str = "pokeapi_data",
    filename: str | None = None,
    show_plot: bool = True
    ):
    """
    Makes a 3D plot of the type distribution for a given generation or list of generations.
    """
    data = _get_type_distribution_data(generation, data_dir)
    if data is None:
        print(f"No data found for generation(s) {generation}.")
        return

    z, x, y, unique_types, customdata = data
    N = len(unique_types)
    
    # Color scale logic
    z_array = np.array(z)
    max_val = np.max(z_array) if z_array.size > 0 else 1
    cutoff = 0.5 / max_val if max_val > 0 else 0.5
    viridis = pc.sequential.Viridis
    custom_colorscale = [[0.0, 'red'], [cutoff, 'red']]
    for i, color in enumerate(viridis):
        frac = i / (len(viridis) - 1)
        custom_colorscale.append([cutoff + frac * (1.0 - cutoff), color])

    if generation == 0:
        title_suffix = "All Generations"
        default_filename = "type_distribution_all.html"
    elif isinstance(generation, int):
        title_suffix = f"Generation {generation}"
        default_filename = f"type_distribution_gen_{generation}.html"
    else:
        title_suffix = f"Generations {', '.join(map(str, sorted(generation)))}"
        default_filename = f"type_distribution_gen_{'_'.join(map(str, sorted(generation)))}.html"

    final_filename = filename if filename else default_filename

    fig = go.Figure(data=[go.Surface(
        z=z, x=x, y=y,
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=custom_colorscale,
        showscale=True
    )])

    fig.update_layout(
        title=f'Pokemon Type Distribution - {title_suffix}',
        scene=dict(
            xaxis=dict(title='Type 1', tickmode='array', tickvals=list(range(N)), ticktext=unique_types),
            yaxis=dict(title='Type 2', tickmode='array', tickvals=list(range(N)), ticktext=unique_types),
            zaxis=dict(title='Frequency'),
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=1000, height=900, autosize=False
    )

    fig.write_html(final_filename)
    print(f"Plot saved to {final_filename}")
    if show_plot:
        fig.show()

def generation_type_distribution(
    data_dir: str = "pokeapi_data",
    filename: str | None = None
):
    """
    Makes a 3D plot of the type distribution for each generation, with each generation having its own subplot in one big plot.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    # First, get the full set of types across ALL data to standardize axes
    full_data = _get_type_distribution_data(0, data_dir)
    if full_data is None:
        print("No data found to determine global type set.")
        return
    global_types = full_data[3]
    N = len(global_types)

    # Determine available generations by reading a sample of folders or all of them
    unique_gens_set = set()
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path): continue
        species_file = os.path.join(folder_path, "species.json")
        if os.path.exists(species_file):
            with open(species_file, "r") as f:
                try:
                    species_data = json.load(f)
                    gen_url = species_data.get("generation", {}).get("url", "")
                    if gen_url:
                        gen_id = int(gen_url.strip("/").split("/")[-1])
                        unique_gens_set.add(gen_id)
                except Exception:
                    continue
    
    unique_gens = sorted(list(unique_gens_set))
    if not unique_gens:
        print("No generations found in the data.")
        return

    num_plots = len(unique_gens)
    cols = 3
    rows = int(np.ceil(num_plots / cols))

    # Adjust spacing to be safe for 3x3
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'surface'}] * cols] * rows,
        subplot_titles=[f"Generation {g}" for g in unique_gens],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Color scale logic (global for all subplots)
    viridis = pc.sequential.Viridis
    # We'll calculate a local max per plot for color scaling within that plot, 
    # or a global max for overall comparison? Let's go with local max for better visibility per gen.
    
    for idx, gen in enumerate(unique_gens):
        r = (idx // cols) + 1
        c = (idx % cols) + 1
        
        data = _get_type_distribution_data(gen, data_dir, fixed_types=global_types)
        if data is None: continue
        z, x, y, _, customdata = data
        
        # Local colorscale for this subplot
        z_array = np.array(z)
        max_val = np.max(z_array) if z_array.size > 0 else 1
        cutoff = 0.5 / max_val if max_val > 0 else 0.5
        local_colorscale = [[0.0, 'red'], [cutoff, 'red']]
        for i, color in enumerate(viridis):
            frac = i / (len(viridis) - 1)
            local_colorscale.append([cutoff + frac * (1.0 - cutoff), color])

        fig.add_trace(
            go.Surface(
                z=z, x=x, y=y,
                customdata=customdata,
                hovertemplate="%{customdata}<extra></extra>",
                colorscale=local_colorscale,
                showscale=(idx == len(unique_gens) - 1), # Only show scale for last plot
                name=f"Gen {gen}"
            ),
            row=r, col=c
        )

        # Update scene layout for this subplot
        scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
        fig.update_layout({
            scene_name: dict(
                xaxis=dict(title='', tickmode='array', tickvals=list(range(N)), ticktext=global_types if idx >= (rows-1)*cols else []),
                yaxis=dict(title='', tickmode='array', tickvals=list(range(N)), ticktext=global_types if idx % cols == 0 else []),
                zaxis=dict(title='Freq'),
                aspectratio=dict(x=1, y=1, z=0.5)
            )
        })

    fig.update_layout(
        title_text='Pokémon Type Distribution by Generation',
        height=400 * rows,
        width=400 * cols,
        showlegend=False
    )

    final_filename = filename if filename else "generation_type_distribution.html"
    fig.write_html(final_filename)
    print(f"Subplot report saved to {final_filename}")
    fig.show()


    
if __name__ == "__main__":
    # Test generation_type_distribution
    generation_type_distribution()
    type_distribution()