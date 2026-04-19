import requests
import plotly.graph_objects as go
from collections import Counter
import numpy as np

def main():
    print("Fetching Pokémon data (this may take a minute or two depending on your internet connection)...")
    type_combos = []
    data_total = 0

    # Get data for the first 1028 Pokémon species and their forms
    for i in range(1, 1029):
        try:
            species_url = f"https://pokeapi.co/api/v2/pokemon-species/{i}/"
            species_response = requests.get(species_url)
            species_response.raise_for_status()
            species_data = species_response.json()
            data_total += len(species_response.content)
            species_combos = set()
            
            for variety in species_data['varieties']:
                p_response = requests.get(variety['pokemon']['url'])
                p_response.raise_for_status()
                pokemon = p_response.json()
                data_total += len(p_response.content)
                
                # The 'types' attribute contains the types
                t_list = pokemon['types']
                t_list = sorted(t_list, key=lambda x: x['slot'])
                type1 = t_list[0]['type']['name'].capitalize()
                if len(t_list) > 1:
                    type2 = t_list[1]['type']['name'].capitalize()
                else:
                    type2 = type1
                    
                # For simplification sake, xy types should be the same as yx types
                t1, t2 = sorted([type1, type2])
                species_combos.add((t1, t2))
                
            for combo in species_combos:
                type_combos.append(combo)
            
            # Print a progress update every 25 Pokémon species
            if i % 25 == 0:
                print(f"Fetched {i}/1028 species...")

        except Exception as e:
            print(f"Failed to fetch data for Pokémon species {i}: {e}")

    # Count the occurrences of each type combination
    combo_counts = Counter(type_combos)
    
    unique_types = sorted(list(set([t for combo in type_combos for t in combo])))
    N = len(unique_types)
    
    # Create the 2D array for the height map but with square pillars
    # We expand each cell into a 4x4 grid of coordinates to create steep 3D square pillars
    x_coords = []
    y_coords = []
    eps = 0.05
    width = 0.4
    
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
            # Ensure diagonal symmetry by always looking up the sorted tuple
            val = combo_counts.get(tuple(sorted((t1, t2))), 0)
            
            # 4 points along the x-axis for this column
            row_0.extend([0, 0, 0, 0])
            row_val.extend([0, val, val, 0])
            
            hover_text = f"Type 1: {t1}<br>Type 2: {t2}<br>Count: {val}"
            hover_row.extend([hover_text, hover_text, hover_text, hover_text])
            
        # 4 points along the y-axis for this row
        z_expanded.append(row_0)
        z_expanded.append(row_val)
        z_expanded.append(row_val)
        z_expanded.append(row_0)
        
        customdata_expanded.append(hover_row)
        customdata_expanded.append(hover_row)
        customdata_expanded.append(hover_row)
        customdata_expanded.append(hover_row)

    max_val = max(combo_counts.values()) if combo_counts else 1
    cutoff = 0.5 / max_val
    
    import plotly.colors
    viridis = plotly.colors.sequential.Viridis
    
    custom_colorscale = [
        [0.0, 'red'],
        [cutoff, 'red'],
    ]
    
    for i, color in enumerate(viridis):
        frac = i / (len(viridis) - 1)
        val = cutoff + frac * (1.0 - cutoff)
        custom_colorscale.append([val, color])

    # Create the 3D height map (Surface plot matching square pillars)
    fig = go.Figure(data=[go.Surface(
        z=z_expanded,
        x=x_coords,
        y=y_coords,
        customdata=customdata_expanded,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=custom_colorscale,
        showscale=True
    )])

    # Setup axis ticks to align with our numerical coordinates
    tickvals = list(range(N))
    fig.update_layout(
        title='Gen 9 Type Distribution (3D Square Pillars, Diagonally Symmetric)',
        scene=dict(
            xaxis=dict(
                title='Type 1',
                tickmode='array',
                tickvals=tickvals,
                ticktext=unique_types
            ),
            yaxis=dict(
                title='Type 2',
                tickmode='array',
                tickvals=tickvals,
                ticktext=unique_types
            ),
            zaxis=dict(title='Frequency'),
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        autosize=False,
        width=1000,
        height=900
    )

    # Save to file
    output_filename = 'type_distribution_3d.html'
    fig.write_html(output_filename)
    print(f"3D height map successfully saved to {output_filename}")
    print(f"Total data downloaded: {data_total / 1024 / 1024:.2f} MB ({data_total} bytes)")
    
    # Display the plot
    fig.show()

if __name__ == "__main__":
    print("Hello World")
    main()
