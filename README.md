# Pic16B-NAP-Pokemon-Type-Classification

## Setup

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Sprites

Run the setup script to download Pokémon sprite assets from the PokéRogue repository (requires Git Bash on Windows):

```bash
bash Data-Acquisition/setup_pokerogue_assets.sh
```

Then split the sprite sheets into individual images:

```bash
python Data-Acquisition/sprite_splitter.py
```

## Usage

### One-Click Analysis

To fetch all data and generate the default visualizations in one go:

```bash
python run_analysis.py
```

### Manual Execution

1. **Data Acquisition**:
   Fetch the latest Pokémon data:
   ```bash
   python Data-Acquisition/pokeapi_data.py
   ```
   > **Experimental:** A threaded version is available for faster downloads (~20x speedup). Re-run safe — skips completed entries and only fetches missing varieties. May occasionally hit PokeAPI rate limits; failed entries are printed but won't stop the run.
   > ```bash
   > python Data-Acquisition/pokeapi_data_threaded.py
   > ```

2. **Run Visualizations**:
   Generate the interactive HTML plots:
   ```bash
   python Data-Analysis/pokeapi_visualizers.py
   ```

## Visualizations

The analysis generates two primary reports:
- `generation_type_distribution.html`: A grid of 3D plots showing the type distribution within each generation.
- `type_distribution_all.html`: A comprehensive 3D plot of all Pokémon types across all generations.
