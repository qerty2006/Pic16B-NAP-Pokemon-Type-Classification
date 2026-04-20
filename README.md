# Pic16B-NAP-Pokemon-Type-Classification

## Setup

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Sprites

Run the setup script to download Pokémon sprite assets from the PokéRogue repository:

```bash
bash setup_assets.sh
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

2. **Run Visualizations**:
   Generate the interactive HTML plots:
   ```bash
   python Data-Analysis/pokeapi_visualizers.py
   ```

## Visualizations

The analysis generates two primary reports:
- `generation_type_distribution.html`: A grid of 3D plots showing the type distribution within each generation.
- `type_distribution_all.html`: A comprehensive 3D plot of all Pokémon types across all generations.
