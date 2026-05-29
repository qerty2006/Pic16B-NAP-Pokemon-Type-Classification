import os
import subprocess
import sys

def main():
    print("=== Pokemon Type Classification Analysis ===")
    
    # 1. Check for data
    if not os.path.exists("pokeapi_data"):
        print("\n[1/2] Data not found. Running data acquisition...")
        subprocess.run([sys.executable, "Data-Acquisition/pokeapi_data.py"], check=True)
    else:
        print("\n[1/2] Data directory 'pokeapi_data' found. Skipping acquisition.")
    
    # 2. Run visualization
    print("\n[2/2] Generating visualizations...")
    subprocess.run([sys.executable, "Data-Analysis/pokeapi_visualizers.py"], check=True)
    
    print("\nAnalysis complete! Visualizations saved as HTML files in the project root.")

if __name__ == "__main__":
    main()
