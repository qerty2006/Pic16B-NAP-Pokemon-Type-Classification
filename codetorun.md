HOW TO SET UP FILES FOR THE PROJECT, replace /Users/nt/miniconda3/bin/python with /Users/nt/miniconda3/bin/python or the python interpreter you are using

"Acquisition"
./Data-Acquisition/setup_pokerogue_assets.sh      
/Users/nt/miniconda3/bin/python /Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Data-Acquisition/sprite_splitter.py
/Users/nt/miniconda3/bin/python /Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Data-Acquisition/pokeapi_data_threaded.py
/Users/nt/miniconda3/bin/python "/Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Data-Acquisition/Model Labeling.py"

"Analysis"
/Users/nt/miniconda3/bin/python /Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Data-Analysis/pokemo
n_filtering.py

"Training + post mortem"
 /Users/nt/miniconda3/bin/python /Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Classification/pipeline.py --epochs 50 --batch-size 64
 /Users/nt/miniconda3/bin/python /Users/nt/Documents/github/Pic16B-NAP-Pokemon-Type-Classification/Classification/visualize_cnn.py -n 100

