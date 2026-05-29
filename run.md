# Run Commands

## One-time setup (already done on this machine)

```powershell
# Accept Anaconda channel ToS (required by recent Miniconda)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

# Allow PowerShell to load the conda hook in your profile
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create the local env and install deps
conda create --yes --prefix ./.conda python=3.11
conda activate ./.conda
pip install -r requirements.txt        # CPU PyTorch by default
bash install_hooks.sh                  # git pre-push safety hook (Git Bash)
```

GPU machines: install the CUDA build before `pip install -r requirements.txt`:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Every session

```powershell
conda activate ./.conda                # prompt shows (.conda)
```

## Data acquisition (run once, in order)

```powershell
bash Data-Acquisition/setup_pokerogue_assets.sh   # download sprite sheets (Git Bash)
python Data-Acquisition/sprite_splitter.py        # split sheets into sprites
python Data-Acquisition/pokeapi_data.py           # fetch type labels
```

## Sanity check

```powershell
python Classification/dataset.py
```

## Train / evaluate / report

```powershell
python Classification/baselines.py
python Classification/train.py --epochs 30
python Classification/evaluate.py
python Classification/generate_report.py           # writes results/index.html
```

## Visualizations

```powershell
python run_analysis.py
```
