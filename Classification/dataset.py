import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TYPES = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting",
    "fire", "flying", "ghost", "grass", "ground", "ice",
    "normal", "poison", "psychic", "rock", "steel", "water",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}

PROJECT_ROOT = Path(__file__).parent.parent
SPRITES_DIR = PROJECT_ROOT / "Data-Acquisition" / "split_sprites"
POKEAPI_DIR = PROJECT_ROOT / "pokeapi_data"
INDEX_CACHE = Path(__file__).parent / ".index_cache.pkl"

CACHE_VERSION = 2  # bumped for multi-label (multi-hot) labels

# National Dex generation cut-offs
GEN_RANGES = [
    (1, 151), (152, 251), (252, 386), (387, 493),
    (494, 649), (650, 721), (722, 809), (810, 905), (906, 1025),
]

# ImageNet normalization — required for pretrained ResNet
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def rgba_to_rgb(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def get_generation(pokemon_id: int) -> int:
    for gen, (lo, hi) in enumerate(GEN_RANGES, 1):
        if lo <= pokemon_id <= hi:
            return gen
    return 0


def gen_stratified_split(index, val_frac=0.15, test_frac=0.15, seed=42):
    """Split per generation so every gen is proportionally represented in each split."""
    by_gen = {}
    for i, (path, _) in enumerate(index):
        gen = get_generation(int(path.parent.name))
        by_gen.setdefault(gen, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)
    for indices in by_gen.values():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return train_idx, val_idx, test_idx


def _fetch_entry(sprite_folder, pokeapi_by_id):
    """Load one Pokemon's index entry. Returns (path, multi_hot_label) or None."""
    pokemon_id = int(sprite_folder.name)
    if pokemon_id > 1025:
        return None

    pokeapi_folder = pokeapi_by_id.get(pokemon_id)
    if pokeapi_folder is None:
        return None

    species_name = pokeapi_folder.name.split("_", 1)[1]
    variety_json = pokeapi_folder / f"{species_name}.json"
    if not variety_json.exists():
        return None

    with open(variety_json) as f:
        data = json.load(f)

    types = sorted(data.get("types", []), key=lambda x: x["slot"])
    if not types:
        return None

    label = np.zeros(len(TYPES), dtype=np.float32)
    has_valid = False
    for t in types:
        name = t["type"]["name"]
        if name in TYPE_TO_IDX:
            label[TYPE_TO_IDX[name]] = 1.0
            has_valid = True
    if not has_valid:
        return None

    frames = sorted(sprite_folder.glob("*.png"))
    if not frames:
        return None

    return (frames[0], label)


def _build_index(use_cache=True):
    """Returns list of (image_path, multi_hot_label) for all base-form Pokemon (IDs 1-1025).
    Caches result to disk — delete .index_cache.pkl if pokeapi_data changes.
    """
    if use_cache and INDEX_CACHE.exists():
        with open(INDEX_CACHE, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get("version") == CACHE_VERSION:
            print("Loading index from cache...")
            return cached["index"]
        print("Cache format changed, rebuilding index...")

    pokeapi_by_id = {}
    for folder in POKEAPI_DIR.iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            pokeapi_by_id[int(parts[0])] = folder

    sprite_folders = [
        p for p in SPRITES_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]

    index = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_fetch_entry, sf, pokeapi_by_id): sf for sf in sprite_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building index"):
            result = future.result()
            if result is not None:
                index.append(result)

    # sort by pokemon ID for reproducibility
    index.sort(key=lambda x: int(x[0].parent.name))

    if use_cache:
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"version": CACHE_VERSION, "index": index}, f)

    return index


class PokemonSpriteDataset(Dataset):
    def __init__(self, transform=None, use_cache=True):
        self.index = _build_index(use_cache=use_cache)
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, label = self.index[idx]
        img = rgba_to_rgb(Image.open(img_path).convert("RGBA"))
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def num_classes():
        return len(TYPES)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset

    print("Loading dataset...")
    ds = PokemonSpriteDataset()
    print(f"Total samples: {len(ds)}")

    print("\nCounting type distribution...")
    type_counts = np.zeros(len(TYPES), dtype=int)
    for _, label in tqdm(ds.index, desc="Counting"):
        type_counts += label.astype(int)
    dual_count = sum(1 for _, label in ds.index if label.sum() == 2)
    print(f"\nDual-type Pokemon: {dual_count} / {len(ds.index)}")
    print("\nType distribution (Pokemon with this type):")
    for t, idx in sorted(TYPE_TO_IDX.items()):
        print(f"  {t:<12} {type_counts[idx]:>4}")

    print("\nComputing split...")
    train_idx, val_idx, test_idx = gen_stratified_split(ds.index)
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    print("\nLoading one batch...")
    loader = DataLoader(Subset(ds, train_idx[:8]), batch_size=4)
    imgs, labels = next(iter(loader))
    print(f"Batch shape: {imgs.shape}  Labels shape: {labels.shape}")
    print(f"Example label (multi-hot): {labels[0].tolist()}")
