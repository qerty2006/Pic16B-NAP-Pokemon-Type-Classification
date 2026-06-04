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
from torchvision.transforms import InterpolationMode

TYPES = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting",
    "fire", "flying", "ghost", "grass", "ground", "ice",
    "normal", "poison", "psychic", "rock", "steel", "water",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}

PROJECT_ROOT = Path(__file__).parent.parent
SPRITES_DIR = Path(__file__).parent / "split_sprites"
POKEAPI_DIR = Path(__file__).parent / "pokeapi_data"
INDEX_CACHE = Path(__file__).parent / ".index_cache.pkl"

# Bump when label format or stored fields change — stale cache will load wrong label shapes
CACHE_VERSION = 2  # bumped for multi-label (multi-hot) labels

# To add a new generation: append (first_id, last_id) to this list
GEN_RANGES = [
    (1, 151), (152, 251), (252, 386), (387, 493),
    (494, 649), (650, 721), (722, 809), (810, 905), (906, 1025),
]

# 224x224 and ImageNet norm required — pretrained EfficientNet-B0 expects this exact input
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5), # Swaps facing direction
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.7),  # Shifts color values slightly
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- NEW GRAYSCALE TRANSFORMS ---
# Uses num_output_channels=3 so the EfficientNet backbone still receives the 3-channel 
# tensor it expects, avoiding shape mismatch errors with pre-trained weights.
GRAYSCALE_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

GRAYSCALE_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5), # Swaps facing direction
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Removed saturation shift
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def rgba_to_rgb(img: Image.Image) -> Image.Image:
    """Composite an RGBA sprite onto a white background and return an RGB image.

    Pokemon sprites use transparency for their background. Pasting onto white
    matches how a browser renders them and gives EfficientNet a consistent input.
    """
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg



def get_generation(pokemon_id: int, folder_name: str = "") -> int:
    """Determine generation from ID range, but override with regional/form tags if present."""
    tags = {
        "alola": 7,
        "galar": 8,
        "hisui": 8,
        "husui": 8,  # supporting user typo
        "paldea": 9,
        "mega": 6,
        "gigantamax": 8,
    }
    name_lower = folder_name.lower()
    for tag, gen in tags.items():
        if tag in name_lower:
            return gen

    for gen, (lo, hi) in enumerate(GEN_RANGES, 1):
        if lo <= pokemon_id <= hi:
            return gen
    return 0


def parse_folder_id(folder_name: str) -> int | None:
    """Helper to safely extract the base integer ID from folder names like '3-mega' or '12'."""
    base_part = folder_name.split("-")[0]
    return int(base_part) if base_part.isdigit() else None


def gen_stratified_split(index, val_frac=0.15, test_frac=0.15, seed=42):
    """Split stratified by (generation, dual/single type) ensuring that ALL
    sprites for a specific Pokemon ID stay together in the same split group."""

    id_to_indices = {}
    id_to_stratum = {}

    # 1. Group all image frame indices by their base Pokemon ID
    for i, (path, label) in enumerate(index):
        pokemon_id = parse_folder_id(path.parent.name)
        if pokemon_id is None:
            continue

        id_to_indices.setdefault(pokemon_id, []).append(i)

        # Determine the demographic stratum based on the first time we see this ID
        if pokemon_id not in id_to_stratum:
            gen = get_generation(pokemon_id)
            is_dual = int(label.sum()) >= 2
            id_to_stratum[pokemon_id] = (gen, is_dual)

    # 2. Group the unique Pokemon IDs by their stratum buckets
    by_stratum = {}
    for pokemon_id, stratum in id_to_stratum.items():
        by_stratum.setdefault(stratum, []).append(pokemon_id)

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)

    # 3. Shuffle IDs inside each stratum, then pull ALL their frames into the same split
    for ids in by_stratum.values():
        ids = np.array(ids)
        rng.shuffle(ids)

        n = len(ids)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        test_ids = ids[:n_test]
        val_ids = ids[n_test:n_test + n_val]
        train_ids = ids[n_test + n_val:]

        # Unpack every single image index belonging to these IDs
        for pid in test_ids:
            test_idx.extend(id_to_indices[pid])
        for pid in val_ids:
            val_idx.extend(id_to_indices[pid])
        for pid in train_ids:
            train_idx.extend(id_to_indices[pid])

    return train_idx, val_idx, test_idx

def gen_gen_split(index, train_gens=(1, 2, 3), val_frac=0.15, test_gens=(4, 5, 6), seed=42):
    """Split based on specific generations for training and testing."""
    train_pool = []
    test_pool = []
    
    for i, (path, label) in enumerate(index):
        folder_name = path.parent.name
        pokemon_id = int(folder_name.split("-")[0])
        gen = get_generation(pokemon_id, folder_name)
        
        if gen in train_gens:
            train_pool.append(i)
        elif gen in test_gens:
            test_pool.append(i)
            
    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool)
    
    n_val = int(len(train_pool) * val_frac)
    val_idx = train_pool[:n_val]
    train_idx = train_pool[n_val:]
    
    # shuffle test as well for consistency
    rng.shuffle(test_pool)
    test_idx = test_pool
    
    return train_idx, val_idx, test_idx


def _fetch_entry(sprite_folder, pokeapi_by_id):
    """Load one Pokemon's index entry. Returns (path, multi_hot_label) or None."""
    pokemon_id = parse_folder_id(sprite_folder.name)

    pokeapi_folder = pokeapi_by_id.get(pokemon_id)
    if pokeapi_folder is None:
        return None
    folder_name_string = sprite_folder.name
    species_name = pokeapi_folder.name.split("_", 1)[1]
    
    # Form-aware file matching logic
    if "-" in folder_name_string:
        form_suffix = folder_name_string.split("-", 1)[1]
        variety_json = pokeapi_folder / f"{species_name}-{form_suffix}.json"
    else:
        # Zygarde base form is zygarde-50
        if species_name == "zygarde":
            variety_json = pokeapi_folder / "zygarde-50.json"
        else:
            variety_json = pokeapi_folder / f"{species_name}.json"

    # Fallback to base species JSON if form-specific one doesn't exist
    if not variety_json.exists():
        variety_json = pokeapi_folder / f"{species_name}.json"
        if not variety_json.exists():
            candidates = [f for f in pokeapi_folder.glob("*.json") if f.name != "species.json"]
            if candidates:
                variety_json = candidates[0]
            else:
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


    return [(frame, label) for frame in frames[:1]]


def _build_index(use_cache=True):
    """Returns list of (image_path, multi_hot_label) for all base-form Pokemon (IDs 1-1025).
    Caches result to disk — delete .index_cache.pkl if pokeapi_data or sprites change.
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
        if p.is_dir() and parse_folder_id(p.name) is not None
    ]

    from collections import defaultdict
    variant_counts = defaultdict(int)
    filtered_folders = []

    # Sort folders so base forms (e.g., '6') come before variants (e.g., '6-mega-x')
    sprite_folders.sort(key=lambda p: (parse_folder_id(p.name), len(p.name)))

    for folder in sprite_folders:
        pokemon_id = parse_folder_id(folder.name)

        # Check if it's a variant folder (contains a hyphen)
        if "-" in folder.name:
            if variant_counts[pokemon_id] >= 3:
                continue  # Skip this variant, we already have 3!
            variant_counts[pokemon_id] += 1

        filtered_folders.append(folder)

    index = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_fetch_entry, sf, pokeapi_by_id): sf for sf in filtered_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building index"):
            result = future.result()
            if result is not None:
                index.extend(result)

    # sort by pokemon ID for reproducibility
    index.sort(key=lambda x: (parse_folder_id(x[0].parent.name), x[0].parent.name, x[0].name))

    if use_cache:
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"version": CACHE_VERSION, "index": index}, f)

    return index


class PokemonSpriteDataset(Dataset):
    """PyTorch Dataset over all base-form Pokemon sprites (IDs 1–1025).

    Each __getitem__ returns a (image_tensor [3, 224, 224], label_tensor [18]) pair.
    The label is multi-hot: positions corresponding to the Pokemon's type(s) are 1.0,
    all others 0.0. Dual-type Pokemon have exactly two 1s; single-type have one.
    """

    def __init__(self, transform=None, use_cache=True):
        self.index = _build_index(use_cache=use_cache)
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, label = self.index[idx]
        img = rgba_to_rgb(Image.open(img_path).convert("RGBA"))
        img = self.transform(img)
        # label is multi-hot: 1.0 at each type index the Pokemon has (1 or 2 entries set)
        return img, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def num_classes():
        return len(TYPES)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset

    print("Loading dataset...")
    ds = PokemonSpriteDataset()
    print(f"Total samples: {len(ds)}")

    # ====================================================================
    print("\n--- Spot-Checking Variant Forms & Types ---")

    # We will look for folders containing these specific words in our index
    test_forms = ["mega", "gigantamax", "alola", "galar", "hisui", "paldea"]
    found_variants = 0

    for path, label in ds.index:
        folder_name = path.parent.name

        # If the folder name contains a hyphen (meaning it's a variant)
        if "-" in folder_name:
            found_variants += 1

            # Convert the multi-hot float array back into text names for printing
            active_types = [TYPES[idx] for idx, val in enumerate(label) if val == 1.0]

            # Print out the first 15 variants found so you can review them
            if found_variants <= 15:
                print(f"Folder: {folder_name:<18} -> Decoded Types: {active_types}")

    print(f"\nTotal variant/form folders successfully loaded: {found_variants}")
    # ====================================================================

    print("\nCounting type distribution...")

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