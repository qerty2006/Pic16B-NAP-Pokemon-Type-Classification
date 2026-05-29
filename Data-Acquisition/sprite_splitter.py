import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = Path(__file__).parent
root_dir = base_dir.parent

# Check Data-Acquisition/ first, then project root for older local downloads.
_candidates = [base_dir / "pokerogue_sprites", root_dir / "pokerogue_sprites"]
sprites_folder = next((p for p in _candidates if p.exists()), _candidates[0])
output_folder = base_dir / "split_sprites"


def process_sheet(json_path):
    output_dir = output_folder / json_path.stem
    if output_dir.exists():
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'textures' in data:
        frames = data['textures'][0]['frames']
    elif 'frames' in data:
        frames = data['frames']
    else:
        tqdm.write(f"Skipping {json_path.name}: no frames found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sprite_sheet = Image.open(json_path.with_suffix(".png"))

    for sprite in frames:
        f = sprite['frame']
        sprite_img = sprite_sheet.crop((f['x'], f['y'], f['x'] + f['w'], f['y'] + f['h']))
        sprite_img.save(output_dir / sprite['filename'])


json_files = list(sprites_folder.glob("*.json"))

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_sheet, p): p for p in json_files}
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Splitting sprites"):
        pass

print(f"Done! Check the '{output_folder}' folder.")
