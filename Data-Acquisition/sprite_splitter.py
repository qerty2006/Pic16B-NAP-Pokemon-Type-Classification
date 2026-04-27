import json
import os
from PIL import Image
from pathlib import Path

#PS: Please keep this py file on the same dir as the pokerogue_sprites folder
base_dir = Path(__file__).parent
sprite_sheets='pokerogue_sprites'

sprites_folder = base_dir / "pokerogue_sprites"
output_folder = base_dir/ ("split_sprites")

#setting up paths and loading data
for json_path in sprites_folder.glob("*.json"):

    image_path = json_path.with_suffix(".png")

    with open(json_path, 'r') as file:
        data = json.load(file)

    # open sprite sheet
    sprite_sheet = Image.open(image_path)

    # Creates a folder with sprites for each Pokemon
    output_dir = output_folder/json_path.stem
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Access the frames list from textures
    if 'textures' in data:
        frames = data['textures'][0]['frames']

    # If 'textures' isn't there, access frames directly
    elif 'frames' in data:
        frames = data['frames']

    for sprite in frames:
        name = sprite['filename']
        f = sprite['frame']

        # Cropping boundaries
        left = f['x']
        top = f['y']
        right = left + f['w']
        bottom = top + f['h']

        # Crop the image
        sprite_img = sprite_sheet.crop((left, top, right, bottom))

        # Save the individual sprite
        sprite_img.save(os.path.join(output_dir, name))
    print(f"Saved: {json_path}")


print("Done! Check the 'split_sprites' folder.")
