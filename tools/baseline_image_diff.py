import json
from pathlib import Path
from matplotlib.testing.compare import get_cache_dir

with open('../baseline_image_modifications/PR1.json') as f:
  data = json.load(f)

added_files = data["files_added"]
deleted_files = data["files_deleted"]
modified_files = data["files_modified"]

# deleted we don't need to show the diff

for file in added_files:
    test = data["test"]
    file = data["file"]
    img_names = data["img_names"]
    for img in img_names
        # comment the image as comment in the github from the developers input


for file in modified_files:
    test = data["prev_test"]
    file = data["prev_file"]
    img_names = data["prev_img_names"]
    new_test = data["new_test"]
    new_file = data["new_file"]
    new_img_names = data["new_img_names"]
    for img in prev_img_names
        cache_dir = Path(get_cache_dir()) / "baseline_images" / img
        cache_dir_img = (cache_dir) / \
                        (img).relative_to(baseline_dir)
        # comment the image as comment in the github from the developers input
        # also comment the img from the cache as a comment in the PR

