import os
import shutil

def is_valid_folder(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

def save_matching_images(image_folder, results, save_folder, num_matches=10):
    os.makedirs(save_folder, exist_ok=True)
    for filename, _ in results[:num_matches]:
        src = os.path.join(image_folder, filename)
        dst = os.path.join(save_folder, filename)
        shutil.copy(src, dst)
