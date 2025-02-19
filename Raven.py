import os
import shutil

def move_png_files(base_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".java"):  # Check if the file is a .png file
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                # Move the file
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} -> {target_path}")

# Example usage
move_png_files("./java-enterprise-examples", "./safe")
