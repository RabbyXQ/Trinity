import os
import pandas as pd

# Directories containing Java files
stego_dir = "./stego"
safe_dir = "./safe"

# Function to read Java files and label them
def read_java_files(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".java"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                code = file.read()
                data.append({"code": code, "label": label})
    return data

# Read files from both directories
stego_data = read_java_files(stego_dir, 1)  # Label 1 for steganographic
safe_data = read_java_files(safe_dir, 0)    # Label 0 for safe

# Combine data and convert to DataFrame
dataset = pd.DataFrame(stego_data + safe_data)

# Save as CSV
dataset.to_csv("stegano_java_dataset.csv", index=False)

print(f"Dataset created with {len(dataset)} samples.")
