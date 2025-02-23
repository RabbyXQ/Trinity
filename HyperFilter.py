import os
import numpy as np
import pandas as pd
import cv2
import hashlib
import time
import gc

# Define the directory paths for malware and benign Fourier-transformed images
MALWARE_DIR = './fouier/malware'
BENIGN_DIR = './fouier/benign'
CACHE_DIR = './cache'  # Directory to store cached Fourier images

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Define the dummy CSV file path
output_file = 'high_freq_noise_delta_features_with_labels.csv'

# Initialize header for CSV if the file does not exist
header = [
    'high_freq_features', 'noise_features', 'delta_high_freq', 'delta_noise', 'label', 'numerical_label'
]

# Create an empty DataFrame with the header columns
df = pd.DataFrame(columns=header)

# Save the dummy CSV file (only with headers)
df.to_csv(output_file, mode='w', index=False)

print(f"Dummy CSV file '{output_file}' created with headers.")

# Function to generate a unique cache key based on the image file path
def generate_cache_key(file_path):
    hash_object = hashlib.md5(file_path.encode())
    return hash_object.hexdigest() + '.npz'

# Function to extract high-frequency components and noise from Fourier-transformed image
def extract_features(fourier_image):
    # Get the magnitude spectrum of the Fourier image
    magnitude_spectrum = np.abs(fourier_image)
    
    # Define a threshold for high-frequency components (e.g., retain the top 10% of frequencies)
    rows, cols = fourier_image.shape
    crow, ccol = rows // 2, cols // 2  # Central point
    
    # Extract high-frequency components (outer part of the spectrum)
    high_freq = magnitude_spectrum[crow-50:crow+50, ccol-50:ccol+50]  # Adjust region as needed
    high_freq_features = high_freq.flatten()
    
    # Extract low-frequency (noise) components (central part of the spectrum)
    noise_region = magnitude_spectrum[:crow, :ccol]  # You can change this region if necessary
    noise_features = noise_region.flatten()
    
    return high_freq_features, noise_features

# Function to process images from a directory and extract features dynamically
def process_directory(directory, label, numerical_label, cache_expiration=3600, output_file='high_freq_noise_delta_features_with_labels.csv'):
    first_row = True  # Track whether it's the first row to write headers
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.png'):  # Process .png files
            file_path = os.path.join(directory, file_name)
            print("Processing..", file_name)

            # Generate a cache key for the current image file
            cache_key = generate_cache_key(file_path)
            cache_path = os.path.join(CACHE_DIR, cache_key)
            
            # Check if the cached Fourier-transformed image exists and is not expired
            if os.path.exists(cache_path):
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < cache_expiration:
                    # Load the cached Fourier image (compressed .npz file)
                    with np.load(cache_path) as data:
                        fourier_image = data['fourier_image']
                    print(f"Loaded cached Fourier transform for {file_name}")
                else:
                    # Cache expired, process the file again
                    print(f"Cache expired for {file_name}, recalculating Fourier transform.")
                    fourier_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    
                    if fourier_image is None or fourier_image.size == 0:
                        continue
                    
                    # Save the computed Fourier image to cache in a compressed format
                    np.savez_compressed(cache_path, fourier_image=fourier_image)
                    print(f"Cached Fourier transform for {file_name}")
            else:
                # No cache exists, compute the Fourier transform
                fourier_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if fourier_image is None or fourier_image.size == 0:
                    continue
                
                # Save the computed Fourier image to cache in a compressed format
                np.savez_compressed(cache_path, fourier_image=fourier_image)
                print(f"Cached Fourier transform for {file_name}")
            
            # Extract high-frequency and noise features
            high_freq_features, noise_features = extract_features(fourier_image)
            
            # Calculate the delta for both high-frequency and noise features
            delta_high_freq = np.diff(high_freq_features, axis=0)
            delta_noise = np.diff(noise_features, axis=0)
            
            # Prepare data for this row
            row = {
                'high_freq_features': high_freq_features.tolist(),
                'noise_features': noise_features.tolist(),
                'delta_high_freq': delta_high_freq.tolist(),
                'delta_noise': delta_noise.tolist(),
                'label': label,
                'numerical_label': numerical_label
            }
            
            # Write the row to CSV
            with open(output_file, mode='a', newline='') as f:
                writer = pd.DataFrame([row])
                writer.to_csv(f, header=first_row, index=False)
                first_row = False  # Subsequent rows should not have headers

            # Free memory by deleting large objects
            del fourier_image
            gc.collect()  # Explicitly call garbage collection to release memory

            # Delete the cache after processing to free up space
            os.remove(cache_path)
            print(f"Cache deleted for {file_name}")
    
# Process both malware and benign directories
process_directory(MALWARE_DIR, 'malware', 1)
process_directory(BENIGN_DIR, 'benign', 0)

print('CSV file with extracted features (high-frequency, noise, and delta) has been saved as high_freq_noise_delta_features_with_labels.csv.')
