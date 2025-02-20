import zipfile
import os
import numpy as np
import cv2
import shutil
import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from androguard.core.apk import APK

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def is_valid_zip(zip_path):
    """
    Check if the ZIP file (XAPK/APK) is valid. If invalid, delete it.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            bad_file = zip_ref.testzip()
            if bad_file:
                logging.warning(f"Invalid ZIP detected in {zip_path}: {bad_file}")
                os.remove(zip_path)
                logging.info(f"Deleted corrupt file: {zip_path}")
                return False
        return True
    except zipfile.BadZipFile:
        logging.error(f"Bad ZIP file: {zip_path}")
        os.remove(zip_path)
        logging.info(f"Deleted corrupt file: {zip_path}")
        return False


def get_package_name(apk_path):
    """
    Extract the package name from an APK.
    """
    try:
        apk = APK(apk_path)
        return apk.get_package()
    except Exception as e:
        logging.error(f"Error extracting package name from {apk_path}: {e}")
        return None


def extract_xapk(xapk_path, temp_dir):
    """
    Extract APK files from an XAPK.
    """
    if not is_valid_zip(xapk_path):
        return []

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with zipfile.ZipFile(xapk_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find all APK files inside the extracted XAPK folder (including nested ones)
    apk_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.apk'):
                apk_files.append(os.path.join(root, file))

    if not apk_files:
        logging.warning(f"No APK files found in {xapk_path}")

    return apk_files


def extract_apk(apk_path, output_dir):
    """
    Extract DEX files from an APK.
    """
    if not is_valid_zip(apk_path):
        return []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dex_files_extracted = []

    with zipfile.ZipFile(apk_path, 'r') as zip_ref:
        dex_files = [file for file in zip_ref.namelist() if file.endswith('.dex')]

        for dex_file in dex_files:
            dex_path = os.path.join(output_dir, os.path.basename(dex_file))
            zip_ref.extract(dex_file, output_dir)
            os.rename(os.path.join(output_dir, dex_file), dex_path)  # Ensure correct path
            dex_files_extracted.append(dex_path)

    return dex_files_extracted


def sha256_hash(file_path):
    """
    Compute the SHA256 hash of a file.
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()


def visualize_dex_as_bitmap(dex_path, output_image_path):
    """
    Convert the binary content of the DEX file into a grayscale bitmap image.
    """
    with open(dex_path, 'rb') as file:
        dex_data = file.read()

    byte_data = np.frombuffer(dex_data, dtype=np.uint8)
    size = len(byte_data)
    side_length = int(np.ceil(np.sqrt(size)))

    byte_data = np.pad(byte_data, (0, side_length**2 - size), 'constant')
    byte_data = np.reshape(byte_data, (side_length, side_length))

    cv2.imwrite(output_image_path, byte_data)
    logging.info(f"Saved: {output_image_path}")


def process_apk(apk_path, output_folder):
    """
    Process a single APK: Extract package name, extract DEX, and generate images.
    """
    package_name = get_package_name(apk_path)

    if not package_name:
        logging.warning(f"Skipping {apk_path} due to missing package name.")
        return

    output_dir = os.path.join(output_folder, package_name)
    dex_files = extract_apk(apk_path, output_dir)

    for dex_file in dex_files:
        dex_hash = sha256_hash(dex_file)[:10]  # Use first 10 chars of SHA256 for uniqueness
        image_output_path = os.path.join(output_dir, f"{package_name}_{dex_hash}.png")
        visualize_dex_as_bitmap(dex_file, image_output_path)


def process_xapk_files(xapk_folder, output_folder):
    """
    Process all XAPK and APK files in the given folder.
    """
    xapk_files = [f for f in os.listdir(xapk_folder) if f.endswith('.xapk') or f.endswith('.apk')]

    with ProcessPoolExecutor() as executor:
        futures = []
        for xapk_file in xapk_files:
            xapk_path = os.path.join(xapk_folder, xapk_file)
            temp_dir = os.path.join(xapk_folder, 'temp_extracted')

            if xapk_file.endswith('.xapk'):
                logging.info(f"Extracting APKs from XAPK: {xapk_file}")
                apk_files = extract_xapk(xapk_path, temp_dir)
            else:
                apk_files = [xapk_path]  # Directly process APK files

            for apk_path in apk_files:
                futures.append(executor.submit(process_apk, apk_path, output_folder))

        # Wait for all processes to complete
        for future in futures:
            future.result()

    # Cleanup extracted files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logging.info(f"Removed temporary directory: {temp_dir}")


if __name__ == "__main__":
    xapk_folder = '/Volumes/Shared/rabbyx/APKS_PLAY'  # Folder containing XAPK/APK files
    output_folder = '/Volumes/Shared/rabbyx/output_images'

    process_xapk_files(xapk_folder, output_folder)
