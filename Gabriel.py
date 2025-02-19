import numpy as np
from scipy.stats import ks_2samp

def compute_entropy(byte_array):
    """Calculate Shannon entropy for a byte array."""
    byte_counts = np.bincount(byte_array, minlength=256)
    probabilities = byte_counts / np.sum(byte_counts)
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def read_java_file(file_path):
    """Read a Java file and return its byte content."""
    with open(file_path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def ks_test(file1, file2):
    """Apply the Kolmogorov-Smirnov test to two Java file byte distributions."""
    d_stat, p_value = ks_2samp(file1, file2)
    return d_stat, p_value

def detect_steganography(benign_file, stego_file, threshold=.1):
    """Detect steganography using entropy and Kolmogorov-Smirnov test."""
    benign_data = read_java_file(benign_file)
    stego_data = read_java_file(stego_file)

    benign_entropy = compute_entropy(benign_data)
    stego_entropy = compute_entropy(stego_data)

    print(f"Benign Entropy: {benign_entropy:.4f}")
    print(f"Steganographic Entropy: {stego_entropy:.4f}")

    d_statistic, p_value = ks_test(benign_data, stego_data)
    print(f"Kolmogorov-Smirnov D-statistic: {d_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    if d_statistic > threshold:
        print("⚠️ Steganography Detected!")
    else:
        print("✅ No Steganography Found.")

# Example Usage
if __name__ == "__main__":
    benign_file = "Leopard.java"
    stego_file = "SimpleMessage.java"
    detect_steganography(benign_file, stego_file)
