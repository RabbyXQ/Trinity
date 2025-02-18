import re
import os
import javalang
import numpy as np
from collections import Counter
from scipy.stats import zscore
from math import log2
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Step 1: Read the Java file
def read_java_file(file_path):
    """Reads the content of a Java file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 2: Parse Java Classes and Dependencies using javalang
def parse_classes_and_dependencies(text):
    """Parses Java classes, interfaces, and their dependencies using javalang."""
    
    # Parse the Java source code using javalang
    tree = javalang.parse.parse(text)
    
    # Extract classes and interfaces from the parse tree
    classes = []
    imports = set()
    method_references = set()

    for node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            classes.append(node.name)
            # Check if the class has any references to other classes in its fields or methods
            for method in node.methods:
                for param in method.parameters:
                    method_references.add(param.type.name)
                if method.return_type:
                    method_references.add(method.return_type.name)

        elif isinstance(node, javalang.tree.Import):
            # Collect imported classes
            imports.add(node.path.split('.')[-1])

    return classes, imports, method_references

# Step 3: Calculate character frequency in the file
def calculate_char_frequency(text):
    """Calculates the frequency of characters in the provided text."""
    char_freq = Counter(text)
    return char_freq

# Step 4: Perform anomaly detection using Z-Score, GESD, and Quantile-based methods
def zscore_detection(char_freq, threshold=2):
    """Detect anomalies using Z-Score method."""
    freq_values = np.array(list(char_freq.values()))
    z_scores = zscore(freq_values)
    anomalies = {list(char_freq.keys())[i]: z_scores[i] for i in range(len(z_scores)) if abs(z_scores[i]) > threshold}
    return anomalies

def gesd_detection(char_freq, threshold=4):
    """Detect anomalies using GESD method."""
    freq_values = np.array(list(char_freq.values()))
    mean = np.mean(freq_values)
    std_dev = np.std(freq_values)
    
    gesd_scores = [(abs(value - mean) / std_dev) for value in freq_values]
    anomalies = {list(char_freq.keys())[i]: gesd_scores[i] for i in range(len(gesd_scores)) if gesd_scores[i] > threshold}
    return anomalies

def quantile_detection(char_freq, lower_multiplier=1.5, upper_multiplier=1.5):
    """Detect anomalies using Quantile-based method."""
    freq_values = np.array(list(char_freq.values()))
    q25, q75 = np.percentile(freq_values, 25), np.percentile(freq_values, 75)
    iqr = q75 - q25
    lower_bound = q25 - lower_multiplier * iqr
    upper_bound = q75 + upper_multiplier * iqr
    
    anomalies = {list(char_freq.keys())[i]: freq_values[i] for i in range(len(freq_values)) 
                 if freq_values[i] < lower_bound or freq_values[i] > upper_bound}
    return anomalies

def calculate_entropy(char_freq):
    """Calculate entropy of character frequencies."""
    total_chars = sum(char_freq.values())
    
    if total_chars == 0:
        return 0  # Return 0 entropy if no characters are found
    
    # Filter out spaces and common characters to calculate entropy based on code-specific patterns
    filtered_freq = {char: freq for char, freq in char_freq.items() if char not in [" ", "\n", "\t", "#", ";"]}
    
    if len(filtered_freq) == 0:
        return 0  # Handle case where filtered frequencies are empty
    
    probabilities = [freq / total_chars for freq in filtered_freq.values()]
    
    # Calculate entropy
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

# Step 5: Perform Fourier Analysis on character frequencies (optional)
def fourier_analysis(char_freq):
    """Perform Fourier transform analysis on character frequencies."""
    freq_values = np.array(list(char_freq.values()))
    N = len(freq_values)
    freq = fftfreq(N)
    magnitude = np.abs(fft(freq_values))
    
    # Plot the FFT magnitude
    plt.figure(figsize=(10, 6))
    plt.plot(freq[:N // 2], magnitude[:N // 2])  # Plot only the positive frequencies
    plt.title("Fourier Transform of Character Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()
    
    return magnitude

# Step 6: Plot character frequencies
def plot_char_frequencies(char_freq):
    """Plot the frequency of characters in a file."""
    plt.figure(figsize=(10, 6))
    plt.bar(char_freq.keys(), char_freq.values())
    plt.title("Character Frequency Distribution")
    plt.xlabel("Character")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.show()

# Step 7: Pattern Recognition for Steganography (Arithmetic operations)
def detect_hidden_message(text):
    """Detects arithmetic operations that may be used for encoding hidden messages."""
    pattern = r'\(.*\s*\+\s*.*\s*-\s*.*\)'
    
    matches = re.findall(pattern, text)
    
    if matches:
        print("Potential hidden message pattern found:")
        for match in matches:
            print("Match: ", match)
    else:
        print("No hidden message pattern detected.")

# Step 8: Detect steganography based on anomalies and entropy
def detect_steganography(all_anomalies, entropy):
    """Detect steganography based on anomalies and entropy."""
    # Using entropy to determine if the file is "normal"
    if entropy < 4 and len(all_anomalies) < 5:
        print("No Steganography Detected.")
    else:
        print("Steganography Likely Detected!")

# Step 9: Run detection
def main(file_path):
    """Main function to check for steganography in a Java file."""
    if not os.path.exists(file_path):
        print("File not found!")
        return
    
    # Step 1: Read the Java file
    text = read_java_file(file_path)
    
    # Step 2: Parse Java classes and dependencies using javalang
    classes, imports, method_references = parse_classes_and_dependencies(text)
    print(f"\nClasses in the file: {classes}")
    print(f"Imported classes: {imports}")
    print(f"Referenced classes in methods: {method_references}")
    
    # Step 3: Calculate character frequencies
    char_freq = calculate_char_frequency(text)
    
    # Step 4: Run all anomaly detection methods with adjusted thresholds
    print("\nRunning GESD Detection...")
    gesd_anomalies = gesd_detection(char_freq, threshold=4)
    if gesd_anomalies:
        print("GESD Anomalies Detected:", gesd_anomalies)
    
    print("\nRunning Z-Score Detection...")
    zscore_anomalies = zscore_detection(char_freq, threshold=2)
    if zscore_anomalies:
        print("Z-Score Anomalies Detected:", zscore_anomalies)
    
    print("\nRunning Quantile Detection...")
    quantile_anomalies = quantile_detection(char_freq, lower_multiplier=1.5, upper_multiplier=1.5)
    if quantile_anomalies:
        print("Quantile-based Anomalies Detected:", quantile_anomalies)
    
    # Step 5: Calculate entropy
    entropy = calculate_entropy(char_freq)
    print("\nEntropy of Character Frequencies:", entropy)
    
    # Step 6: Perform Fourier Analysis (optional for more in-depth analysis)
    print("\nPerforming Fourier Analysis...")
    fft_magnitude = fourier_analysis(char_freq)
    
    # Step 7: Optional: Plot the character frequencies
    plot_char_frequencies(char_freq)

    # Step 8: Detect hidden message patterns
    detect_hidden_message(text)
    
    # Step 9: Detect steganography based on anomalies and entropy
    all_anomalies = {**gesd_anomalies, **zscore_anomalies, **quantile_anomalies}
    detect_steganography(all_anomalies, entropy)

if __name__ == "__main__":
    file_path = "SimpleMessage.java"  # Replace with the actual file path
    main(file_path)
