import re
import os
import javalang
import numpy as np
from collections import Counter
from scipy.stats import zscore, median_abs_deviation
from math import log2
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

stegoFiles = set()

def read_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def parse_classes_and_dependencies(text):
    try:
        tree = javalang.parse.parse(text)
    except javalang.parser.JavaSyntaxError:
        return set(), set(), set()
    
    imports, classes, method_references = set(), set(), set()
    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            classes.add(node.name)
            for method in node.methods:
                if method.return_type:
                    method_references.add(method.return_type.name)
                for param in method.parameters:
                    method_references.add(param.type.name)
        elif isinstance(node, javalang.tree.Import):
            imports.add(node.path)
    
    return classes, imports, method_references

def calculate_char_frequency(text):
    return Counter(text)

def zscore_detection(char_freq, threshold=2):
    freq_values = np.array(list(char_freq.values()))
    if len(freq_values) < 2:
        return {}
    z_scores = zscore(freq_values)
    return {list(char_freq.keys())[i]: z_scores[i] for i in range(len(z_scores)) if abs(z_scores[i]) > threshold}

def mad_detection(char_freq, threshold=3):
    freq_values = np.array(list(char_freq.values()))
    if len(freq_values) < 2:
        return {}
    mad_values = median_abs_deviation(freq_values)
    median_val = np.median(freq_values)
    anomalies = {list(char_freq.keys())[i]: abs(freq_values[i] - median_val) / mad_values for i in range(len(freq_values)) if abs(freq_values[i] - median_val) / mad_values > threshold}
    return anomalies

def calculate_entropy(char_freq):
    total_chars = sum(char_freq.values())
    if total_chars == 0:
        return 0
    probabilities = [freq / total_chars for freq in char_freq.values() if freq > 0]
    return -sum(p * log2(p) for p in probabilities)

def fourier_analysis(char_freq):
    freq_values = np.array(list(char_freq.values()))
    if len(freq_values) < 2:
        return []
    magnitude = np.abs(fft(freq_values))
    return magnitude[:len(magnitude) // 2]

def detect_hidden_message(text):
    patterns = [
        r'[\w]+=.*?[\w]+[+\-*/%&|^][\w]+;',  # Arithmetic & bitwise ops
        r'[A-Za-z0-9+/=]{16,}'  # Base64-like encoding
    ]
    matches = [match for pattern in patterns for match in re.findall(pattern, text)]
    return matches

def detect_steganography(java_file, anomalies, entropy, fft_magnitude):
    high_freq_peaks = sum(1 for mag in fft_magnitude if mag > np.mean(fft_magnitude) * 1.5)
    if entropy > 4 or len(anomalies) > 5 or high_freq_peaks > 5:
        stegoFiles.add(java_file)

def main(file_path):
    if not os.path.exists(file_path):
        return
    text = read_java_file(file_path)
    parse_classes_and_dependencies(text)
    char_freq = calculate_char_frequency(text)
    anomalies = {**zscore_detection(char_freq, threshold=2), **mad_detection(char_freq, threshold=3)}
    entropy = calculate_entropy(char_freq)
    fft_magnitude = fourier_analysis(char_freq)
    hidden_messages = detect_hidden_message(text)
    detect_steganography(file_path, anomalies, entropy, fft_magnitude)
    if hidden_messages:
        print(f"Hidden messages detected in {file_path}: {hidden_messages}")

def crawl_for_java_files(base_dir):
    return [os.path.join(root, file) for root, _, files in os.walk(base_dir) for file in files if file.endswith(".java")]

base_dir = "./GeoPointer"
java_files = crawl_for_java_files(base_dir)

if __name__ == "__main__":
    for java_file in java_files:
        main(java_file)
    print("Files where steganography is detected:")
    print("\n".join(stegoFiles) if stegoFiles else "None")