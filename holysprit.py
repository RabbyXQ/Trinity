import os
import re
import javalang
import numpy as np
import joblib
from collections import Counter
from scipy.stats import zscore
from math import log2
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

# Load ML Model
MODEL_PATH = "ml_kit_model.pkl"
if os.path.exists(MODEL_PATH):
    ml_model = joblib.load(MODEL_PATH)
else:
    ml_model = None

# Step 1: Read Java file content
def read_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 2: Extract dependencies and classes
def parse_classes_and_dependencies(text):
    try:
        tree = javalang.parse.parse(text)
    except javalang.parser.JavaSyntaxError as e:
        print(f"Java syntax error: {e}")
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


# Step 3: Fetch imported Java files
def fetch_imported_files(imports, base_dir):
    java_files = {}
    for imp in imports:
        file_path = os.path.join(base_dir, *imp.split('.')) + ".java"
        if os.path.exists(file_path):
            java_files[imp] = read_java_file(file_path)
    return java_files

# Step 4: Character frequency analysis
def calculate_char_frequency(text):
    return Counter(text)

# Step 5: Anomaly detection
def detect_anomalies(char_freq):
    freq_values = np.array(list(char_freq.values())).reshape(-1, 1)
    z_scores = zscore(freq_values)
    isolation_forest = IsolationForest(contamination=0.1).fit(freq_values)
    anomaly_scores = isolation_forest.predict(freq_values)
    
    anomalies = {
        char: (z_scores[i], anomaly_scores[i])
        for i, char in enumerate(char_freq.keys())
        if abs(z_scores[i]) > 2 or anomaly_scores[i] == -1
    }
    return anomalies

# Step 6: Calculate entropy
def calculate_entropy(char_freq):
    total_chars = sum(char_freq.values())
    probabilities = [freq / total_chars for freq in char_freq.values()]
    return -sum(p * log2(p) for p in probabilities if p > 0)

# Step 7: Fourier Transform analysis
def fourier_transform_analysis(text):
    signal = np.array([ord(char) for char in text])
    fft_values = np.abs(np.fft.fft(signal))
    return fft_values

# Step 8: Cosine similarity check
def cosine_similarity_check(base_text, new_text):
    base_vector = np.array([ord(char) for char in base_text]).reshape(1, -1)
    new_vector = np.array([ord(char) for char in new_text]).reshape(1, -1)
    similarity = cosine_similarity(base_vector, new_vector)[0][0]
    return similarity

# Step 9: Detect steganography using ML
def detect_steganography(text, anomalies, entropy, fft_values):
    features = [entropy, len(anomalies), np.mean(fft_values), np.std(fft_values)]
    if ml_model:
        prediction = ml_model.predict([features])
        print("⚠️ Possible steganography detected!" if prediction[0] == 1 else "✅ No steganography detected.")
    else:
        print("⚠️ ML model not loaded, using rule-based detection.")
        if entropy > 4 or len(anomalies) > 10:
            print("⚠️ Possible steganography detected!")
        else:
            print("✅ No steganography detected.")

# Step 10: Additional anomaly detection methods
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

# Step 11: Perform Fourier Analysis on character frequencies (optional)
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

# Step 12: Plot character frequencies
def plot_char_frequencies(char_freq):
    """Plot the frequency of characters in a file."""
    plt.figure(figsize=(10, 6))
    plt.bar(char_freq.keys(), char_freq.values())
    plt.title("Character Frequency Distribution")
    plt.xlabel("Character")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.show()

# Step 13: Pattern Recognition for Steganography (Arithmetic operations)
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

# Step 14: Detect steganography based on anomalies and entropy
def detect_steganography(text, all_anomalies, entropy, fft_values):
    """Detect steganography based on anomalies and entropy."""
    # Using entropy to determine if the file is "normal"
    if entropy < 4 and len(all_anomalies) < 5:
        print("No Steganography Detected.")
    else:
        print("Steganography Likely Detected!")

# Step 15: Main function to check for steganography
def luke(base_dir, main_file):
    file_path = os.path.join(base_dir, main_file)
    if not os.path.exists(file_path):
        print("File not found!")
        return
    
    text = read_java_file(file_path)
    classes, imports, method_references = parse_classes_and_dependencies(text)
    java_files = fetch_imported_files(imports, base_dir)
    
    print(f"Classes: {classes}\nImports: {imports}\nMethod References: {method_references}")
    
    char_freq = calculate_char_frequency(text)
    anomalies = detect_anomalies(char_freq)
    entropy = calculate_entropy(char_freq)
    fft_values = fourier_transform_analysis(text)
    
    print(f"\nAnomalies detected: {anomalies}")
    print(f"Entropy: {entropy}")
    
    detect_steganography(text, anomalies, entropy, fft_values)
    
    # Plot character frequencies and Fourier transform
    plot_char_frequencies(char_freq)
    fourier_analysis(char_freq)


def analyze_all_java_files(base_dir, main_file):
    # Read and process the main file
    file_path = main_file
    if not os.path.exists(file_path):
        print("Main file not found!")
        return
    
    # Process the main file
    text = read_java_file(file_path)
    classes, imports, method_references = parse_classes_and_dependencies(text)
    print(f"Classes: {classes}\nImports: {imports}\nMethod References: {method_references}")

    # Apply z-score and GESD anomaly detection
    char_freq = calculate_char_frequency(text)
    anomalies_zscore = zscore_detection(char_freq)
    anomalies_gesd = gesd_detection(char_freq)
    anomalies_quantile = quantile_detection(char_freq)
    
    print("Anomalies detected using Z-Score:", anomalies_zscore)
    print("Anomalies detected using GESD:", anomalies_gesd)
    print("Anomalies detected using Quantile:", anomalies_quantile)

    # Perform Fourier Transform analysis
    fft_values = fourier_transform_analysis(text)
    print(f"Fourier Transform Values: {fft_values}")

    # Calculate Entropy
    entropy = calculate_entropy(char_freq)
    print(f"Entropy: {entropy}")

    # Check for potential steganography based on rules and ML
    detect_steganography(text, anomalies_zscore, entropy, fft_values)
    detect_hidden_message(text)



def crawl_for_java_files(base_dir):
    # List to store all Java file paths
    java_files = []
    
    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".java"):  # Check if the file is a .java file
                java_files.append(os.path.join(root, file))  # Add to the list
    
    return java_files


# Start the analysis on a test file (replace with an actual file path)
base_dir = "./"  # Change to the directory containing the files

java_files = crawl_for_java_files(base_dir)

if java_files:
    print("Found Java files:")
    for java_file in java_files:
        main_file = java_file
        print("Analyzing: " + java_file)
        print(end="\n")
        analyze_all_java_files(base_dir, main_file)
