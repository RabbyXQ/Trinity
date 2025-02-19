from transformers import pipeline

# Load trained CodeBERT model
classifier = pipeline("text-classification", model="./stegano_codebert", tokenizer="./stegano_codebert")

# Read Java code from a file
def read_java_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Specify the Java file path
file_path = "./Madman.java"  # Replace with your actual Java file

# Read Java code
java_code = read_java_file(file_path)

# Truncate the code to fit within 512 tokens
max_length = 512
if len(java_code.split()) > max_length:
    java_code = " ".join(java_code.split()[:max_length])

# Classify the Java code
result = classifier(java_code)

# Print classification result
print("Classification:", result)
