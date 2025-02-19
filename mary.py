import os

def crawl_for_java_files(base_dir):
    # List to store all Java file paths
    java_files = []
    
    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".java"):  # Check if the file is a .java file
                java_files.append(os.path.join(root, file))  # Add to the list
    
    return java_files

# Example usage
base_dir = "./android-java-examples/app/src/main/java/"  # Replace with your base directory
java_files = crawl_for_java_files(base_dir)

# Print the found Java files
if java_files:
    print("Found Java files:")
    for java_file in java_files:
        print(java_file)
else:
    print("No Java files found.")
