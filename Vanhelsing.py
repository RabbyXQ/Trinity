import os
import random
import base64
import string

# Configuration
OUTPUT_DIR = "./stego"
NUM_STEGANOGRAPHIC_FILES = 200  # Number of files
LARGE_FILE_SIZE = (500, 1000)  # Java file line range

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hidden messages for steganography
HIDDEN_MESSAGES = [
    "StegoPayload", "SecretCode", "HiddenData", "MalwareString", 
    "Base64CovertMessage", "UnicodeStego", "BinaryEmbeddedData"
]

# Function to generate a random Java class name
def random_classname():
    return "StegoJava" + str(random.randint(1000, 9999))

# Function to generate a random string
def random_string(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to generate a large Java filler code
def generate_large_code():
    lines = random.randint(*LARGE_FILE_SIZE)
    return "\n".join(f"    System.out.println(\"Line {i}\");" for i in range(lines))

# Unicode Zero-Width Steganography
def zero_width_steganography(message):
    """Converts a message into Zero-Width Unicode characters"""
    zero_width_chars = {
        "0": "\u200B",  # Zero Width Space
        "1": "\u200C"   # Zero Width Non-Joiner
    }
    binary = ''.join(format(ord(c), '08b') for c in message)
    return ''.join(zero_width_chars[b] for b in binary)

# Function to encode an image file in Base64 (for image steganography)
def encode_image():
    """Encodes an image into Base64 (for embedding inside Java files)"""
    sample_image = "sample.jpg"
    if not os.path.exists(sample_image):
        with open(sample_image, "wb") as f:
            f.write(os.urandom(1024))  # Create a random image-like file

    with open(sample_image, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Steganographic Java templates
STEGANOGRAPHIC_TEMPLATES = [
    # 1. Hidden Message in Comments
    """public class {classname} {{
    public static void main(String[] args) {{
        // Hidden message: {hidden_message}
        {filler_code}
    }}
}}
""",

    # 2. Hidden Message in Base64
    """import java.util.Base64;
public class {classname} {{
    public static void main(String[] args) {{
        String encoded = "{encoded_message}"; // Encoded hidden message
        byte[] decoded = Base64.getDecoder().decode(encoded);
        System.out.println("Running Java program...");
    }}
}}
""",

    # 3. Hidden Message in Variable Names
    """public class {classname} {{
    public static void main(String[] args) {{
        String {hidden_message} = "Just a normal variable";
        {filler_code}
    }}
}}
""",

    # 4. Hidden Message in File Paths
    """import java.io.File;
public class {classname} {{
    public static void main(String[] args) {{
        File secretFile = new File("{hidden_message}.txt"); // Steganographic file path
        {filler_code}
    }}
}}
""",

    # 5. Unicode Zero-Width Steganography in Strings
    """public class {classname} {{
    public static void main(String[] args) {{
        String hidden = "{zero_width_message}"; // Stego using Zero-Width Unicode
        {filler_code}
    }}
}}
""",

    # 6. Base64 Encoded Image Embedded in Code
    """import java.util.Base64;
public class {classname} {{
    public static void main(String[] args) {{
        String base64Image = "{encoded_image}"; // Stego Image
        System.out.println("Image steganography applied.");
        {filler_code}
    }}
}}
"""
]

# Function to create steganographic Java files
def create_steganographic_files():
    for i in range(NUM_STEGANOGRAPHIC_FILES):
        classname = random_classname()
        template = random.choice(STEGANOGRAPHIC_TEMPLATES)
        hidden_message = random.choice(HIDDEN_MESSAGES)
        encoded_message = base64.b64encode(hidden_message.encode()).decode()
        zero_width_message = zero_width_steganography(hidden_message)
        encoded_image = encode_image()
        filler_code = generate_large_code()

        java_code = template.format(
            classname=classname,
            hidden_message=hidden_message,
            encoded_message=encoded_message,
            zero_width_message=zero_width_message,
            encoded_image=encoded_image,
            filler_code=filler_code
        )

        filename = os.path.join(OUTPUT_DIR, f"stego_{classname}.java")

        with open(filename, "w") as f:
            f.write(java_code)

# Generate the dataset
create_steganographic_files()

print(f"✅ Steganographic dataset created in '{OUTPUT_DIR}' with {NUM_STEGANOGRAPHIC_FILES} Java files.")
