import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageSteganography {

    // Method to encode the secret message in the image
    public static void encodeMessage(BufferedImage image, String message, String outputImagePath) throws IOException {
        int width = image.getWidth();
        int height = image.getHeight();
        
        // Convert the message to a binary string (ASCII representation)
        String binaryMessage = stringToBinary(message) + "1111111111111110"; // Appending a delimiter
        
        int messageIndex = 0;
        
        // Loop through all the pixels of the image and encode the message
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Get the RGB value of the pixel
                int pixel = image.getRGB(x, y);
                
                // Extract the red, green, and blue components
                int red = (pixel >> 16) & 0xFF;
                int green = (pixel >> 8) & 0xFF;
                int blue = pixel & 0xFF;

                // Modify the LSB of each color component with the message bits
                if (messageIndex < binaryMessage.length()) {
                    red = (red & 0xFE) | (binaryMessage.charAt(messageIndex) - '0');
                    messageIndex++;
                }
                if (messageIndex < binaryMessage.length()) {
                    green = (green & 0xFE) | (binaryMessage.charAt(messageIndex) - '0');
                    messageIndex++;
                }
                if (messageIndex < binaryMessage.length()) {
                    blue = (blue & 0xFE) | (binaryMessage.charAt(messageIndex) - '0');
                    messageIndex++;
                }

                // Set the new pixel value back to the image
                int newPixel = (red << 16) | (green << 8) | blue;
                image.setRGB(x, y, newPixel);
                
                // If the entire message has been encoded, break out of the loop
                if (messageIndex >= binaryMessage.length()) {
                    break;
                }
            }
            if (messageIndex >= binaryMessage.length()) {
                break;
            }
        }
        
        // Write the modified image to the output file
        File outputFile = new File(outputImagePath);
        ImageIO.write(image, "png", outputFile);
    }

    // Method to decode the hidden message from the image
    public static String decodeMessage(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        StringBuilder binaryMessage = new StringBuilder();
        
        // Loop through all the pixels of the image and extract the LSB of each color component
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = image.getRGB(x, y);
                
                // Extract the red, green, and blue components
                int red = (pixel >> 16) & 0xFF;
                int green = (pixel >> 8) & 0xFF;
                int blue = pixel & 0xFF;

                // Extract the LSB of each color component and build the binary message
                binaryMessage.append(red & 1);
                binaryMessage.append(green & 1);
                binaryMessage.append(blue & 1);
            }
        }
        
        // Convert the binary message to a string
        String binaryMessageString = binaryMessage.toString();
        
        // Find the delimiter and remove the binary data after it
        int delimiterIndex = binaryMessageString.indexOf("1111111111111110");
        if (delimiterIndex != -1) {
            binaryMessageString = binaryMessageString.substring(0, delimiterIndex);
        }
        
        // Convert the binary string back to a normal string (ASCII)
        return binaryToString(binaryMessageString);
    }

    // Helper method to convert a string to binary
    public static String stringToBinary(String message) {
        StringBuilder binaryString = new StringBuilder();
        for (char c : message.toCharArray()) {
            binaryString.append(String.format("%8s", Integer.toBinaryString(c)).replace(' ', '0'));
        }
        return binaryString.toString();
    }

    // Helper method to convert binary back to a string
    public static String binaryToString(String binaryString) {
        StringBuilder message = new StringBuilder();
        for (int i = 0; i < binaryString.length(); i += 8) {
            String byteString = binaryString.substring(i, i + 8);
            message.append((char) Integer.parseInt(byteString, 2));
        }
        return message.toString();
    }

    public static void main(String[] args) {
        try {
            // Load the image where the message will be hidden
            String imagePath = "inputImage.png";  // Replace with your image file
            BufferedImage image = ImageIO.read(new File(imagePath));

            // Step 1: Encode a secret message into the image
            String secretMessage = "This is a hidden message!";
            String outputImagePath = "outputImage.png";  // Replace with the desired output file path
            encodeMessage(image, secretMessage, outputImagePath);
            System.out.println("Message hidden in image successfully!");

            // Step 2: Decode the hidden message from the image
            BufferedImage decodedImage = ImageIO.read(new File(outputImagePath));
            String decodedMessage = decodeMessage(decodedImage);
            System.out.println("Decoded Message: " + decodedMessage);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
