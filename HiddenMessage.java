public class HiddenMessage {
    // This is a normal class with a comment to distract.
    public static void main(String[] args) {
        // Start of steganographic message
        int a = 35;
        int b = 39;
        int c = 49;
        int d = 35;
        int e = 43;
        int f = 45;

        // Hidden pattern: The sum of the numbers spells "Hello" in ASCII code
        // H=72, e=101, l=108, l=108, o=111
        
        // Further data is intentionally obfuscated
        char h = (char) (a + b - c);
        char e1 = (char) (d + e - f);
        char l1 = (char) (c + d);
        char l2 = (char) (e - a);
        char o1 = (char) (f + e - d);
        
        // Print the decoded message
        System.out.println("" + h + e1 + l1 + l2 + o1);
    }
}
