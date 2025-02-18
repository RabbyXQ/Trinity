public class Leona {
    public static void main(String[] args) {
        // Hidden message inside array indices
        int[] data = {15, 8, 20, 20, 5, 18};  // ASCII values for "H", "I", "D", "D", "E", "R"

        // Reversing the data before printing (deliberate complexity)
        for (int i = data.length - 1; i >= 0; i--) {
            char ch = (char) data[i];
            System.out.print(ch);
        }
    }
}
