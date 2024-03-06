import java.io.*;

public class Binary2ActFile
{
    public static final String OUTPUT_FILE = "IMG_test_cases.txt";
    public static final String BINARY_FILE = "_processed_test.bin";

    public static void write2File(String inputFile, FileWriter out, int width, int height) throws Exception
    {
        InputStream inputStream = new FileInputStream(inputFile);

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                double val = inputStream.read() / 255.0;
                out.write(Double.toString(val) + " ");
            }
        }
        inputStream.close();
    }
    public static void main(String[] args) throws Exception
    {
        FileWriter out = new FileWriter(OUTPUT_FILE);
        String[] imageProcessList = Image2GrayBin.listFiles(BINARY_FILE);
        String fileName;
        String[][] output = {{"0", "0", "0", "0", "1"},
                             {"0", "0", "0", "1", "0"},
                             {"0", "0", "1", "0", "0"},
                             {"0", "1", "0", "0", "0"},
                             {"1", "0", "0", "0", "0"}};
        for (int imgType = 1; imgType <= 5; imgType++) {
            for (int imgNum = 1; imgNum <= 1; imgNum++) {
                write2File("./imgs/binary_processed/" + imgType + "_" + imgNum + BINARY_FILE, out, 100, 100);

                for (int i = 0; i < 5; i++)
                {
                    out.write(output[imgType - 1][i] + " ");
                }
                out.write("\n");
        }
        }

        out.close();

    }
}