import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.Scanner;

public class ProcessGray {

    public static final String BIN_INPUT = ".bin";
    public static final int IMG_HEIGHT = 1000;
    public static final int IMG_WIDTH = 1000;
    public static final int[] CROP_1_3_4 = {450, 150, 950, 650};
    public static final int[] CROP_2_5 = {200, 250, 800, 850};

    public static int[][] create2DIntMatrixFromFile(String inputFile, int width, int height) throws Exception
    {
        int[][] matrix = new int[width][height];
        InputStream inputStream = new FileInputStream(inputFile);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                int val = (int) inputStream.read();
                if (val != -1)
                    matrix[i][j] = val;
            }
        }

        inputStream.close();
        return matrix;
    }

    public static void write2Binary(PelArray pixels, String outFile) throws Exception
    {
        int[][] mat = pixels.getPelArray();
        FileOutputStream fOutStream = new FileOutputStream(outFile);
        DataOutputStream out = new DataOutputStream(fOutStream);

        for (int i = 0; i < mat.length; i++)
        {
            for (int j = 0; j < mat[0].length; j++)
            {
                out.writeByte((byte) mat[i][j]);
            }
        }
    }

    public static boolean check134(String fileNameExt)
    {
        return fileNameExt.charAt(0) == '1' || fileNameExt.charAt(0) == '3' || fileNameExt.charAt(0) == '4';
    }
    public static void main(String[] args) throws Exception
    {
        PelArray pixels = new PelArray();
        PelArray pixelsProcess;
        String fileName;
        String[] imageFilesList = Image2GrayBin.listFiles(BIN_INPUT);

        int xCom, yCom;

        for (String fileNameExt : imageFilesList)
        {
            fileName = Image2GrayBin.IMG_DIR + fileNameExt.substring(0, fileNameExt.length() - BIN_INPUT.length());

            pixels.setPelArray(create2DIntMatrixFromFile(Image2GrayBin.IMG_DIR + fileNameExt, IMG_WIDTH, IMG_HEIGHT));
            pixelsProcess = pixels;
            // pixelsProcess = pixels.rotateCCW90();
            // pixelsProcess = check134(fileNameExt) ? pixelsProcess.crop(CROP_1_3_4[0], CROP_1_3_4[1], CROP_1_3_4[2], CROP_1_3_4[3]) : pixelsProcess.crop(CROP_2_5[0], CROP_2_5[1], CROP_2_5[2], CROP_2_5[3]);
            pixelsProcess = pixelsProcess.forceMin(70, PelArray.BLACK);
            // write2Binary(pixelsProcess, fileName + "_beforeCOM" + BIN_INPUT);

            // xCom = pixelsProcess.getXcom();
            // yCom = pixelsProcess.getYcom();
            // System.out.println(xCom + ", " + yCom);
            // pixelsProcess = pixelsProcess.crop(Math.max(0, xCom - 250), Math.max(0, yCom - 250), Math.min(500, xCom + 250), Math.min(500, yCom + 250));
            // pixelsProcess = pixelsProcess.scale(100, 100);
            write2Binary(pixelsProcess, fileName + "_processed" + BIN_INPUT);
            System.out.printf("Finished %s processing\n", fileName);
        }
    }
}
