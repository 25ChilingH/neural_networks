import java.awt.image.*;
import java.io.*;

import javax.imageio.ImageIO;

public class Image2GrayBin
{
    public static final String IMG_DIR = "./imgs/";
    public static final String IMG_INPUT = ".bmp";

    public static String[] listFiles(String img_input) throws IOException
    {
        File directoryPath = new File(IMG_DIR);
        // Creating filter for jpg files
        FilenameFilter jpgFilefilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                String lowercaseName = name.toLowerCase();
                if (lowercaseName.endsWith(img_input)) {
                    return true;
                } else {
                    return false;
                }
            }
        };

        return directoryPath.list(jpgFilefilter);
    } // public static void jpgToBmp()

    public static void convertFormat(String inputImagePath, String outputImagePath) throws IOException
    {
        File inputFile = new File(inputImagePath);
        File outputFile = new File(outputImagePath);

        // reads input image from file
        BufferedImage inputImage = ImageIO.read(inputFile);
        // inputImage = inputImage.getSubimage(500, 1000, 500, 500);

        System.out.println("Original Image Dimension: " + inputImage.getWidth() + "x" + inputImage.getHeight());

        // writes to the output image in specified format
        ImageIO.write(inputImage, "bmp", outputFile);
    } 

    public static void main(String[] args) throws IOException
    {
        String[] imageFilesList = listFiles(IMG_INPUT);
        String fileName = "";
        for (String fileNameExt : imageFilesList) {
            fileName = IMG_DIR + fileNameExt.substring(0, fileNameExt.length() - IMG_INPUT.length());

            // System.out.println("Converting JPG to BMP: " + fileName);
            // convertFormat(IMG_DIR + fileNameExt, fileName + ".bmp");
            
            System.out.println("Converting BMP to Grayscale: " + fileName);
            BMP2OneByte.main(new String[]{fileName + ".bmp", fileName + ".bin"});
        }

    }
    
}
