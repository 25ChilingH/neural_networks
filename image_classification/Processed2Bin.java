import java.io.IOException;

public class Processed2Bin
{
      public static void main(String[] args) throws IOException
      {
            String[] imageProcessList = Image2GrayBin.listFiles("_processed.bin");
            String[] argsToBGR2BMP;
            String fileName;

            for (String fileNameExt : imageProcessList)
            {
                  fileName = Image2GrayBin.IMG_DIR + fileNameExt.substring(0, fileNameExt.length() - ProcessGray.BIN_INPUT.length());
                  argsToBGR2BMP = new String[] {"gray", Integer.toString(100), Integer.toString(100), Image2GrayBin.IMG_DIR + fileNameExt, fileName + ".bmp"};
                  BGR2BMP.main(argsToBGR2BMP);
            }
      }
}