import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ScaleBMP {

   public static BufferedImage convertToBufferedImage(Image img) {

      if (img instanceof BufferedImage) {
         return (BufferedImage) img;
      }

      // Create a buffered image with transparency
      BufferedImage bi = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

      Graphics2D graphics2D = bi.createGraphics();
      graphics2D.drawImage(img, 0, 0, null);
      graphics2D.dispose();

      return bi;
   }

   public static void main(String[] args) throws IOException {
      for (int imgType = 1; imgType <= 5; imgType++) {
         for (int imgNum = 1; imgNum <= 6; imgNum++) {
            BufferedImage bmp = ImageIO.read(new File(Image2GrayBin.IMG_DIR + imgType + "_" + imgNum + ".bmp"));

            int w = 100;
            int h = 100;

            Image scaledIMG = bmp.getScaledInstance(w, h, Image.SCALE_SMOOTH);
            
            BufferedImage scaledBMP = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            scaledBMP.getGraphics().drawImage(scaledIMG, 0, 0, null);

            File scaledBMPFile = new File("./imgs/bmp_processed/" + imgType + "_" + imgNum + "_processed.bmp");
            ImageIO.write(scaledBMP, "bmp", scaledBMPFile);
         }
      }
   }

}
