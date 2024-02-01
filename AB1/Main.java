import java.io.*;
import java.util.*;

/*
 * September 10, 2023
 * 
 * Chiling Han
 * 
 * The main class for running and training the A-B-1 perceptron network. It loads in a
 * network configuration file that stores the settings of the network
 * and also the various training hyperparameters. Then, it prints out relevant
 * information to running or training such as the calculated truth table.
 * 
 * Example usage:
 * java Main ./OR/OR_cases.txt ./OR/OR_weights.txt
 */
public class Main
{
   public static final String DEFAULT_CONFIG_FILE = "./general.cfg"; // the default configuration file path

   /*
    * This method returns a file path to the training/test cases inputted at the command line.
    *
    * 
    * @param scan    A text scanner used to get user input for the file path
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    */
   private static String getCaseFile(Scanner scan, String[] args)
   {
      String caseFile;
      if (args.length >= 1)
      {
         caseFile = args[0];
      }
      else
      {
         System.out.print("Enter your train/test cases file path: ");
         caseFile = scan.nextLine();
      }
      return caseFile;
   } // private static String getCaseFile(Scanner scan, String[] args)

   /*
    * This method returns a file path to the weights inputted at the command line if
    * the user chooses to load instead of randomize weights.
    *
    * @param scan    A text scanner used to get user input for the file path
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    */
   private static String getWeightsFile(Scanner scan, String[] args)
   {
      String weightsFile;
      if (args.length >= 2)
      {
         weightsFile = args[1];
      }
      else
      {
         System.out.print("Enter your weights file path: ");
         weightsFile = scan.nextLine();
      }
      return weightsFile;
   } // private static String getWeightsFile(Scanner scan, String[] args)

   /*
    * The main method interfaces between the user and the perceptron. Prints the network configuration and
    * training hyperparameters before running/training. At the end of running, it will output the truth table.
    * At the end of training, it will output training exit information such as the number of training
    * iterations and the average error.
    * 
    *
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    */
   public static void main(String[] args) throws IOException
   {
      Perceptron neuralNet = new Perceptron();
      Scanner scan = new Scanner(System.in);
      Properties prop = new Properties();
      prop.load(new FileInputStream(DEFAULT_CONFIG_FILE));

      /*
       * The network configuration file './general.cfg' is documented with detail on what each String/key
       * represents in the perceptron.
       */
      String trainOrRun = prop.getProperty("TRAIN_OR_RUN");
      String randOrLoadWeights = prop.getProperty("RAND_OR_LOAD");
      neuralNet.noLayers = Integer.valueOf(prop.getProperty("NO_LAYERS"));
      neuralNet.activationString = prop.getProperty("NO_ACTIVATIONS");
      neuralNet.noCases = Integer.valueOf(prop.getProperty("NO_CASES"));
      neuralNet.maxIterations = Integer.valueOf(prop.getProperty("MAX_ITERATIONS"));
      neuralNet.errorThreshold = Double.valueOf(prop.getProperty("ERROR_THRESHOLD"));
      neuralNet.lambda = Double.valueOf(prop.getProperty("LEARNING_FACTOR"));
      neuralNet.minWeights = Double.valueOf(prop.getProperty("WEIGHTS_MIN_VAL"));
      neuralNet.maxWeights = Double.valueOf(prop.getProperty("WEIGHTS_MAX_VAL"));

      neuralNet.caseFile = getCaseFile(scan, args);
      switch (randOrLoadWeights)
      {
         case "load":
            neuralNet.weightsFile = getWeightsFile(scan, args);
            break;
         case "rand":
            neuralNet.weightsFile = "";
            break;
      }

      switch (trainOrRun)
      {
         case "train":
            neuralNet.initTrain();
            neuralNet.printTrainingConfig();
            neuralNet.trainAndReport();
            break;
         case "run":
            neuralNet.initNetwork();
            neuralNet.printNetworkConfig();
            neuralNet.runAndReport();
            break;
      } // switch (trainOrRun)

      scan.close();
   } // public static void main(String args[]) throws IOException
   
} // public class Main
