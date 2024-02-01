import java.io.*;
import java.util.*;

/*
 * September 15, 2023
 * 
 * Chiling Han
 * 
 * The class serves as the architecture for a A-B-C perceptron. It will configure the network,
 * including the number of layers, the number of activation units, the initial weights of
 * the perceptron, and other training hyperparameters.
 * 
 * Furthermore, this class can train and run a perceptron by calculating the values of final outputs
 * given any number of inputs and also comparing them to the corresponding user-given expected values.
 * 
 * int noLayers               - Number of layers in the network
 * int[] noActivations        - Number of activations in each layer
 * String caseFile            - The truth table of the case being tested
 * int noCases                - The number of train/test cases
 * int maxIterations          - Maximum number of iterations allowed
 * double errorThreshold      - The minimum error to reach before ending the training cycle
 * double lambda              - The learning factor used to control how much the weights are modified for each step
 * String weightsFile         - A file containing the weights to use for adjusting activation values
 * double minWeights          - The lower end of the range of weights used
 * double maxWeights          - The upper end of the range of weights used
 * String trainOrRun          - Determines whether we are training or running the perceptron
 * String randOrLoad          - Determines whether we are randomizing or loading weights to the perceptron
 * boolean exportWeights      - Determines whether we are saving weights to a file at the end of training
 * double avgError            - The average error of the errors returned from running all test cases
 * int epoch                  - Counts the current number of iterations
 * int[] noActivations        - Number of activations in each layer
 * int[][] caseIn             - Input values for training/test cases
 * double[][] caseOut         - Output values for training/test cases
 * double[][][] weights       - Weight values used to adjust the activation values
 * double[][][] deltaWeights  - The change in weights between each step
 * double[][] activations     - Stores the values for each activation node
 * double[][] thetas          - The dot products of each activation value and its corresponding weights (theta in design doc)
 * double[][] outputs         - The output values of the network after training
 * double[][] psi             - Represents lower case psis in the design doc
 */
public class Perceptron
{
   public int noLayers;
   public int[] noActivations;

   public String caseFile;
   public int noCases;

   public int maxIterations;
   public double errorThreshold;
   public double lambda;

   public String weightsFile;
   public double minWeights;
   public double maxWeights;
   public String saveWeightsFile;

   public String trainOrRun, randOrLoadWeights;
   public boolean exportWeights;

   private double avgError;
   private int epoch;

   private double[][] caseIn;
   private double[][] caseOut;
   private double[][][] weights, deltaWeights;
   private double[][] activations, thetas;
   private double[][] outputs;
   private double[] psi;

   public static final String DEFAULT_CONFIG_FILE = "./general.cfg"; // the default configuration file path

   /*
    * Sets the network configuration parameters and stores them as instance variables
    * in this class. The network configuration file './general.cfg' is documented with
    * detail on what each String/key represents in the perceptron.
    * 
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    * @throws IOException    if there is an error during input or output operations
    */
   public void setConfigParams(String[] args) throws IOException
   {
      Scanner scan = new Scanner(System.in);
      Properties prop = new Properties();
      prop.load(new FileInputStream(DEFAULT_CONFIG_FILE));
      
      saveWeightsFile = prop.getProperty("SAVE_WEIGHTS_FILE");
      exportWeights = saveWeightsFile.length() >= 1;
      noLayers = Integer.valueOf(prop.getProperty("NO_LAYERS"));
      noActivations = getNoActivations(prop.getProperty("NO_ACTIVATIONS"), noLayers);
      noCases = Integer.valueOf(prop.getProperty("NO_CASES"));
      maxIterations = Integer.valueOf(prop.getProperty("MAX_ITERATIONS"));
      errorThreshold = Double.valueOf(prop.getProperty("ERROR_THRESHOLD"));
      lambda = Double.valueOf(prop.getProperty("LEARNING_FACTOR"));
      minWeights = Double.valueOf(prop.getProperty("WEIGHTS_MIN_VAL"));
      maxWeights = Double.valueOf(prop.getProperty("WEIGHTS_MAX_VAL"));
      caseFile = getCaseFile(scan, args);

      randOrLoadWeights = prop.getProperty("RAND_OR_LOAD");
      switch (randOrLoadWeights)
      {
         case "load":
            weightsFile = getWeightsFile(scan, args);
            break;
         case "rand":
            weightsFile = "";
            break;
      }

      trainOrRun = prop.getProperty("TRAIN_OR_RUN");
   } // public void setConfigParams(String[] args) throws IOException

   /*
    * Initializes the network by calling a method to allocate memory for arrays to run the network
    * and allocates memory to training-specific arrays.
    *
    */
   public void allocateTrainingArrays()
   {
      allocateRunningArrays();

      deltaWeights = new double[noLayers][][];
      for (int n = 0; n < noLayers; n++)
      {
         deltaWeights[n] = new double[noActivations[n]][noActivations[n + 1]];
      }

      thetas = new double[noLayers + 1][];
      for (int n = 0; n <= noLayers; n++)
      {
         thetas[n] = new double[noActivations[n]];
      }

      psi = new double[noActivations[noLayers]];

   } // public void allocateTrainingArrays()

   /*
    * Initializes the network by allocating memory to the arrays necessary for running the network.
    * This method also loads in values from files supplied by the user.
    *
    */  
   public void allocateRunningArrays()
   {
      activations = new double[noLayers + 1][];
      for (int n = 0; n <= noLayers; n++)
      {
         activations[n] = new double[noActivations[n]];
      }

      caseIn = new double[noCases][noActivations[0]];
      caseOut = new double[noCases][noActivations[noLayers]];

      weights = new double[noLayers][][];
      for (int n = 0; n < noLayers; n++)
      {
         weights[n] = new double[noActivations[n]][noActivations[n + 1]];
      }

      outputs = new double[noCases][noActivations[noLayers]];

   } // public void allocateRunningArrays()

   /*
    * Fills in the values for each array
    *
    * @throws FileNotFoundException    if a provided file is invalid or no file is provided
    */
   public void populateArrays() throws FileNotFoundException
   {
      Scanner scan = new Scanner(new File(caseFile));
      String[] caseInOut;
      int k;
      for (int ii = 0; ii < noCases; ii++)
      {
         caseInOut = scan.nextLine().split(" ");
 
         for (k = 0; k < noActivations[0]; k++)
         {
            caseIn[ii][k] = Double.valueOf(caseInOut[k]);
         }
         
         for (int i = k; i < noActivations[0] + noActivations[noLayers]; i++)
         {
            caseOut[ii][i - k] = Double.valueOf(caseInOut[i]);
         }

      } // for (int ii = 0; ii < noCases; ii++)

      boolean load = weightsFile.length() >= 1;
      
      if (load)
         loadWeightsFromFile();
      else
         randomizeWeights();

   } // public void populateArrays()

   /*
    * Displays the running network configuration including number of activations and
    * the truth table corresponding to the selected network.
    */
   public void echoRunConfig()
   {
      System.out.println("FILE CONFIGURATION\n-------------------------");

      System.out.println("Path to configuration file: '" + DEFAULT_CONFIG_FILE + "'");

      switch (randOrLoadWeights)
      {
         case "load":
            System.out.println("Path to weights file to load from: '" + weightsFile + "'");
            break;
         default:
            System.out.println("Will not load weights");
            break;
      }

      System.out.println("Will not save weights");

      System.out.println("\nNETWORK CONFIGURATION\n-------------------------");

      System.out.println("Activations:");
      printActivations();

      System.out.println("\nTRUTH TABLE (T)");
      printTruthTable(caseOut);
   } // public void printNetworkConfig()

   /*
    * Displays the training network configuration and prints the hyperparameters.
    */
   public void echoTrainingConfig()
   {
      System.out.println("FILE CONFIGURATION\n-------------------------");

      System.out.println("Path to configuration file: '" + DEFAULT_CONFIG_FILE + "'");

      switch (randOrLoadWeights)
      {
         case "load":
            System.out.println("Path to weights file to load from: '" + weightsFile + "'");
            break;
         default:
            System.out.println("Will not load weights");
            break;
      }

      if (exportWeights)
         System.out.println("Path to weights file to save to: '" + saveWeightsFile + "'");
      else
         System.out.println("Will not save weights");

      System.out.println("\nNETWORK CONFIGURATION\n-------------------------");

      System.out.println("Activations:");
      printActivations();

      System.out.println("\nTRUTH TABLE (T)");
      printTruthTable(caseOut);

      System.out.println("\nTRAINING PARAMETERS\n-------------------------");
      System.out.println("Weights range: " + minWeights + " to " + maxWeights);
      System.out.println("Maximum iterations: " + maxIterations);
      System.out.println("Error threshold: " + errorThreshold);
      System.out.println("Lambda value: " + lambda);
   } // public void printTrainingConfig()

   /*
    * Runs the networks and displays the results.
    */
   public void runAndReport()
   {
      runNetwork();
      System.out.println("\nTRUTH TABLE (F)");
      printTruthTable(outputs);
   }

   /*
    * Trains the network and displays the results. Saves the weights in a file if a file
    * path is provided.
    *
    * @throws IOException    if there is an error during input or output operations
    */
   public void trainAndReport() throws IOException
   {
      long start = System.currentTimeMillis();
      trainNetwork();
      long end = System.currentTimeMillis();

      runAndReport();

      if (exportWeights)
         writeWeights();

      System.out.println("\nTRAINING EXIT INFO\n-------------------------");
      System.out.print("Training stopped because: ");

      if (avgError <= errorThreshold)
         System.out.println("average error is below the error threshold.");
      else if (epoch >= maxIterations)
         System.out.println("maximum iterations exceeded");

      System.out.println("Mean Error: " + avgError);
      System.out.println("Iterations reached: " + epoch);
      System.out.println("Time of training (ms): " + (end - start));
   } // public void trainAndReport() throws IOException

   /*
    * Trains the network using steepest descent to minimize the error value and near the target values.
    * 
    * Training ends if the maximum number of iterations is reached or the calculated error
    * is below the error threshold.
    *
    * @throws IOException    if there is an error during input or output operations
    */
   public void trainNetwork() throws IOException
   {
      epoch = 0;
      avgError = Double.MAX_VALUE;

      while (epoch < maxIterations && avgError > errorThreshold)
      {
         for (int ii = 0; ii < noCases; ii++)
         {
            train(ii);
         }

         avgError = meanError();
         epoch++;
      }
   } // public void trainNetwork() throws IOException

   /*
    * Runs the network for all the given test cases
    */
   public void runNetwork()
   {
      for (int ii = 0; ii < noCases; ii++)
      {
         run(ii);
      }
   }

   /*
    * Trains the network for the given case and adjusts the values of the weights based on the learning factor
    *
    * @param caseIndex  the given training case to train the network on
    *
    */
   private void train(int caseIndex)
   {
      runTrain(caseIndex);

      updateDeltaWeights(caseIndex);

      for (int n = 0; n < noLayers; n++)
      {
         for (int k = 0; k < noActivations[n]; k++)
         {
            for (int j = 0; j < noActivations[n + 1]; j++)
            {
               weights[n][k][j] += deltaWeights[n][k][j];
            }
         }
      }
   } // private void train(int caseIndex)
   
   /*
    * Runs the network by calculating the output values for the given test case
    *
    * @param caseIndex  the given test case to run the network on
    */
   private void run(int caseIndex)
   {
      double dotProduct;

      for (int k = 0; k < noActivations[0]; k++)
      {
         activations[0][k] = caseIn[caseIndex][k];
      }

      for (int n = 0; n < noLayers; n++)
      {
         for (int j = 0; j < noActivations[n + 1]; j++)
         {
            dotProduct = 0.0;

            for (int k = 0; k < noActivations[n]; k++)
            {
               dotProduct += activations[n][k] * weights[n][k][j];
            }
            activations[n + 1][j] = activationF(dotProduct);
         }
      } // for (int n = 0; n < noLayers; n++)

      for (int i = 0; i < noActivations[noLayers]; i++)
      {
         outputs[caseIndex][i] = activations[noLayers][i];
      }
   } // private void run(int caseIndex)

   /*
    * Runs the network by calculating the output values for the given test case
    * and stores the dot products in the thetas array.
    *
    * @param caseIndex  the given test case to run the network on
    */
   private void runTrain(int caseIndex)
   {
      for (int k = 0; k < noActivations[0]; k++)
      {
         thetas[0][k] = caseIn[caseIndex][k];
         activations[0][k] = caseIn[caseIndex][k];
      }

      for (int n = 0; n < noLayers; n++)
      {
         for (int j = 0; j < noActivations[n + 1]; j++)
         {
            thetas[n + 1][j] = 0.0;

            for (int k = 0; k < noActivations[n]; k++)
            {
               thetas[n + 1][j] += activations[n][k] * weights[n][k][j];
            }
            activations[n + 1][j] = activationF(thetas[n + 1][j]);
         }
      } // for (int n = 0; n < noLayers; n++)

      for (int i = 0; i < noActivations[noLayers]; i++)
      {
         outputs[caseIndex][i] = activations[noLayers][i];
      }

   } // private void runTrain(int caseIndex)

   /*
    * Updates the change in weights based on updated activation node values and the partial derivative
    * of the error.
    *
    * @param caseIndex  the given training case to update the weights of the network for
    *
    */
   private void updateDeltaWeights(int caseIndex)
   {
      double omega, capitalOmega, capitalPsi;

      for (int i = 0; i < noActivations[noLayers]; i++)
      {
         omega = caseOut[caseIndex][i] - outputs[caseIndex][i];
         psi[i] = omega * derivActivationF(thetas[noLayers][i]);
   
         for (int j = 0; j < noActivations[noLayers - 1]; j++)
         {
            deltaWeights[noLayers - 1][j][i] = psi[i] * -activations[noLayers - 1][j] * -lambda;
         }
      }
      
      for (int n = 1; n < noLayers; n++)
      {
         for (int j = 0; j < noActivations[n]; j++)
         {
            capitalOmega = 0.0;

            for (int i = 0; i < noActivations[noLayers]; i++)
            {
               capitalOmega += psi[i] * weights[n][j][i];
            }

            for (int k = 0; k < noActivations[n - 1]; k++)
            {
               capitalPsi = derivActivationF(thetas[n][j]) * capitalOmega;
               deltaWeights[n - 1][k][j] = capitalPsi * -activations[n - 1][k] * -lambda;
            }
         } // for (int j = 0; j < noActivations[n]; j++)
      } // for (int n = 1; n < noLayers; n++)

   } // public void updateDeltaWeights(int caseIndex)

   /*
    * Generates a random value for a weight within the given range
    * 
    * @param min  the lower end of the range within which weights are randomized
    * @param max  the upper end of the range within which weights are randomized
    */
   private double randWeights(double min, double max)
   {
      return Math.random() * (max - min) + min;
   }

   /*
    * Calculates the average error value based on the sum of error values returned from running all test cases
    */
   public double meanError()
   {
      double totalError = 0.0;

      for (int ii = 0; ii < noCases; ii++)
      {
         totalError += errorFunction(ii);
      }

      return totalError / (double) noCases;
   } // public double meanError()

   /*
    * Computes the error for one given case using the formula 1/2(F - T)^2
    *
    * @param caseIndex  the given training case to minimize the error function for
    */
   private double errorFunction(int caseIndex)
   {
      double error = 0.0;

      for (int i = 0; i < noActivations[noLayers]; i++)
      {
         error += (caseOut[caseIndex][i] - outputs[caseIndex][i]) * (caseOut[caseIndex][i] - outputs[caseIndex][i]);
      }

      return error / 2.0;

   } // private double errorFunction(int caseIndex)

   /*
    * Calculates the sigmoid function for a given value x
    *
    * @param x    the value passed to the sigmoid function
    */
   private double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   /*
    * Calculates the derivative of the sigmoid function for a given value x
    *
    * @param x    the value passed to the derivative of the sigmoid function
    */
   private double derivSigmoid(double x)
   {
      double val = sigmoid(x);
      return val * (1.0 - val);
   }

   /*
    * Computes an activation function
    *
    * @param x    the value passed to the activation function
    */
   private double activationF(double x)
   {
      return sigmoid(x);
   }

   /*
    * Computes the derivative of an activation function
    *
    * @param x    the value passed to the derivative of activation function
    */
   private double derivActivationF(double x)
   {
      return derivSigmoid(x);
   }

   /*
    * Displays the number of activations in each layer
    */
   private void printActivations()
   {
      for (int n = 0; n < noLayers; n++)
      {
         System.out.print(noActivations[n] + "-");
      }
      System.out.println(noActivations[noLayers]);
   }

   /*
    * Displays the values of a truth table given the outputs
    *
    * @param fOut    the given outputs to display
    */
   private void printTruthTable(double[][] fOut)
   {
      for (int ii = 0; ii < noCases; ii++)
      {
         for (int k = 0; k < noActivations[0]; k++)
         {
            System.out.print(caseIn[ii][k] + "  ");
         }

         System.out.print("| ");

         for (int i = 0; i < noActivations[noLayers]; i++)
         {
            System.out.print(fOut[ii][i] + "  ");
         }

         System.out.println();
      } // for (int ii = 0; ii < noCases; ii++)
   } // private void printTruthTable(double[][] fOut)

   /*
    * Uses a FileWriter to write weights to a user-given file
    *
    * @throws IOException    if there is an error during input or output operations
    */
   private void writeWeights() throws IOException
   {
      FileWriter fw = new FileWriter(saveWeightsFile);
   
      for (int n = 0; n < noLayers; n++)
      {
         for (int k = 0; k < noActivations[n]; k++)
         {
            for (int j = 0; j < noActivations[n + 1]; j++)
            {
               fw.write(Double.toString(weights[n][k][j]) + "\n");
            }
         }
      }

      fw.close();
   } // private void writeWeights()


   /*
    * Fills the weights array with values of weights loaded from a file.
    *
    * @throws FileNotFoundException    if a provided file is invalid or no file is provided
    */
   private void loadWeightsFromFile() throws FileNotFoundException
   {
      Scanner scan = new Scanner(new File(weightsFile));

      for (int n = 0; n < noLayers; n++)
      {
         for (int k = 0; k < noActivations[n]; k++)
         {
            for (int j = 0; j < noActivations[n + 1]; j++)
            {
               weights[n][k][j] = scan.nextDouble();
            }
         }
      } // for (int n = 0; n < noLayers; n++)

   } // private void loadWeightsFromFile() throws FileNotFoundException

   /*
    * Fills the weights array with values of weights randomized between a given 
    * range of weight values.
    */
   private void randomizeWeights()
   {
      for (int n = 0; n < noLayers; n++)
      {
         for (int k = 0; k < noActivations[n]; k++)
         {
            for (int j = 0; j < noActivations[n + 1]; j++)
            {
               weights[n][k][j] = randWeights(minWeights, maxWeights);
            }
         }
      } // for (int n = 0; n < noLayers; n++)

   } // private void randomizeWeights()

   /*
    * This method returns a file path to the training/test cases inputted at the command line.
    *
    * 
    * @param scan    A text scanner used to get user input for the file path
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    */
   private String getCaseFile(Scanner scan, String[] args)
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
   } // private String getCaseFile(Scanner scan, String[] args)
 
   /*
    * This method returns a file path to the weights inputted at the command line if
    * the user chooses to load instead of randomize weights.
    *
    * @param scan    A text scanner used to get user input for the file path
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    */
   private String getWeightsFile(Scanner scan, String[] args)
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
   } // private String getWeightsFile(Scanner scan, String[] args)
 
   /*
    * Returns the number of activations in each layer as an integer array
    *
    * @param activationString    represents the number of activations in each layer as a String
    * @param noLayers            number of layers of the network
    */
   private int[] getNoActivations(String activationString, int noLayers)
   {
      String[] activationStringArray = activationString.split(" ");
      int[] activationIntArray = new int[noLayers + 1];
      for (int n = 0; n <= noLayers; n++)
      {
         activationIntArray[n] = Integer.valueOf(activationStringArray[n]);
      }
 
      return activationIntArray;
   } // private int[] getNoActivations(int noLayers, String activationString)
 

} // public class Perceptron
