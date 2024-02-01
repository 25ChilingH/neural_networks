import java.io.*;
import java.util.*;

/*
 * November 15, 2023
 * 
 * Chiling Han
 * 
 * The class serves as the architecture for an n-layer perceptron. It will configure the network,
 * including the number of layers, the number of activation units, the initial weights of
 * the perceptron, and other training hyperparameters.
 * 
 * Furthermore, this class can train by backpropagation and run a perceptron by calculating the values of final outputs
 * given any number of inputs and also comparing them to the corresponding user-given expected values.
 * 
 * public void setConfigParams(String[] args) throws IOException
 * public void allocateTrainingArrays()
 * public void allocateRunningArrays()
 * public void populateArrays() throws FileNotFoundException
 * public void echoRunConfig()
 * public void echoTrainingConfig()
 * public void runAndReport()
 * public void trainAndReport() throws IOException
 * public void trainNetwork() throws IOException
 * public void runNetwork()
 * private void train(int caseIndex)
 * private void run(int caseIndex)
 * private void runTrain(int caseIndex)
 * private void updateWeights(int caseIndex)
 * private double randWeights(double min, double max)
 * public double meanError()
 * private double errorFunction(int caseIndex)
 * private double sigmoid(double x)
 * private double derivSigmoid(double x)
 * private double activationF(double x)
 * private double derivActivationF(double x)
 * private String activationString()
 * private void printTruthTable(double[][] fOut)
 * private void writeWeights(String weightsFile) throws IOException
 * private void loadWeightsFromFile() throws FileNotFoundException
 * private void randomizeWeights()
 * private String getConfigFile(String[] args)
 * private int[] getNoActivations(String activationString, int noLayers)
 */
public class Perceptron
{
   public int noLayers;
   public int[] noActivations;

   public String configFile, caseFile;
   public int noCases;

   public int maxIterations, keepAlive;
   public double errorThreshold;
   public double lambda;

   public String weightsFile;
   public double minWeights;
   public double maxWeights;
   public String saveWeightsFile;

   public String trainOrRun, randOrLoadWeights;
   public boolean exportWeights;
   public boolean showInputs;

   private double avgError;
   private int epoch;

   private double[][] caseIn;
   private double[][] caseOut;
   private double[][][] weights;
   private double[][] activations, thetas;
   private double[][] outputs;
   private double[][] psi;

   public static final String DEFAULT_CONFIG_FILE = "./general.cfg"; // the default configuration file path
   public static final int CMD_LINE_CONFIG_I = 0; // the index of the configuration file string in the runtime arguments
   public static final int MK_INDEX = 2; // represents the index of the mk layer

   /*
    * Sets the network configuration parameters and stores them as instance variables
    * in this class. The network configuration file './general.cfg' is documented with
    * detail on what each String/key represents in the perceptron.
    * 
    * @param args            A 1D array of Strings that holds runtime arguments to configure the perceptron.
    * @throws IOException    if there is an error during input or output operations
    */
   public void setConfigParams(String[] args) throws IOException
   {
      Properties prop = new Properties();
      
      configFile = getConfigFile(args);
      prop.load(new FileInputStream(configFile));
      
      saveWeightsFile = prop.getProperty("SAVE_WEIGHTS_FILE");
      caseFile = prop.getProperty("CASE_FILE");

      exportWeights = prop.getProperty("SAVE_WEIGHTS").equals("y");
      showInputs = prop.getProperty("SHOW_INPUTS").equals("y");
      noLayers = Integer.valueOf(prop.getProperty("NO_LAYERS"));
      noActivations = getNoActivations(prop.getProperty("NO_ACTIVATIONS"), noLayers);
      noCases = Integer.valueOf(prop.getProperty("NO_CASES"));
      maxIterations = Integer.valueOf(prop.getProperty("MAX_ITERATIONS"));
      keepAlive = Integer.valueOf(prop.getProperty("KA_ITERATIONS"));
      errorThreshold = Double.valueOf(prop.getProperty("ERROR_THRESHOLD"));
      lambda = Double.valueOf(prop.getProperty("LEARNING_FACTOR"));
      minWeights = Double.valueOf(prop.getProperty("WEIGHTS_MIN_VAL"));
      maxWeights = Double.valueOf(prop.getProperty("WEIGHTS_MAX_VAL"));

      randOrLoadWeights = prop.getProperty("RAND_OR_LOAD");
      switch (randOrLoadWeights)
      {
         case "load":
            weightsFile = prop.getProperty("LOAD_WEIGHTS_FILE");
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

      thetas = new double[noLayers + 1][];
      for (int n = 0; n <= noLayers; n++)
      {
         thetas[n] = new double[noActivations[n]];
      }

      psi = new double[noLayers + 1][];
      for (int n = 0; n <= noLayers; n++)
      {
         psi[n] = new double[noActivations[n]];
      }

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

      System.out.println("Path to configuration file: '" + configFile + "'");

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
      System.out.println(activationString());

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
      System.out.println(activationString());

      System.out.println("\nTRUTH TABLE (T)");
      printTruthTable(caseOut);

      System.out.println("\nTRAINING PARAMETERS\n-------------------------");
      System.out.println("Weights range: " + minWeights + " to " + maxWeights);
      System.out.println("Maximum iterations: " + maxIterations);
      System.out.println("Error threshold: " + errorThreshold);
      System.out.println("Lambda value: " + lambda);

      System.out.println("\n-------------------------");
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
         writeWeights(saveWeightsFile);

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

         if ((epoch + 1) % keepAlive == 0)
         {
            System.out.println("Iteration #" + (epoch + 1) + " Avg Error: " + avgError);
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
      updateWeights(caseIndex);
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
         activations[0][k] = caseIn[caseIndex][k];
      }

      for (int n = 0; n < noLayers - 1; n++)
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

      double omega;
      for (int i = 0; i < noActivations[noLayers]; i++) // loop for running over the last layer and calculating psi values
      {
         thetas[noLayers][i] = 0.0;

         for (int j = 0; j < noActivations[noLayers - 1]; j++)
         {
            thetas[noLayers][i] += activations[noLayers - 1][j] * weights[noLayers - 1][j][i]; 
         }
         activations[noLayers][i] = activationF(thetas[noLayers][i]);
         outputs[caseIndex][i] = activations[noLayers][i]; // loading activations to output (F) array

         omega = caseOut[caseIndex][i] - outputs[caseIndex][i];
         psi[noLayers][i] = omega * derivActivationF(thetas[noLayers][i]);
      } // for (int i = 0; i < noActivations[noLayers]; i++)

   } // private void runTrain(int caseIndex)

   /*
    * Updates the weights based on updated activation node values and the partial derivative
    * of the error.
    *
    * @param caseIndex  the given training case to update the weights of the network for
    *
    */
   private void updateWeights(int caseIndex)
   {
      double capitalOmega, deltaWeights;
      for (int n = noLayers - 1; n >= MK_INDEX; n--)
      {
         for (int j = 0; j < noActivations[n]; j++)
         {
            capitalOmega = 0.0;

            for (int i = 0; i < noActivations[n + 1]; i++)
            {
               capitalOmega += psi[n + 1][i] * weights[n][j][i];
               deltaWeights = lambda * activations[n][j] * psi[n + 1][i];
               weights[n][j][i] += deltaWeights;
            }

            psi[n][j] = derivActivationF(thetas[n][j]) * capitalOmega;
         } // for (int j = 0; j < noActivations[n]; j++)
      } // for (int n = noLayers - 1; n >= MK_INDEX; n--)

      for (int k = 0; k < noActivations[1]; k++) // loop to calculate weights for first and second layers
      {
         capitalOmega = 0.0;

         for (int j = 0; j < noActivations[MK_INDEX]; j++)
         {
            capitalOmega += psi[MK_INDEX][j] * weights[1][k][j];
            deltaWeights = lambda * activations[1][k] * psi[MK_INDEX][j];
            weights[1][k][j] += deltaWeights;
         }

         psi[1][k] = derivActivationF(thetas[1][k]) * capitalOmega;

         for (int m = 0; m < noActivations[0]; m++)
         {
            deltaWeights = lambda * activations[0][m] * psi[1][k];
            weights[0][m][k] += deltaWeights;
         }
      } // for (int k = 0; k < noActivations[1]; k++)

   } // public void updateWeights(int caseIndex)

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
    * Returns a String with the number of activations in each layer
    */
   private String activationString()
   {
      String printActivation = "";
      for (int n = 0; n < noLayers; n++)
      {
         printActivation += noActivations[n] + "-";
      }
      printActivation += noActivations[noLayers];
      return printActivation;
   } // private String activationString()

   /*
    * Displays the values of a truth table given the outputs
    *
    * @param fOut    the given outputs to display
    */
   private void printTruthTable(double[][] fOut)
   {
      if (showInputs)
      {
         for (int ii = 0; ii < noCases; ii++)
         {
            System.out.print(showInputs(ii));
            System.out.print("| ");
            System.out.println(showOutputs(fOut, ii));
         }
      }
      else
      {
         for (int ii = 0; ii < noCases; ii++)
         {
            System.out.printf("Case #%d: ", ii);
            System.out.println(showOutputs(fOut, ii));
         }
      }
   } // private void printTruthTable(double[][] fOut)

   /*
    * Returns the inputs of a truth table given the case index
    * 
    * @param ii   the given case index
    */
   private String showInputs(int ii)
   {
      String inputs = "";
      for (int k = 0; k < noActivations[0]; k++)
      {
         inputs += caseIn[ii][k] + "  ";
      }

      return inputs;
   }

   /*
    * Returns the outputs of a truth table given the case index and output array values
    * 
    * @param ii      the given case index
    * @param fOut    the given outputs to display
    */
   private String showOutputs(double[][] fOut, int ii)
   {
      String outputs = "";
      for (int i = 0; i < noActivations[noLayers]; i++)
      {
         outputs += fOut[ii][i] + "  ";
      }

      return outputs;
   }

   /*
    * Uses a FileWriter to write number of activations in each layer and weights to a user-given file
    *
    * @param weightsFile     the path of the file to save the weights to
    * @throws IOException    if there is an error during input or output operations
    */
   private void writeWeights(String weightsFile) throws IOException
   {
      FileWriter fw = new FileWriter(weightsFile);

      fw.write(activationString() + "\n");
   
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

      if (!scan.nextLine().equals(activationString()))
      {
         throw new FileNotFoundException("The specified weights file to load from does not match the network configuration.");
      }

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
    * This method returns a file path to the configuration file.
    *
    * @param args    A 1D array of Strings that holds runtime arguments to configure the perceptron.
    */
    private String getConfigFile(String[] args)
    {
       String configFile;
       if (args.length >= CMD_LINE_CONFIG_I + 1)
       {
         configFile = args[CMD_LINE_CONFIG_I];
       }
       else
       {
         configFile = DEFAULT_CONFIG_FILE;
       }
       return configFile;
    } // private String getConfigFile(String[] args)
 
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
