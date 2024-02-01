import java.io.*;
import java.util.*;

/*
 * September 10, 2023
 * 
 * Chiling Han
 * 
 * The class serves as the architecture for a A-B-1 perceptron. It will configure the network,
 * including the number of layers, the number of activation units, the initial weights of
 * the perceptron, and other training hyperparameters.
 * 
 * Furthermore, this class can train and run a perceptron by calculating the values of final outputs
 * given any number of inputs and also comparing them to the corresponding user-given expected values.
 * 
 * int noLayers               - Number of layers in the network
 * String activationString    - Represents the number of activations in each layer as a String
 * String caseFile            - The truth table of the case being tested
 * int noCases                - The number of train/test cases
 * int maxIterations          - Maximum number of iterations allowed
 * double errorThreshold      - The minimum error to reach before ending the training cycle
 * double lambda              - The learning factor used to control how much the weights are modified for each step
 * String weightsFile         - A file containing the weights to use for adjusting activation values
 * double minWeights          - The lower end of the range of weights used
 * double maxWeights          - The upper end of the range of weights used
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
   public String activationString;

   public String caseFile;
   public int noCases;

   public int maxIterations;
   public double errorThreshold;
   public double lambda;

   public String weightsFile;
   public double minWeights;
   public double maxWeights;

   private double avgError;
   private int epoch;

   private int[] noActivations;
   private double[][] caseIn;
   private double[][] caseOut;
   private double[][][] weights, deltaWeights;
   private double[][] activations, thetas;
   private double[][] outputs;
   private double[] psi;

   /*
    * Initializes the network by allocating memory to the arrays necessary for running the network.
    * This method also loads in values from files supplied by the user.
    *
    * @throws FileNotFoundException    if a provided file is invalid or no file is provided
    */
   public void initNetwork() throws FileNotFoundException
   {
      loadNoActivations();
      loadActivationNetwork();
      loadThetas();
      loadCases();
      outputs = new double[noCases][noActivations[noLayers]];
      loadWeights();
   }

   /*
    * Initializes the network by calling a method to allocate memory for arrays to run the network
    * and allocates memory to training-specific arrays.
    *
    */
   public void initTrain() throws FileNotFoundException
   {
      initNetwork();
      loadDeltaWeights();
      psi = new double[noActivations[noLayers]];
   }

   /*
    * Displays the network configuration including number of activations and the truth table corresponding to the
    * selected network (XOR, OR, AND).
    */
   public void printNetworkConfig()
   {
      System.out.println("NETWORK CONFIGURATION\n-------------------------");

      System.out.println("Activations:");
      printActivations();

      System.out.println("\nTRUTH TABLE (T)");
      printTruthTable(caseOut);
   }

   /*
    * Displays the network configuration by calling the printNetworkConfig() method
    * and prints parameters used for training the network.
    */
   public void printTrainingConfig()
   {
      printNetworkConfig();
      System.out.println("\nTRAINING PARAMETERS\n-------------------------");
      System.out.println("Weights range: " + minWeights + " to " + maxWeights);
      System.out.println("Maximum iterations: " + maxIterations);
      System.out.println("Error threshold: " + errorThreshold);
      System.out.println("Lambda value: " + lambda);
   }

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
    * Trains the network and displays the results.
    *
    * @throws FileNotFoundException    if a provided file is invalid or no file is provided
    */
   public void trainAndReport() throws FileNotFoundException
   {
      trainNetwork();
      runAndReport();

      System.out.println("\nTRAINING EXIT INFO\n-------------------------");
      System.out.print("Training stopped because: ");

      if (avgError <= errorThreshold)
         System.out.println("average error is below the error threshold.");
      else if (epoch >= maxIterations)
         System.out.println("maximum iterations exceeded");

      System.out.println("Mean Error: " + avgError);
      System.out.println("Iterations reached: " + epoch);
   } // public void trainAndReport() throws FileNotFoundException

   /*
    * Trains the network using steepest descent to minimize the error value and near the target values.
    * 
    * Training ends if the maximum number of iterations is reached or the calculated error
    * is below the error threshold.
    *
    * @throws FileNotFoundException    if a provided file is invalid or no file is provided
    */
   public void trainNetwork() throws FileNotFoundException
   {
      initTrain();

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
   } // public void trainNetwork() throws FileNotFoundException

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
      run(caseIndex);

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

      outputs[caseIndex][0] = activations[noLayers][0];

   } // private void run(int caseIndex)

   /*
    * Updates the change in weights based on updated activation node values and the partial derivative
    * of the error.
    *
    * @param caseIndex  the given training case to update the weights of the network for
    *
    */
   private void updateDeltaWeights(int caseIndex)
   {
      double omega = caseOut[caseIndex][0] - outputs[caseIndex][0];
      psi[0] = omega * derivActivationF(thetas[noLayers][0]);

      for (int j = 0; j < noActivations[noLayers - 1]; j++)
      {
         deltaWeights[noLayers - 1][j][0] = psi[0] * -activations[noLayers - 1][j] * -lambda;
      }

      for (int n = 1; n < noLayers; n++)
      {
         for (int j = 0; j < noActivations[n]; j++)
         {
            double capitalOmega = 0.0;

            capitalOmega += psi[0] * weights[n][j][0];

            for (int k = 0; k < noActivations[n - 1]; k++)
            {
               double capitalPsi = derivActivationF(thetas[n][j]) * capitalOmega;
               deltaWeights[n - 1][k][j] = capitalPsi * -activations[n - 1][k] * -lambda;
            }
         }
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

      return totalError / noCases;
   } // public double meanError()

   /*
    * Computes the error for one given case using the formula 1/2(F - T)^2
    *
    * @param caseIndex  the given training case to minimize the error function for
    */
   private double errorFunction(int caseIndex)
   {
      double error = 0.0;

      error += (caseOut[caseIndex][0] - outputs[caseIndex][0]) * (caseOut[caseIndex][0] - outputs[caseIndex][0]);
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

         System.out.print(fOut[ii][0]);
         System.out.println();
      } // for (int ii = 0; ii < noCases; ii++)
   } // private void printTruthTable(double[][] fOut)

   /*
    * Allocates memory for the thetas array, which will contain values of the
    * weighted sums for each activation unit
    */
   private void loadThetas()
   {
      thetas = new double[noLayers + 1][];

      for (int n = 0; n <= noLayers; n++)
      {
         thetas[n] = new double[noActivations[n]];
      }
   }

   /*
    * Allocates memory for the deltaWeights array, which will contain values of the
    * change between the previous and current weights for each layer.
    */
   private void loadDeltaWeights()
   {
      deltaWeights = new double[noLayers][][];

      for (int n = 0; n < noLayers; n++)
      {
         deltaWeights[n] = new double[noActivations[n]][noActivations[n + 1]];
      }
   }

   /*
    * Fills the weights array with values of weights loaded from a file.
    */
   private void loadWeightsFromFile() throws FileNotFoundException
   {
      Scanner scan = new Scanner(new File(weightsFile));

      for (int n = 0; n < noLayers; n++)
      {
         weights[n] = new double[noActivations[n]][noActivations[n + 1]];

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
         weights[n] = new double[noActivations[n]][noActivations[n + 1]];

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
    * Fills the weights array with values of weights either loaded from a file or randomized between a given 
    * range of weight values.
    */
   private void loadWeights() throws FileNotFoundException
   {
      weights = new double[noLayers][][];

      boolean load = weightsFile.length() >= 1;
      
      if (load)
         loadWeightsFromFile();
      else
         randomizeWeights();

   } // private void loadWeights()

   /*
    * Loads in the truth table values for all train/test cases
    *
    * @throws FileNotFoundException    if a provided file is invalid or no file is provided
    */
   private void loadCases() throws FileNotFoundException
   {
      caseIn = new double[noCases][noActivations[0]];
      caseOut = new double[noCases][noActivations[noLayers]];

      Scanner scan = new Scanner(new File(caseFile));
      String[] caseInOut;

      for (int ii = 0; ii < noCases; ii++)
      {
         caseInOut = scan.nextLine().split(" ");

         int k;
         for (k = 0; k < noActivations[0]; k++)
         {
            caseIn[ii][k] = Double.valueOf(caseInOut[k]);
         }
         
         caseOut[ii][0] = Double.valueOf(caseInOut[k]);

      } // for (int ii = 0; ii < noCases; ii++)

   } // private void loadCases() throws FileNotFoundException

   /*
    * Allocates memory for the activation units of the network
    */
   private void loadActivationNetwork()
   {
      activations = new double[noLayers + 1][];

      for (int n = 0; n <= noLayers; n++)
      {
         activations[n] = new double[noActivations[n]];
      }
   }

   /*
    * Loads in the number of activations in each layer of the network
    */
   private void loadNoActivations()
   {
      noActivations = new int[noLayers + 1];
      String[] activationCount = activationString.split(" ");

      for (int n = 0; n <= noLayers; n++)
      {
         noActivations[n] = Integer.valueOf(activationCount[n]);
      }
   }

} // public class Perceptron
