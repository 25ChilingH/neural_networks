import java.io.*;
import java.util.*;

/*
 * November 1, 2023
 * 
 * Chiling Han
 * 
 * The main class for running and training by backpropagation the n-layer perceptron network.
 * It loads in a network configuration file that stores the settings of the network
 * and also the various training hyperparameters. Then, it prints out relevant
 * information to running or training such as the calculated truth table.
 * 
 * Example usage:
 * java Main
 */
public class Main
{
   /*
    * The main method interfaces between the user and the perceptron. Prints the network configuration and
    * training hyperparameters before running/training. At the end of running, it will output the truth table.
    * At the end of training, it will output training exit information such as the number of training
    * iterations and the average error.
    * 
    *
    * @param args            A 1D array of Strings that holds runtime arguments to configure the perceptron.
    * @throws IOException    if there is an error during input or output operations
    */
   public static void main(String[] args) throws IOException
   {
      Perceptron neuralNet = new Perceptron();

      neuralNet.setConfigParams(args);
      switch (neuralNet.trainOrRun)
      {
         case "train":
            neuralNet.allocateTrainingArrays();
            neuralNet.populateArrays();
            neuralNet.echoTrainingConfig();
            neuralNet.trainAndReport();
            break;
         case "run":
            neuralNet.allocateRunningArrays();
            neuralNet.populateArrays();
            neuralNet.echoRunConfig();
            neuralNet.runAndReport();
            break;
      } // switch (trainOrRun)

   } // public static void main(String args[]) throws IOException
   
} // public class Main
