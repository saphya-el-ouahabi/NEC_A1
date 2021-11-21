import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Random;

public class NeuralNet {

  private int L;
  private int[] n;
  private double[][] xi;
  private double[][] sum_xi;
  private double[][][] w;
  private double[][] theta;
  private double[][] delta;
  private double[][] sum_delta;
  private double[][][] d_w;
  private double[][] d_theta;
  private double[][][] d_w_prev;
  private double[][] d_theta_prev;
  private double[][] h;

  //layers = [size(dataTrain-1),9,5,1]
  public NeuralNet(int[] layers) {
    L = layers.length;
    n = layers.clone();
 
    xi = new double[L][]; //activation
    for (int lay = 0; lay < L; lay++) {
    	xi[lay] = new double[n[lay]];
    	for (int i=0;i<n[lay];i++) {
    		xi[lay][i]= 0;
    	}
    }
    
    sum_xi = new double[L][];
    for (int lay = 0; lay < L; lay++) {
    	sum_xi[lay] = new double[n[lay]];
    	for (int i=0;i<n[lay];i++) {
    		sum_xi[lay][i]= 0;
    	}
    }
    
    delta = new double[L][];
    for (int lay = 0; lay < L; lay++) {
      delta[lay] = new double[n[lay]];
    }
    
    sum_delta = new double[L][];
    for (int lay = 0; lay < L; lay++) {
    	sum_delta[lay] = new double[n[lay]];
    }
    
    theta = new double[L][];
    for (int lay = 0; lay < L; lay++) {
    	theta[lay] = new double[n[lay]];
    	for (int i=0;i<n[lay];i++) {
    		theta[lay][i]= 0;
    	}
    }
    
    d_theta = new double[L][];
    for (int lay = 0; lay < L; lay++) {
    	d_theta[lay] = new double[n[lay]];
    	for (int i=0;i<n[lay];i++) {
    		d_theta[lay][i]= 0;
    	}
    }
    
    d_theta_prev = new double[L][];
    for (int lay = 0; lay < L; lay++) {
    	d_theta_prev[lay] = new double[n[lay]];
    	for (int i=0;i<n[lay];i++) {
    		d_theta_prev[lay][i]= 0;
    	}
    }
    
    h = new double[L][];
    for (int lay = 0; lay < L; lay++) {
    	h[lay] = new double[n[lay]];
    }

    w = new double[L][][];
    for (int lay = 1; lay < L; lay++) {
      w[lay] = new double[n[lay]][n[lay - 1]];
    }
    
    d_w_prev = new double[L][][];
    for (int lay = 1; lay < L; lay++) {
    	d_w_prev[lay] = new double[n[lay]][n[lay - 1]];
    	for (int i=0;i<n[lay];i++) {
    		for (int j=0;j<n[lay-1];j++) {
    			d_w_prev[lay][i][j]= 0;
    		}
    	}
    }
    
    d_w = new double[L][][];
    for (int lay = 1; lay < L; lay++) {
    	d_w[lay] = new double[n[lay]][n[lay - 1]];
    	for (int i=0;i<n[lay];i++) {
    		for (int j=0;j<n[lay-1];j++) {
    			d_w[lay][i][j]= 0;
    		}
    	}
    }
  }


  public String toString() {
    String s = "L = " + L + "\n";
    s += "n = [";
    for (int lay = 0; lay < L; lay++) s += " " + n[lay];
    s += "]\n";
    return s;
  }
  
  public void writeFile(double[][] listOutputTest, double[] targetTest, String file) {
	  file=file+"_resultBP.txt";
	  PrintWriter writer;
	try {
		writer = new PrintWriter(file, "UTF-8");
		writer.println("Real"+"\t"+"Predict"+"\n");
		for (int i = 0; i< targetTest.length;i++) {
			writer.println(targetTest[i]+"\t"+listOutputTest[i][0]);
		}
		writer.close();
	} catch (FileNotFoundException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	} catch (UnsupportedEncodingException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
  }
  
  public void randomizeWeights() {
	  for (int l = 1; l < this.w.length; l++) {
	      for (int i = 0; i < this.w[l].length; i++) {
	    	  for (int j = 0; j < this.w[l][i].length; j++) {
	    		  Random r = new Random();
		          this.w[l][i][j] = r.nextDouble();
	    	  }
	      }
	  }
  }
  
  public void randomizeThresholds() {
	  for (int l = 0; l < this.n.length; l++)
	      for (int i = 0; i < this.n[l]; i++) {
	    	  Random r = new Random();
	    	  this.theta[l][i] = r.nextDouble()*2-1;
	      }
  }

  public double[] feedForward(double[] pattern) {
	  xi[0]=pattern;
	  for (int lay = 1; lay<L;lay++) {
		  for (int i=0; i< n[lay]; i++) {
			  double field = -theta[lay][i];
			  for (int j = 0; j<n[lay-1] ; j++) {
				  field+= w[lay][i][j]*xi[lay-1][j];
			  }
			  h[lay][i]=field;
			  
			  xi[lay][i]=sigmoid(field);
			  sum_xi[lay][i]+=xi[lay][i];
		  }
	  }
	  return xi[L-1];
  }
  
  public void updateWeights() {
	  double nbizarre = 0.05; //element à faire varier
	  double alpha = 0.1; //element à faire varier
	  for (int lay=1; lay<L-1 ;lay++) {
		  for (int i=0;i<n[lay];i++) {
			  for (int j=0;j<n[lay-1];j++) {
				  d_w[lay][i][j] = -nbizarre*sum_delta[lay][i]*sum_xi[lay-1][j]+alpha*d_w_prev[lay][i][j] ;
				  d_w_prev[lay][i][j]=d_w[lay][i][j];
				  w[lay][i][j]= w[lay][i][j]+d_w[lay][i][j];
			  }
		  }
	  }
  }
  
  public void updateThresholds() {
	  double nbizarre = 0.02; //element à faire varier
	  double alpha = 0.2; //element à faire varier
	  for (int lay=1; lay<L-1 ;lay++) {
		  for (int i=0;i<n[lay];i++) {
			  d_theta[lay][i] = nbizarre*sum_delta[lay][i]+alpha*d_theta_prev[lay][i] ;
			  d_theta_prev[lay][i]=d_theta[lay][i];
			  theta[lay][i]= theta[lay][i]+d_theta[lay][i];
		  }
		  
	  }
  }
  
  public double sigmoid(double h) {
	  return (1/(1+ Math.exp(-h)));
  }
  

  public void bpError(double[] output, double[] targetTrain) {
	  for (int i=0; i<n[L-1]; i++) {
		  double derivSig = sigmoid(h[L-1][i])*(1-sigmoid(h[L-1][i]));
		  delta[L-1][i]=derivSig*(output[i]-targetTrain[i]);
		  sum_delta[L-1][i]+=delta[L-1][i];
	  }
	  
	  for (int lay=L-1; lay>0;lay--) {
		  for (int j=0;j<n[lay-1];j++) {
			  double error=0;
			  for (int i=0;i<n[L-1]; i++) {
				  error+=delta[lay][i]*w[lay][i][j];
			  }
			  double derivSig2 = sigmoid(h[lay-1][j])*(1-sigmoid(h[lay-1][j]));
			  delta[lay-1][j]=derivSig2*error;
			  sum_delta[lay-1][j]+=delta[lay-1][j];
		  }
	  }
  }

  
  public double mape(double[][] pattern, double[] target) {
	  double result = 0;
	  double sum = 0;
	  for (int i=0; i < pattern.length; i++) {
		  for (int j=0; j<pattern[i].length ;j++) {
			  result+= Math.abs(pattern[i][j]-target[i]);
			  sum+=pattern[i][j];
		  }
	  }
	  return (result/sum)*100;
  }
  
  public void backPropagation(String file, double[][] dataTrain, double[][] dataTest, double[] targetTrain, double[] targetTest) {
	  double[] pattern = new double[0];
	  this.randomizeWeights();
	  this.randomizeThresholds();
	  int epoch = 10;
	  for (int num_epoch = 1; num_epoch < epoch; num_epoch++) {
		  for (int num_pat = 0; num_pat < 5; num_pat++) {
			  int num = (int) (Math.random()*dataTrain.length);
			  pattern=dataTrain[num];
			  double[] output = feedForward(pattern);
			  bpError(output, targetTrain);
			  updateWeights();
			  updateThresholds();
		  }
		  double[][] listOutput = new double[dataTrain.length][n[0]];
		  
		  for (int num_pat = 0; num_pat < dataTrain.length; num_pat++) {
			  pattern=dataTrain[num_pat];
			  listOutput[num_pat] = feedForward(pattern);  
		  }
		  double quadr_Error_Train = mape(listOutput, targetTrain);
		  
		  double[][] listOutputTest = new double[dataTest.length][1];
		  for (int num_pat = 0; num_pat < dataTest.length; num_pat++) {
			  pattern=dataTest[num_pat];
			  listOutputTest[num_pat] = feedForward(pattern);
		  }
		  double quadr_Error_Test = mape(listOutputTest, targetTest);
		  writeFile(listOutputTest, targetTest, file);
		  System.out.println("mape_Train : "+quadr_Error_Train);
		  System.out.println("mape_Test : "+quadr_Error_Test);
	  }
  }

  /*
  # Optional: Plot the evolution of the training and validation errors
  Feed−forward all test patterns
  Descale the predictions of test patterns, and evaluate them
  */
  
  public static void main(String[] args) {
	
	System.out.println("File : A1-turbine"+"\n");
	Dataset dataset = new Dataset("A1-turbine-norm.txt", 451, 5, 0.85);
    int[] layers = {4, 9, 5, 1};
    NeuralNet nn = new NeuralNet(layers);
    nn.backPropagation("A1-turbine",dataset.getTrainData(), dataset.getTestData(), dataset.getTrainTarget(), dataset.getTestTarget());
    
    System.out.println("\n"+"File : A1-synthetic"+"\n");
	Dataset dataset2 = new Dataset("A1-synthetic_normalized.txt", 1000, 10, 0.80);
    int[] layers2 = {9, 9, 5, 1};
    NeuralNet nn2 = new NeuralNet(layers2);
    nn2.backPropagation("A1-synthetic",dataset2.getTrainData(), dataset2.getTestData(), dataset2.getTrainTarget(), dataset2.getTestTarget());
  	
    System.out.println("\n"+"File : insurance_dataset"+"\n");
	Dataset dataset3 = new Dataset("insurance_dataset_norm.txt", 1193, 7, 0.80);
    int[] layers3 = {6, 9, 5, 1};
    NeuralNet nn3 = new NeuralNet(layers3);
    nn3.backPropagation("insurance_dataset",dataset3.getTrainData(), dataset3.getTestData(), dataset3.getTrainTarget(), dataset3.getTestTarget());
  	
  }

}
