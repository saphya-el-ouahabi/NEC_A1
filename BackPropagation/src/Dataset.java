import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Dataset {
	double[][] trainData;
	double[][] testData;
	double[] trainTarget;
	double[] testTarget;
	int nbColumns;
	
	public Dataset(String fileName, int nbLines, int nbColumns, double percentage) {
		this.nbColumns=nbColumns;
		int nb = (int) (0.8*nbLines);
		this.trainData = new double[nb][nbColumns-1];
		this.testData = new double[nbLines-nb][nbColumns-1];
		this.trainTarget = new double[nb];
		this.testTarget = new double[nbLines-nb];
		
        // -File class needed to turn stringName to actual file
        File file = new File(fileName);
        try{
            // -read from filePooped with Scanner class
            Scanner inputStream = new Scanner(file);
            
            //Remove the name of the colums
            for (int i=0; i<nbColumns;i++) {
            	inputStream.next();            	
            }
            
            // hashNext() loops line-by-line
            double[] lineTrain = new double[nbColumns-1];
            double[] lineTest = new double[nbColumns-1];
                  
            int cpt = 0;
            int cpt2 = 0;
            int cpt3 = 0;
            while(inputStream.hasNext()){    	
                //read single line, put in string
            	double data = Double.parseDouble(inputStream.next());
            	//DATATRAIN
            	if (cpt2<nb) {
            		
            		// Remove the target Column
                	if (cpt==nbColumns-1) {
                		trainTarget[cpt2]=data;
                    	cpt=0;
                    	trainData[cpt2]=lineTrain;
            			cpt2++;
            			lineTrain= new double[nbColumns-1];
			
            		// Retrieve the other columns
                    }else {
    	            	lineTrain[cpt]=data;
    	            	cpt++;
                    }
            	}
            	//DATATEST
            	else {
            		
            		// Remove the target Column
                	if (cpt==nbColumns-1) {
                		testTarget[cpt3]=data;
                    	cpt=0;
                    	testData[cpt3]=lineTest;
                    	cpt3++;
            			cpt2++;
            			lineTest= new double[nbColumns-1];
            			
            		// Retrieve the other columns
                    }else {
    	            	lineTest[cpt]=data;
    	            	cpt++;
    	            	
    	            	
                    }
            	}
            }
            
            // after loop, close scanner
            inputStream.close();
        }catch (FileNotFoundException e){
            e.printStackTrace();
        }
	}

	public double[][] getTrainData() {
		return trainData;
	}


	public double[][] getTestData() {
		return testData;
	}


	public double[] getTrainTarget() {
		return trainTarget;
	}


	public double[] getTestTarget() {
		return testTarget;
	}
	
	
}
