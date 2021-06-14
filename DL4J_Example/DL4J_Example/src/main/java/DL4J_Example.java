import org.jlab.groot.data.GraphErrors;
import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;

public class DL4J_Example {
	public static void main(String[] args) {
		//location of the positive and negative sample hipo files
		
		//For use in JLab
		String hipoLocPos="/volatile/clas12/osg2/tyson/job_3143/output/";
		String hipoLocNeg="/volatile/clas12/osg2/tyson/job_3144/output/";
		Boolean JLab=true;
		
		//For use in Glasgow:
		/*String hipoLocPos="/w/work5/jlab/hallb/clas12/rg-b/simulations/en/sim_en_";
		String hipoLocNeg="/w/work5/jlab/hallb/clas12/rg-b/simulations/eg/sim_eg_";
		Boolean JLab=false;*/
		
		//number of testing and training events
		int NTrain=100000;
		int NTest=100000;
		
		//Create the training and testing datasets
		DataSet TrainData = CreateDataset(hipoLocPos,hipoLocNeg,NTrain,0,JLab);
		DataSet TestData = CreateDataset(hipoLocPos,hipoLocNeg,NTest,400,JLab);
		long inputSize=TrainData.getFeatures().shape()[1];
		long outputSize=TrainData.getLabels().shape()[1];
		
		System.out.println("input "+inputSize+" output "+outputSize);
		
		//Create the network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(42)
	            .updater(new Adam())
	            .l2(1e-4)
	            .list()
	            .layer(new DenseLayer.Builder()
	                .nIn(inputSize) // Number of input datapoints.
	                .nOut(50) // Number of Nodes in Second Layer
	                .activation(Activation.RELU) // Activation function.
	                .weightInit(WeightInit.XAVIER) // Weight initialization.
	                .build())
	            .layer(new DenseLayer.Builder()
		                .nIn(50) // Number of nodes in second layer
		                .nOut(25) // Number of output datapoints.
		                .activation(Activation.RELU) // Activation function.
		                .weightInit(WeightInit.XAVIER) // Weight initialization.
		                .build())
	            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                .nIn(25)
	                .nOut(outputSize)
	                .activation(Activation.SOFTMAX)
	                .weightInit(WeightInit.XAVIER)
	                .build())
	            .build();
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.init();

		// pass a training listener that reports score every iteration
		int eachIterations = 1;
		network.addListeners(new ScoreIterationListener(eachIterations));

		int nbEpochs=10;
		
		GraphErrors gTrainAcc= new GraphErrors();
		GraphErrors gTestAcc= new GraphErrors();
		//train for multiple epochs
		for(int epoch=0;epoch<nbEpochs;epoch++) {
			network.fit(TrainData); //training
			
			//log the accuracy at each epoch to check for overfitting
			INDArray outputTrain=network.output(TrainData.getFeatures());
			Evaluation evalTrain = new Evaluation(2);
			evalTrain.eval(TrainData.getLabels(), outputTrain);
			gTrainAcc.addPoint(epoch, evalTrain.accuracy(), 0, 0);
			
			INDArray outputTest=network.output(TestData.getFeatures());
			Evaluation evalTest = new Evaluation(2);
			evalTest.eval(TestData.getLabels(), outputTest);
			gTestAcc.addPoint(epoch, evalTest.accuracy(), 0, 0);
		}
		PlotOverfitting(gTrainAcc,gTestAcc);
		
		//test the network and make some plots to evaluate its performance
		INDArray output=network.output(TestData.getFeatures());
		Evaluation eval = new Evaluation(2);
		eval.eval(TestData.getLabels(), output);
		
		System.out.println(eval.stats());
		
		PlotResponse(NTest,output,TestData.getLabels());
		double bestRespTh=PlotMetricsVsResponse(NTest,output,TestData.getLabels());
		
	}
	
	/*
	 * This function reads the hipo files in the negative and positive sample location,
	 * then creates a dataset from these which is returned for training or testing
	 * 
	 * Arguments:
	 * 			hipoLocPos: The location of the positive sample hipo files.
	 *          hipoLocNeg: The location of the negative sample hipo files.
	 * 			NPreds: Number of desired entries in the array.
	 * 			filestart: offset in the file numbers to be read.
	 * 			JLab: boolean used to check if this example is being run at JLab as there's some difference
	 * in the directory structure.
	 * 
	 * Returns:
	 * 			INDArray list with the features array at index 0 and the labels at index 1.
	 */
	public static DataSet CreateDataset(String hipoLocPos, String hipoLocNeg, int NPreds,int filestart, Boolean JLab) {
		INDArray[] posData=ReadFromHipo(hipoLocPos,NPreds,2112,filestart,JLab);
		INDArray[] negData=ReadFromHipo(hipoLocNeg,NPreds,22,filestart,JLab);
		INDArray Vars = Nd4j.vstack(posData[0],negData[0]);
		INDArray Labels=Nd4j.vstack(posData[1],negData[1]);
		DataSet dataset = new DataSet(Vars,Labels);
		return dataset;
	}
	
	/*
	 * Reads as many hipo files as necessary from the location specified in hipoLoc to fill
	 * a dataset of size NPreds. This can be the positive or negative sample dataset depending
	 * on partPID (with 2112 specifying the positive sample and 22 the negative sample).
	 * An offset in the starting file number is provided with filestart.
	 * This method then reads the specified hipo files, and fills a feature array with
	 * the energy deposition and DU/DV/DW variables for each calorimeter subsystem for
	 * a particle of pid partPID. Other requirements are that this particle is in the FD
	 * and that there is a hit in at least one of the calorimeter subsystems.
	 * A labels array is also filled at the same time, and both the features and labels 
	 * are returned as a list.
	 * 
	 * Arguments:
	 * 			hipoLoc: The location of the hipo files.
	 * 			NPreds: Number of desired entries in the array.
	 * 			partPID: the PID of the particle
	 * 			filestart: offset in the file numbers to be read.
	 * 			JLab: boolean used to check if this example is being run at JLab as there's some difference
	 * in the directory structure.
	 * 
	 * Returns:
	 * 			INDArray list with the features array at index 0 and the labels at index 1.
	 */
	public static INDArray[] ReadFromHipo(String hipoLoc, int NPreds,int partPID,int filestart, Boolean JLab) {
		INDArray Vars=Nd4j.zeros(NPreds/2,12); //fastest way to initialise INDArray
		INDArray Labels=Nd4j.zeros(NPreds/2,2);
		int NPred=0;
		int fNumber=0+filestart;
		while(NPred<NPreds/2) {
			String fName=hipoLoc+fNumber+".hipo";
			if(JLab) {
				fName=hipoLoc+"simu_"+fNumber+"/dst.hipo";
			}
			HipoReader reader = new HipoReader();
			reader.open(fName);
			Event event = new Event();
			Bank ec = new Bank(reader.getSchemaFactory().getSchema("REC::Calorimeter"));
			Bank parts = new Bank(reader.getSchemaFactory().getSchema("REC::Particle"));
			while (reader.hasNext() == true && NPred<(NPreds/2)) {
				reader.nextEvent(event);
				event.read(ec);
				event.read(parts);
				//ec.show();
				
				//Need to get particle index from REC::Particle bank
				int neutralIndex=-1;
				for (int i = 0; i < parts.getRows(); i++) {
					int pid = parts.getInt("pid", i);
					int status = parts.getInt("status", i);
					if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
						if (pid == partPID) {
							neutralIndex=i;
						}
					}
				}
				float[] ecVars=new float[12];
				for(int k=0;k<12;k++) {
					ecVars[k]=0;
				}
				int CalMtp=0;
				for (int i = 0; i < ec.getRows(); i++) {
					int pindex = ec.getInt("pindex", i);
					if(pindex==neutralIndex) {
						int layer=ec.getInt("layer", i);
						if(layer==1) {
							ecVars[0]=ec.getFloat("energy", i);
							ecVars[1]=ec.getFloat("du", i);
							ecVars[2]=ec.getFloat("dv", i);
							ecVars[3]=ec.getFloat("dw", i);
							CalMtp++;
						} else if(layer==4) {
							ecVars[4]=ec.getFloat("energy", i);
							ecVars[5]=ec.getFloat("du", i);
							ecVars[6]=ec.getFloat("dv", i);
							ecVars[7]=ec.getFloat("dw", i);
							CalMtp++;
						} else if(layer==7) {
							ecVars[8]=ec.getFloat("energy", i);
							ecVars[9]=ec.getFloat("du", i);
							ecVars[10]=ec.getFloat("dv", i);
							ecVars[11]=ec.getFloat("dw", i);
							CalMtp++;
						} 
					}
				}//read ec bank
				if(CalMtp!=0) {
					for(int k=0;k<12;k++) {
						Vars.putScalar(new int[] {NPred,k}, ecVars[k]);
					}
					if(partPID==2112) {
						Labels.putScalar(new int[] {NPred,0}, 1);
						Labels.putScalar(new int[] {NPred,1}, 0);
					} else if (partPID==22) {
						Labels.putScalar(new int[] {NPred,0}, 0);
						Labels.putScalar(new int[] {NPred,1}, 1);
					}
					NPred++;
				}//fill output arrays
			}//while reading file
			fNumber++;
		}//while loop over files
		INDArray[] out=new INDArray[2];
		out[0]=Vars;
		out[1]=Labels;
		return out;
	
	}//end ReadFromHipo
	
	/*
	 * Calculates the Accuracy, Efficiency and Purity of the neural network as a function for a given
	 * threshold on the response.
	 * 
	 * Arguments:
	 * 			NEvents: The length of the predictions array.
	 * 			predictions: Contains the output of the neural network
	 * 			Labels: Contains the true classes for each prediction
	 * 			RespTh: the threshold applied to the classifier response
	 * 
	 * Returns:
	 * 			INDArray which is 3*1. The first row contains the accuracy, the second the purity,
	 * the third the efficiency of the neural network
	 */
	public static INDArray getMetrics(int NEvents,INDArray predictions, INDArray Labels, double RespTh) {
		INDArray metrics = Nd4j.zeros(3,1);
		double TP=0,FN=0,FP=0,TN=0;
		for(int i=0;i<NEvents;i+=1) {
			if(predictions.getFloat(i,0)!=-1) {
				if(Labels.getFloat(i,0)==1) {
					if(predictions.getFloat(i,0)>RespTh) {
						TP++;
					} else {
						FN++;
					}//Check model prediction
				} else if(Labels.getFloat(i,0)==0) {
					if(predictions.getFloat(i,0)>RespTh) {
						FP++;
					} else {
						TN++;
					}//Check model prediction
				}//Check true label
			}//Check that prediction not equals -1, used when binning in P to ignore values
		}//loop over events
		double Acc=(TP+TN)/(TP+TN+FP+FN);
		double Pur=TP/(TP+FP);
		double Eff=TP/(TP+FN);
		metrics.putScalar(new int[] {0,0}, Acc);
		metrics.putScalar(new int[] {1,0}, Pur);
		metrics.putScalar(new int[] {2,0}, Eff);
		return metrics;
	}//End of getMetrics
	
	/*
	 * Plots the Accuracy, Efficiency and Purity of the neural network as a function of the lower
	 * threshold on the response. Returns the threshold at which the Purity is maximised whilst keeping
	 * the Efficiency above 0.995.
	 * 
	 * Arguments:
	 * 			NEvents: The length of the predictions array.
	 * 			predictions: Contains the output of the neural network
	 * 			Labels: Contains the true classes for each prediction
	 * 
	 * Returns:
	 * 			Returns the threshold on the response at which the Accuracy is maximised.
	 */
	public static double PlotMetricsVsResponse(int NEvents,INDArray predictions, INDArray Labels) {
		GraphErrors gAcc= new GraphErrors();
		GraphErrors gEff= new GraphErrors();
		GraphErrors gPur= new GraphErrors();
		double bestRespTh=0;
		double bestAcc=0;
		
		//Loop over threshold on the response
		for(double RespTh=0.01; RespTh<0.99;RespTh+=0.01) {
			INDArray metrics =getMetrics(NEvents, predictions, Labels, RespTh);
			double Acc=metrics.getFloat(0,0);
			double Pur=metrics.getFloat(1,0);
			double Eff=metrics.getFloat(2,0);
			gAcc.addPoint(RespTh, Acc, 0, 0);
			gPur.addPoint(RespTh, Pur, 0, 0);
			gEff.addPoint(RespTh, Eff, 0, 0);
			if (Acc>bestAcc) {
				bestAcc=Acc;
				bestRespTh=RespTh;
			}
			
		}//Increment threshold on response
		
		System.out.format("%n Best Accuracy : %.3f at a threshold on the response of %.3f %n%n",bestAcc,bestRespTh);
		
		TCanvas canvasMR = new TCanvas("Metrics vs Response",800,500);
		canvasMR.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
		canvasMR.getCanvas().getPad(0).setLegend(true);
		canvasMR.getCanvas().getPad(0).setLegendPosition(400, 300);
		canvasMR.setLocationRelativeTo(null);
		canvasMR.setVisible(true);
		
		gAcc.setTitle("Accuracy (Green)");
		gAcc.setTitleX("Classifier Response");
		gAcc.setTitleY("Metrics");
		gAcc.setMarkerColor(3);
		gAcc.setMarkerStyle(8);
		canvasMR.draw(gAcc,"AP");
		
		gPur.setTitle("Purity (Red)");
		gPur.setTitleX("Classifier Response");
		gPur.setTitleY("Metrics");
		gPur.setMarkerColor(2);
		gPur.setMarkerStyle(8);
		canvasMR.draw(gPur,"sameAP");
		
		gEff.setTitle("Efficiency (Blue)");
		gEff.setTitleX("Classifier Response");
		gEff.setTitleY("Metrics");
		gEff.setMarkerColor(4);
		gEff.setMarkerStyle(8);
		canvasMR.draw(gEff,"sameAP");
		canvasMR.getCanvas().getPad(0).setTitle(canvasMR.getTitle());
		
		return bestRespTh;
	}//End of PlotMetricsVSResponse
	
	
	/*
	 * Plots the Accuracy evaluated on the training and testing datasets at each training epoch.
	 * A clear indication of overfitting is when the accuracy decreases for the training dataset
	 * but increases for the testing dataset.
	 * 
	 * Arguments:
	 * 			gTrainAcc: Graph containing the accuracy at each epoch evaluated on the training dataset
	 * 			gTestAcc: Graph containing the accuracy at each epoch evaluated on the testing dataset
	 */
	public static void PlotOverfitting(GraphErrors gTrainAcc,GraphErrors gTestAcc) {
		
		TCanvas canvasO= new TCanvas("Accuracy vs Epoch",800,500);
		canvasO.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
		canvasO.getCanvas().getPad(0).setLegend(true);
		canvasO.getCanvas().getPad(0).setLegendPosition(400, 300);
		canvasO.setLocationRelativeTo(null);
		canvasO.setVisible(true);
		
		gTrainAcc.setTitle("Training Accuracy (Red)");
		gTrainAcc.setTitleX("Classifier Response");
		gTrainAcc.setTitleY("Metrics");
		gTrainAcc.setMarkerColor(2);
		gTrainAcc.setMarkerStyle(8);
		canvasO.draw(gTrainAcc,"AP");
		
		gTestAcc.setTitle("Testing Accuracy (Blue)");
		gTestAcc.setTitleX("Classifier Response");
		gTestAcc.setTitleY("Metrics");
		gTestAcc.setMarkerColor(4);
		gTestAcc.setMarkerStyle(8);
		canvasO.draw(gTestAcc,"sameAP");
		canvasO.getCanvas().getPad(0).setTitle(canvasO.getTitle());
		

	}//End of PlotOverfitting
	
	/*
	 * Plots the classifier response given the output of the classifier and the true classes.
	 * 
	 * Arguments:
	 * 			NEvents: The length of the output array.
	 * 			output: Contains the the neural network predictions.
	 * 			Labels: Contains the true classes for each prediction.
	 */
	public static void PlotResponse(int NEvents,INDArray output, INDArray Labels) {
		H1F hRespPos = new H1F("Response_Positive_Sample", 100, 0, 1);
		H1F hRespNeg = new H1F("Response_Negative_Sample", 100, 0, 1);
		//Sort predictions into those made on the positive/or negative samples
		for(int i=0;i<NEvents;i+=1) {
			if(Labels.getFloat(i,0)==1) {
				hRespPos.fill(output.getFloat(i,0));
			} else if(Labels.getFloat(i,0)==0) {
				hRespNeg.fill(output.getFloat(i,0));
			}
		}
		
		TCanvas canvasResp = new TCanvas("Classifier Response",800,500);
		canvasResp.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
		canvasResp.getCanvas().getPad(0).getAxisY().setLog(true);
		canvasResp.getCanvas().getPad(0).setLegend(true);
		canvasResp.getCanvas().getPad(0).setLegendPosition(100, 20);
		canvasResp.setLocationRelativeTo(null);
		canvasResp.setVisible(true);
		
		hRespPos.setTitle("Positive Sample Response (Blue)");
		hRespPos.setTitleX("Classifier Response");
		hRespPos.setTitleY("Counts");
		hRespPos.setLineWidth(2);
		hRespPos.setLineColor(4);
		hRespPos.setFillColor(4);
		canvasResp.draw(hRespPos);
		
		hRespNeg.setTitle("Negative Sample Response (Red)");
		hRespNeg.setTitleX("Classifier Response");
		hRespNeg.setTitleY("Counts");
		hRespNeg.setLineWidth(3);
		hRespNeg.setLineColor(2);
		canvasResp.draw(hRespNeg,"same");
		canvasResp.getCanvas().getPad(0).setTitle(canvasResp.getTitle());
		
		
	}//End of PlotResponse

}//end class
