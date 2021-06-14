# DeepLearning4J Example

In this example you will learn how to use the deeplearning4j library in Java with the CLAS12 Hipo file reading tools (see [Hipo Library](https://github.com/gavalian/hipo) ). 

This example is based on a PID classification task where the aim is to distinguish between simulated photons and neutrons detected in the CLAS12 Forward Detector based on their energy deposition in the calorimeters, and the shower width in U/V/W. 

NB: Even if the accuracy returned by the algorithm is over 90% this example is by no means complete and is not ready to be deployed in an analysis.

In principle this directory is an eclipse-workspace and you should be able to load this as is (although unfortunately I'm no expert in this).

The DL4J_Example class contains several methods for plotting relevant metrics, and reading in data from .hipo files. The main thing to note here is that you need to pass in the location of the data files, at JLab these can be found at /volatile/clas12/osg2/tyson/job_3143 and job_3144. This is done by default at the start of main. The rest of main then sets up the network architecture, trains the network and evaluates it. 

You can also decide to use GPU processing if you want, in which case you'll need to uncomment lines 34 and 40 of the pom.xml file (and comment out lines 35 and 41).