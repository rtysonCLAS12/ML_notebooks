# DeepLearning4J Example

In this example you will learn how to use the deeplearning4j library in Java with the CLAS12 Hipo file reading tools (see [Hipo Library](https://github.com/gavalian/hipo) ). 

This example is based on a PID classification task where the aim is to distinguish between simulated photons and neutrons detected in the CLAS12 Forward Detector based on their energy deposition in the calorimeters, and the shower width in U/V/W. 

NB: Even if the accuracy returned by the algorithm is over 90% this example is by no means complete and is not ready to be deployed in an analysis.

## Compile with Maven

First off many thanks to Angela Biselli for helping me figure this out.

- cd to where you want to download the repository.
- git clone https://github.com/rtysonCLAS12/ML_notebooks.git
- cd DL4J_Example/DL4J_Example
- mvn package
- run with java -jar target/DL4J_Example-0.0.1-SNAPSHOT-jar-with-dependencies.jar

When compiling maven will download all the necessary librairies, this will take up to ~1Gb, and produce a ~1Gb .jar file. You can change where the librairies are copied to in JLab with:
- cp /group/clas12/packages/maven/3.5.0/conf/settings.xml ~/.m2/
- set a new location in settings.xml for the repository by adding a line 
"<localRepository> /path/to/desired/location </localRepository>"


In principle this directory is also an eclipse-workspace and you should be able to load this as is (although unfortunately I'm no expert in this). Once you have loaded the (maven) project you can run this as usual, it should be ready to go!

## Additional Comments

The DL4J_Example class contains several methods for plotting relevant metrics, and reading in data from .hipo files. The main thing to note here is that you need to pass in the location of the data files, at JLab these can be found at /volatile/clas12/osg2/tyson/job_3143 and job_3144. This is done by default at the start of main. The rest of main then sets up the network architecture, trains the network and evaluates it. 

You can also decide to use GPU processing if you want, in which case you'll need to uncomment lines 22 to 26 and lines 34 and 40 of the pom.xml file (and comment out lines 35 and 41). Note if you are compiling your code to a jar file adding the redist in lines 22 to 26 greatly increases the size of your jar (and corresponding libraries). 

Some additional remarks:
- Here's a link to the deeplearning4j library  [documentation](https://deeplearning4j.konduit.ai/).
- The deeplearning4j library uses the [nd4j library](https://deeplearning4j.konduit.ai/nd4j/overview) notably for its own datatypes (INDArray etc).
- In general for classification problems it's useful to have balanced datasets with the same amount of events from each classes as this makes calculating relevant metrics much simpler.
- In general it's also useful to normalise your data as this will make it easier for your model to learn, especially when different variables have widly different ranges.
- It's generally hard to know what network architecture to use for any given task. You can therefore try playing about with the number of layers, the number of nodes per layer, their activation functions or the network updater and the loss function to try and see if you can improve the network performance in this way.
- The network is here trained on simulated data. The advantage of this is that it's easier to know what goes into your training sample, and to therefore avoid introducing any biases which might lead your model to learning the wrong thing. The disadvantage is that you need to make sure that your simulations are representative of CLAS12 data, and you need to evaluate the performance of your network on CLAS12 data.
- The above would constistute the next steps if you wanted to take this example further: you should try plotting the distributions of your features for neutron and photon classes, potentially adding more if you think that would help, and then think of ways to validate your model.
