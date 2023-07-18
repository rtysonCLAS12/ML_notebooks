#various imports
import numpy as np
from matplotlib import pyplot as plt
import time
from object_condesation_functions import train_GNet_trackID,load_data,test_GNet
from tensorflow.keras.models import load_model

#nicer plotting style
plt.rcParams.update({'font.size': 30,
                    #'font.family':  'Times New Roman',
                    'legend.edgecolor': 'black',
                    'xtick.minor.visible': True,
                    'ytick.minor.visible': True,
                    'xtick.major.size':15,
                    'xtick.minor.size':10,
                    'ytick.major.size':15,
                     'ytick.minor.size':10})

width=20
height=20
depth=5*width

nbTracks=2

resolution=0#(width/10)
nbNoise=4#round((width*height)/10)

ticks=1

vmax=24

allow_overlap=False

saveDir='/home/richardt/public_html/Toy_Tracker_EIC/object_condensation/initial_testing/'
endName=''


dataPath='/scratch/richardt/Toy_Tracker_EIC/data/'

hits, size, truth=load_data(dataPath,4,vmax)#4

#print(train_hits)
GNet_track_identifier=train_GNet_trackID(hits,size,truth,saveDir,endName,dataPath)

#saving file doesnt work because layers have same name?
#GNet_track_identifier.save("condensation_network.h5")

test_GNet(1000,GNet_track_identifier,width,height,depth,vmax,True)

