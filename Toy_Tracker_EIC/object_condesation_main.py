#various imports
import numpy as np
from matplotlib import pyplot as plt
import time
from object_condesation_functions import train_GNet_trackID

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

allow_overlap=False



train_hits=np.load('toy_tracker_data/hits_0.npy')
train_size=np.load('toy_tracker_data/size_0.npy')
train_truth=np.load('toy_tracker_data/truth_0.npy')

for j in range(1,2):
    train_hits=np.vstack((train_hits,np.load('toy_tracker_data/hits_'+str(j)+'.npy')))
    train_size=np.vstack((train_size,np.load('toy_tracker_data/size_'+str(j)+'.npy')))
    train_truth=np.vstack((train_truth,np.load('toy_tracker_data/truth_'+str(j)+'.npy')))

train_hits=train_hits[:,:24,:]
train_size=train_size[:24,:]
train_truth=train_truth[:,:24,:]

#print(train_hits)
GNet_track_identifier=train_GNet_trackID(train_hits,train_size,train_truth)