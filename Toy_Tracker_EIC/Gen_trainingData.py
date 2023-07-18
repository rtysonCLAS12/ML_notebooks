import numpy as np
from object_condesation_functions import get_hits_GNet
from Toy_Tracker_functions import *

nb_events=10000

width=20
height=20
depth=5*width

nbTracks=2

resolution=0#(width/10)
nbNoise=4#round((width*height)/10)

ticks=1

vmax=24#width*height*4

allow_overlap=False

true_tracks,event_array=genEvent(depth,width,height,nbTracks,resolution,nbNoise,allow_overlap)

print(event_array.shape)

distorted_tracks=distort_tracks(true_tracks.copy(),width,height)
print('True Tracks')
print(true_tracks)
print('Distorted Tracks')
print(distorted_tracks)

addTracks(event_array,distorted_tracks,-1)

plotEvent(event_array,width,height,ticks)

track_candidates,labels=find_track_candidates(event_array,true_tracks,True)

print('True Tracks from all Candidates')
print(track_candidates[labels[:,1]==1])

for j in range(10,20):
    print('\nIteration '+str(j))
    train_hits=np.zeros((1,1))
    train_size=np.zeros((1,1))
    train_truth=np.zeros((1,1))

    for i in range(nb_events):
    
        if (i%1000)==0:
            print('Generated '+str(i)+' events')

        true_tracks,event_array=genEvent(depth,width,height,nbTracks,resolution,nbNoise,allow_overlap)

        hits,size,truth=get_hits_GNet(event_array,width,height,depth,vmax,true_tracks)
    
        if i==0:
            train_hits=hits
            train_size=size
            train_truth=truth
        else:
            train_hits=np.vstack((train_hits,hits))
            train_size=np.vstack((train_size,size))
            train_truth=np.vstack((train_truth,truth))

    np.save('/scratch/richardt/Toy_Tracker_EIC/data/truth_'+str(j)+'.npy',train_truth)
    np.save('/scratch/richardt/Toy_Tracker_EIC/data/size_'+str(j)+'.npy',train_size)
    np.save('/scratch/richardt/Toy_Tracker_EIC/data/hits_'+str(j)+'.npy',train_hits)
    


true_tracks,event_array=genEvent(depth,width,height,nbTracks,resolution,nbNoise,allow_overlap)
    
hits,size,truth=get_hits_GNet(event_array,width,height,depth,vmax,true_tracks)

print('True Tracks')
print(norm(true_tracks.copy(),width,height))
print('\n hits')
print(hits)
print('\nsize')
print(size)
print('\ntruth')
print(truth)
