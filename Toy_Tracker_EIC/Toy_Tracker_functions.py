#various imports
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random

# generates a line
# arguments: size in z,x,y
# returns a track as an array size (8) containing (x_layer1,y_layer1,...,x_layer4,y_layer4).
def genLine(depth,width,height):
    x1=random.randint(0,width-1)
    y1=random.randint(0,height-1)
    z1=0*depth
    x2=random.randint(0,width-1)
    y2=random.randint(0,height-1)
    z2=1*depth
    
    l=(x2 - x1)
    m=(y2 - y1)
    n=(z2 - z1)
    # (x – x1)/l = (y – y1)/m = (z – z1)/n
    z3=2*depth
    x3=l*((z3-z1)/n)+x1
    y3=m*((z3-z1)/n)+y1
    
    z4=3*depth
    x4=l*((z4-z1)/n)+x1
    y4=m*((z4-z1)/n)+y1
    
    return np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

# checks if all hits in a track are within the layer
# arguments: a track and the x,y size of the layers
# returns true if the track is contained in the detector, false otherwise
def check_acceptance(track,width,height):
    track_in_det=False
    
    #loop over all hits
    for i in range(0,len(track),2):
        #print(track[i])
        #print(track[i+1])
        
        #check if hit is within the detector
        if track[i]>0 and track[i]<width:
            if track[i+1]>0 and track[i+1]<height:
                track_in_det=True
            else:
                track_in_det=False
        else:
            track_in_det=False
    return track_in_det

# smears a track by adding a value randomly sampled from a gaussian with mean 0, standard deviation res
# arguments: the track to be smeared and the standard deviation of the gaussian, x,y size ofthe layers
# returns: smeared track
def smear(track,res,width,height):
    if res!=0:
        track=track+np.random.normal(0,res,8)
    #check track hasn't been smeared out of layer
    for i in range(0,8,2):
        #if the track is a number between width-0.5 and width it'll be rounded to width
        #layer size goes from 0 to width-1 so recast as width-1
        #same with height
        if track[i]>(width-1):
            track[i]=width-1
        if track[i]<0:
            track[i]=0
        if track[i+1]>(height-1):
            track[i+1]=height-1
        if track[i+1]<0:
            track[i+1]=0 
            
    #print(track)
    #print(np.rint(track).astype(int))
    #need to cast the array to integers as smearing can be float
    return np.rint(track).astype(int)
    

# generate a track by generating a line, checking that the track is in the detector and smearing the track
# arguments: z,x,y size of detector and resolution
# returns: a smeared track in the detector
def genTrack(depth,width,height,res):
    track=np.zeros((1,8))
    track_in_det=False
    while track_in_det==False:
        track=genLine(depth,width,height)    
        track_in_det=check_acceptance(track,width,height)
        
    return smear(track,res,width,height)

# scale an array of tracks between 0 and 1
# arguments: array of tracks, x/y size of layers.
# returns: an array of scaled tracks
def norm(X,width,height):
    X=X.astype(np.float64)
    X[:,0]=X[:,0]/width
    X[:,1]=X[:,1]/height
    X[:,2]=X[:,2]/width
    X[:,3]=X[:,3]/height
    X[:,4]=X[:,4]/width
    X[:,5]=X[:,5]/height
    X[:,6]=X[:,6]/width
    X[:,7]=X[:,7]/height
    return X

# unscale an array of tracks between 0 and 1
# arguments: array of tracks, x/y size of layers.
# returns: an array of unscaled tracks
def unnorm(X,width,height):
    X=X.astype(np.float32)
    X[:,0]=X[:,0]*width
    X[:,1]=X[:,1]*height
    X[:,2]=X[:,2]*width
    X[:,3]=X[:,3]*height
    X[:,4]=X[:,4]*width
    X[:,5]=X[:,5]*height
    X[:,6]=X[:,6]*width
    X[:,7]=X[:,7]*height
    return X.astype(int)

# adds noise to an event by randomly picking N points in each layer, noise are set to -1 in the event array
# arguments: x,y size of layers, the array containing all layers, the amount of noisy hits to add
# returns: the array containing all layers with added noise
def noiseEvent(width,height,event_lattice,nbNoise):
    for layer in range(0,4):
        randomX=np.random.randint(0,width-1, size=nbNoise, dtype=int)
        randomY=np.random.randint(0,height-1, size=nbNoise, dtype=int)
        for i in range(0,nbNoise):
            event_lattice[randomX[i],randomY[i],layer]=-1 #allows to see noise in plots
    return event_lattice

# plots a single layer of the tracker
# arguments: the layer, x,y size of the layer, how often to add ticks (ie every N), the event canvas containing
# all layers, the layer number
def plotLayer(layer,width,height,ticks,axes,i):
    #fig=plt.figure(figsize = (20,20))
    sns.heatmap(layer,ax=axes[i],cmap='vlag_r', vmin=-1, vmax=1)
    for ind, label in enumerate(axes[i].get_xticklabels()):
        if (ind % ticks) == 0:  # every Nth label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    for ind, label in enumerate(axes[i].get_yticklabels()):
        if (ind % ticks) == 0:  # every Nth label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    #axes[i].invert_yaxis()
    axes[i].set(xlabel='x [AU]')
    axes[i].set(ylabel='y[AU]')
    #plt.show()

# plots an event as a series of heatmaps representing each layer in one canvas
# arguments: the array containing all layers, x,y size of the layer, how often to add ticks (ie every N)
def plotEvent(event_lattice,width,height,ticks):
    fig, axes = plt.subplots(1, 4, figsize=(40, 7))#, sharey=True)
    fig.suptitle('Tracker Layers 1 to 4')
    for i in range(0,4):
        plotLayer(event_lattice[:,:,i],width,height,ticks,axes,i)
    plt.show()
     
# distorts in track in an array of tracks by randomly shifting the position of a hit in a randomly chosen layer
# arguments: array of tracks, x,y size of the layers
# returns: array of distorted tracks, same size as the array of original tracks
def distort_tracks(tracks,width,height):
    
    fake_tracks=tracks.copy()
    
    #loop over tracks
    for i in range(fake_tracks.shape[0]):
        #choose layer to distort at random
        layer=np.random.randint(0,4, size=1, dtype=int)
        new_x=fake_tracks[i,2*layer]
        new_y=fake_tracks[i,2*layer+1]
        #make sure you don't accidentally pick same x,y position
        while new_x==fake_tracks[i,2*layer]:
            new_x=np.random.randint(0,width-1, size=1, dtype=int)
        while new_y==fake_tracks[i,2*layer]:
            new_y=np.random.randint(0,height-1, size=1, dtype=int)
        fake_tracks[i,2*layer]=new_x[0]
        fake_tracks[i,2*layer+1]=new_y[0]
        
    
    return fake_tracks

# add an array of tracks to the array containing all layers
# arguments: array containg all layers, array containing tracks, whether to add tracks as 1 or -1 to array containg
# all layers, this allows to ID noise as -1 and true hits as 1
def addTracks(event_lattice,tracks,sign):
    for i in range(0,tracks.shape[0]):
        addTrack(event_lattice,tracks[i],sign)

# add a single track to the array containg all layers
# arguments: array containg all layers, track, whether to add tracks as 1 or -1 to array containg
# all layers, this allows to ID noise as -1 and true hits as 1
def addTrack(event_lattice,track,sign):
    for j in range(0,len(track),2):
        #track[j+1]=y, track[j]=x, numpy array filled with a[y,x]
        
        #only fill unempty cells
        if event_lattice[track[j+1],track[j],round(j/2)]==0:
            event_lattice[track[j+1],track[j],round(j/2)]=sign
        
    return event_lattice

# find hits (ie non-zero element) in a given layer
# arguments: array containg all hits in a layer
# returns: array containg all x,y position of hits in a layer
def find_hits(layer):
    nz=np.nonzero(layer) #nz[1] corresponds to x pos (columns),nz[0] corresponds to y pos (rows)
    hits=np.hstack((nz[1].reshape((len(nz[1]),1)),nz[0].reshape((len(nz[0]),1))))
    #print(hits)
    return hits

# find hits (ie non-zero element) in a given layer, scale the output array between 0 and 1
# arguments: array containg all hits in a layer, x,y size of layers
# returns: array containg all x,y position of hits in a layer scaled between 0 and 1
def find_hits_normed(layer,width,height):
    nz=np.nonzero(layer) #nz[1] corresponds to x pos (columns),nz[0] corresponds to y pos (rows)
    nz_x=nz[1].reshape((len(nz[1]),1)).astype(np.float64)
    nz_y=nz[0].reshape((len(nz[0]),1)).astype(np.float64)
    hits=np.hstack((nz_x/width,nz_y/height))
    #print(hits)
    return hits



# find all track candidates in an event, ID if these are true tracks or fake tracks. IDing tracks is slow and should
# only be used for training. An argument allows to decide whether or not to ID tracks
# arguments: array containg all layers, array containing all true tracks, whether or not we want to label tracks
# as true or false.
# returns: all possible tracks and array containing label as to whether a track is true or fake, this will be
# non meaningful if the argument meaningful_labels==False
def find_track_candidates(event_lattice,tracks,meaningful_labels):
    hits_l0=find_hits(event_lattice[:,:,0])
    hits_l1=find_hits(event_lattice[:,:,1])
    hits_l2=find_hits(event_lattice[:,:,2])
    hits_l3=find_hits(event_lattice[:,:,3])
    
    new_tracks=np.zeros((1,8))
    
    n_tracks=0
    #create all permutations of hits in layer
    for l0 in hits_l0:
        for l1 in hits_l1:
            for l2 in hits_l2:
                for l3 in hits_l3:
                    track=np.hstack((l0,l1,l2,l3))
                    if n_tracks==0:
                        new_tracks=track
                    else:
                        new_tracks=np.vstack((new_tracks,track))
                    n_tracks=n_tracks+1
                    
    #print(new_tracks)
    
    #label array has two column per row. 
    #first column is 1 for fake tracks, 0 for true tracks. second column is 0 for fake tracks, 1 for true
    label=np.ones((new_tracks.shape[0],1))
    label=np.hstack((label,np.zeros((new_tracks.shape[0],1))))
    
    if meaningful_labels==True:
        for i in range(0,new_tracks.shape[0]):
            for j in range(0,tracks.shape[0]):
                #print(new_tracks[i])
                #print(tracks[j])
                if(np.array_equal(new_tracks[i],tracks[j])):
                    label[i,0]=0
                    label[i,1]=1
            
    #print(label)
    
    return new_tracks, label

#check if a track overlaps with an array of tracks
# argument: track and array of tracks
# returns: True if there's no overlap, False otherwise
def check_no_overlap(track,all_tracks):
    no_overlap=True
    for i in range(0,all_tracks.shape[0]):
        for j in range(0,8,2):
            #overlap defined as tracks sharing a hit
            if track[j]==all_tracks[i,j] and track[j+1]==all_tracks[i,j+1]:
                no_overlap=False
    return no_overlap
    
# generate an event containg a given number of tracks and noisy hits per layer
# arguments: size in z,x,y, number of tracks to generate, amount of smearing on hit position in tracks
# number of noisy hits per layer, True if want to allow tracks to share a hit, false otherwise
# returns: an array containing all true tracks and an array containing all layers
def genEvent(depth,width,height,nbTracks,res,nbNoise,allow_overlap):
    all_tracks=np.zeros((1,8))
    event_lattice=np.zeros((height,width,4))
    nbGenTracks=0
    while nbGenTracks<nbTracks:
        track=genTrack(depth,width,height,res)
        
        keep_track=True
        if allow_overlap==False and nbGenTracks>0:
            keep_track=check_no_overlap(track,all_tracks)
            
        if keep_track==True:
            event_lattice=addTrack(event_lattice,track,1)
            
            if nbGenTracks==0:
                all_tracks=track.reshape(1,8)
            else:
                all_tracks=np.vstack((all_tracks,track))
            nbGenTracks=nbGenTracks+1
            
    event_lattice=noiseEvent(width,height,event_lattice,nbNoise)

    return all_tracks, event_lattice
        
    