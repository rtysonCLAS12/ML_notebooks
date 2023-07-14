#various imports
import numpy as np
from matplotlib import pyplot as plt
import time
from Toy_Tracker_functions import *
from Toy_Tracker_functions import norm, unnorm
from trackID_C12_functions import *

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

norm_true_tracks=norm(true_tracks.copy(),width,height) #use copy argument to avoid changing true_tracks
unnormed_true_tracks=unnorm(norm_true_tracks.copy(),width,height)

print('True Tracks')
print(true_tracks)
print('Normalised tracks')
print(norm_true_tracks)
print('Unnormalised tracks')
print(unnormed_true_tracks)

nb_train_events=10000

#training set of track candidates and labels
train_tracks=np.zeros((1,8))
train_labels=np.zeros((1,8))

for i in range(nb_train_events):
    
    if (i%500)==0:
        print('Generated '+str(i)+' events')
    
    true_tracks,event_array=genEvent(depth,width,height,nbTracks,resolution,nbNoise,allow_overlap)
    
    #can distort tracks, this is faster but didn't work as well
    
    #distorted_tracks=distort_tracks(true_tracks.copy(),width,height)
    #labels_pos=np.zeros((true_tracks.shape[0],1))
    #labels_pos=np.hstack((labels_pos,np.ones((true_tracks.shape[0],1))))
    #labels_neg=np.ones((distorted_tracks.shape[0],1))
    #labels_neg=np.hstack((labels_neg,np.zeros((distorted_tracks.shape[0],1))))
    
    #if i==0:
    #    train_tracks=np.vstack((true_tracks,distorted_tracks))
    #    train_labels=np.vstack((labels_pos,labels_neg))
    #else:
    #    train_tracks=np.vstack((train_tracks,true_tracks,distorted_tracks))
    #    train_labels=np.vstack((train_labels,labels_pos,labels_neg))
    
    #add track candidates and labels to training set
    track_candidates,labels=find_track_candidates(event_array,true_tracks,True)
    #balance dataset here to avoid keeping loads of fake tracks we won't use
    track_candidates,labels=balance_dataset(track_candidates,labels)
    if i==0:
        train_tracks=track_candidates
        train_labels=labels
    else:
        train_tracks=np.vstack((train_tracks,track_candidates))
        train_labels=np.vstack((train_labels,labels))

print(train_tracks.shape)
train_tracks=norm(train_tracks,width,height)
#train classifier, 50 epochs ok without smearing, 100 otherwise
track_identifier=train_trackID(train_tracks,train_labels,100)

nb_test_events=1000

AvEff=0
AvPur=0

AvEff_cr=0
AvPur_cr=0

AvTime_getEvent=0
AvTime_getCandidates=0
AvTime_apply=0
AvTime_apply_cr=0

AvNbTracks=0

for i in range(nb_test_events):
    
    if (i%100)==0:
        print('Generated '+str(i)+' events')

    #timing to get an event
    t0_getEvent = time.time()
    
    #generate event
    true_tracks,event_array=genEvent(depth,width,height,nbTracks,resolution,nbNoise,allow_overlap)
    
    t1_getEvent = time.time()
    AvTime_getEvent=AvTime_getEvent+(t1_getEvent-t0_getEvent)

    #timing to find track candidates (not requiring meaningful labels)
    t0_getCandidates = time.time()
    
    track_candidates,labels=find_track_candidates(event_array,true_tracks,False)
    
    t1_getCandidates = time.time()
    AvTime_getCandidates=AvTime_getCandidates+(t1_getCandidates-t0_getCandidates)
    
    #number of tracks per event
    AvNbTracks=AvNbTracks+track_candidates.shape[0]
    
    #normalise track candidates
    track_candidates=norm(track_candidates,width,height)
    
    #timing to apply ID selecting only tracks with best response
    t0_apply = time.time()
    
    #apply track identification selecting only tracks with best response
    selected_tracks,rejected_tracks=apply_trackID(track_identifier,track_candidates.copy())
    
    t1_apply = time.time()
    AvTime_apply=AvTime_apply+(t1_apply-t0_apply)
    
    #timing to apply ID when cutting on the response
    t0_apply_cr = time.time()
    
    selected_tracks_cr,rejected_tracks_cr=apply_trackID_cutResp(track_identifier,track_candidates.copy(),0.9)
    
    t1_apply_cr = time.time()
    AvTime_apply_cr=AvTime_apply_cr+(t1_apply_cr-t0_apply_cr)

    #unnorm selected and rejected tracks
    selected_tracks=unnorm(selected_tracks,width,height)
    rejected_tracks=unnorm(rejected_tracks,width,height)
    #unnorm selected and rejected tracks
    selected_tracks_cr=unnorm(selected_tracks_cr,width,height)
    rejected_tracks_cr=unnorm(rejected_tracks_cr,width,height)
    
    #print the first event
    if i==0:
        print('')
        print('True Tracks, size: '+str(true_tracks.shape[0]))
        print(true_tracks)
        print('')
        print('Selected Tracks, size: '+str(selected_tracks_cr.shape[0]))
        print(selected_tracks_cr)
        print('')
        print('Rejected Tracks, size: '+str(rejected_tracks_cr.shape[0]))
        print(rejected_tracks_cr)
        print('')
        event_array_2=np.zeros((height,width,4))
        addTracks(event_array_2,selected_tracks_cr,1)
        plotEvent(event_array,width,height,ticks)
        plotEvent(event_array_2,width,height,ticks)

    #get metrics, these will be averaged over all events
    eff,pur=calculate_metrics(true_tracks,selected_tracks,rejected_tracks)
    eff_cr,pur_cr=calculate_metrics(true_tracks,selected_tracks_cr,rejected_tracks_cr)
    #print(eff)
    #print(pur)
    AvEff=AvEff+eff
    AvPur=AvPur+pur
    AvEff_cr=AvEff_cr+eff_cr
    AvPur_cr=AvPur_cr+pur_cr
    
#average metrics, nb of tracks and times
AvEff=AvEff/nb_test_events
AvPur=AvPur/nb_test_events

AvEff_cr=AvEff_cr/nb_test_events
AvPur_cr=AvPur_cr/nb_test_events


AvTime_getEvent=AvTime_getEvent/nb_test_events
AvTime_getCandidates=AvTime_getCandidates/nb_test_events
AvTime_apply=AvTime_apply/nb_test_events
AvTime_apply_cr=AvTime_apply_cr/nb_test_events

AvNbTracks=AvNbTracks/nb_test_events

print('')
print('Fraction of true tracks that survive '+str(AvEff))
print('Fraction of false tracks that survive '+str(AvPur))
print('')
print('Fraction of true tracks that survive '+str(AvEff_cr)+' with Resp>0.9')
print('Fraction of false tracks that survive '+str(AvPur_cr)+' with Resp>0.9')

print('')
print('On average had '+str(AvNbTracks)+' tracks per event')
print('Generating an event took on average '+str(AvTime_getEvent)+'s')
print('Finding all track candidates took on average '+str(AvTime_getCandidates)+'s')
print('Applying the track ID took on average '+str(AvTime_apply)+'s')
print('Applying the track ID (when cutting on response) took on average '+str(AvTime_apply_cr)+'s')