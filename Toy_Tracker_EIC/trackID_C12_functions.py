#various imports
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
K = keras.backend

# train a neural network to identify true tracks.
# arguments: list of track candidates, labels to know if this is a true or fake track, number of training epochs
# returns: the trained neural network
def train_trackID(track_candidates,labels,nEpochs):
    
    #make sure that there are as many true tracks as there are false. Otherwise the classifier might be biased.
    X,y=balance_dataset(track_candidates,labels)
    
    #split tracks into a training or testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=5)
    
    print('Train/test sample shape for true tracks')
    print(X_train[y_train[:,1]==1].shape)
    print(X_test[y_test[:,1]==1].shape)
    print('Train/test sample shape for fake tracks')
    print(X_train[y_train[:,1]==0].shape)
    print(X_test[y_test[:,1]==0].shape)
    
    #define neural network architecture
    model = Sequential()
    model.add(Dense(track_candidates.shape[1],input_dim=track_candidates.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax')) #lat layer has two nodes, one for false track, one for true track
    
    opti=Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opti,metrics=["accuracy"])
    model.summary()
    
    #train
    history=model.fit(X_train,y_train,epochs=nEpochs, validation_data=(X_test, y_test), verbose=2)
    
    #plot loss as a function of training epoch
    fig = plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    #plot the output (ie response) of the neural network on the testing set
    plot_trackID_response(model,X_test,y_test)
    
    return model

# make sure there is as many true and fake tracks in a dataset
# arguments: list of track candidates, labels to know if this is a true or fake track
# returns: balanced list of track candidates, labels to know if this is a true or fake track
def balance_dataset(track_candidates,labels):
    X_pos=track_candidates[labels[:,1]==1]
    X_neg=track_candidates[labels[:,1]==0]
    y_pos=labels[labels[:,1]==1]
    y_neg=labels[labels[:,1]==0]
    
    #randomly shuffle the arrays as sometimes these have all fake tracks at the end
    p_pos = np.random.permutation(len(X_pos))
    p_neg = np.random.permutation(len(X_neg))
    
    X_pos=X_pos[p_pos]
    y_pos=y_pos[p_pos]
    X_neg=X_neg[p_neg]
    y_neg=y_neg[p_neg]
    
    #check if we have more fake or true tracks
    if(len(X_neg)>len(X_pos)):
        X_neg=X_neg[0:len(X_pos),:]
        y_neg=y_neg[0:len(X_pos),:]
    else:
        X_pos=X_pos[0:len(X_neg),:]
        y_pos=y_pos[0:len(X_neg),:]
        
    #print(X_pos.shape)
    #print(X_neg.shape)
        
    return np.vstack((X_pos,X_neg)),np.vstack((y_pos,y_neg))

# plot the response of the neural network when predicting on a set of tracks
# arguments: the neural network, list of track candidates, labels to know if this is a true or fake track
def plot_trackID_response(track_identifier,track_candidates,labels):
    y_pred = track_identifier.predict(track_candidates)[:, 1]
    y=labels[:, 1]
    
    fig = plt.figure(figsize=(20,20))
    plt.hist(y_pred[y==1], range=[0,1],bins=100,color='royalblue', label='True Tracks')#, range=[2.7,3.3]
    plt.hist(y_pred[y==0], range=[0,1],bins=100, edgecolor='firebrick',label='False Tracks',hatch='/',fill=False)
    plt.legend(loc='upper center')#can change upper to lower and center to left or right
    plt.xlabel('Response')
    plt.yscale('log', nonpositive='clip')
    plt.title('Track ID Response')
    plt.show()
    
# apply track identification by only keeping tracks above a certain threshold
# arguments: the neural network, list of track candidates, the threshold on the response
# returns: array of track to keep and array of tracks to throw away
def apply_trackID_cutResp(track_identifier,track_candidates,resp_th):
    #get the response for the set of tracks
    y_pred = track_identifier.predict(track_candidates)[:, 1]
    kept_tracks=track_candidates[y_pred>=resp_th]
    rejected_tracks=track_candidates[y_pred<resp_th]
        
    return kept_tracks,rejected_tracks

# apply track identification by only keeping track with highest response by discarding
# tracks that share hits with tracks that have a higher response than them
# arguments: the neural network, list of track candidates
# returns: array of track to keep and array of tracks to throw away
def apply_trackID(track_identifier,track_candidates):
    #get the response for the set of tracks
    y_pred = track_identifier.predict(track_candidates)[:, 1]
     
    #sort tracks and response by the response going from lowest to highest
    sort = np.argsort(y_pred)
    y_pred=y_pred[sort]
    track_candidates=track_candidates[sort]
    
    kept_tracks=np.zeros((1,8))
    rejected_tracks=np.zeros((1,8))
    
    #ID which selected track we're comparing to
    track_number=0
    
    #keep going until we have no tracks left
    while track_candidates.size != 0:
        
        #last track in array has highest response
        if track_number==0:
            kept_tracks=track_candidates[-1].reshape(1,8)
        else:
            kept_tracks=np.vstack((kept_tracks,track_candidates[-1]))
         
        #keep all tracks except the one with the highest response
        track_candidates=track_candidates[:-1]
        
        rows_to_del=[]
        #loop over tracks
        for i in range(0,track_candidates.shape[0]):
            to_del=False
            #loop over hits in track
            for j in range(0,8,2):
                #discard track if it shares a hit with the selected track
                if kept_tracks[track_number,j]==track_candidates[i,j] and kept_tracks[track_number,j+1]==track_candidates[i,j+1]:
                    to_del=True
            if to_del==True:
                rows_to_del.append(i)
        
        #keep the tracks we're rejecting
        if track_number==0:
            rejected_tracks=track_candidates[rows_to_del, :]
        else:
            rejected_tracks=np.vstack((rejected_tracks,track_candidates[rows_to_del, :]))
        track_candidates=np.delete(track_candidates, rows_to_del, axis=0)
                
        track_number=track_number+1
        
    return kept_tracks,rejected_tracks
    
# calculate the fraction of true and fake tracks that are selected by the alogrithm
# arguments: arrays of true tracks, selected tracks and rejected tracks
def calculate_metrics(true_tracks,selected_tracks,rejected_tracks):
    TP=0
    FP=0
    FN=0
    
    # check if a true track is in the selected tracks
    for i in range(0,selected_tracks.shape[0]):
        matched=False
        for j in range(0,true_tracks.shape[0]):
            #print(new_tracks[i])
            #print(tracks[j])
            if(np.array_equal(selected_tracks[i],true_tracks[j])):
                matched=True
        if matched==True:
            TP=TP+1
        else:
            FP=FP+1
        
    #check if a true track is in the rejected tracks
    for i in range(0,rejected_tracks.shape[0]):
        matched=False
        for j in range(0,true_tracks.shape[0]):
            #print(new_tracks[i])
            #print(tracks[j])
            if(np.array_equal(rejected_tracks[i],true_tracks[j])):
                matched=True
        if matched==True:
            FN=FN+1
            
    eff=TP/(TP+FN)
    bg_eff=FP/(selected_tracks.shape[0]+rejected_tracks.shape[0]-true_tracks.shape[0])
    
    return eff,bg_eff
    