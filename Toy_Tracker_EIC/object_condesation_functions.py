from Toy_Tracker_functions import norm, unnorm, find_hits_normed
from garnet import GarNetStack
from Layers import GravNet_simple, GlobalExchange
from betaLosses import object_condensation_loss
import numpy as np
from matplotlib import pyplot as plt
import math
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,BatchNormalization,Concatenate, Lambda#,concatenate
import tensorflow.keras as keras
K = keras.backend

# scale an array of tracks between 0 and 1. Here the array contains all tracks in an event stacked on one row
# arguments: array of tracks, x/y size of layers, number of tracks per event.
# returns: an array of scaled tracks
def norm_GNet(X,width,height,nbTracks):
    X=X.astype(np.float32)
    for i in range(0,nbTracks):
        X[:,i*8+0]=X[:,i*8+0]/width
        X[:,i*8+1]=X[:,i*8+1]/height
        X[:,i*8+2]=X[:,i*8+2]/width
        X[:,i*8+3]=X[:,i*8+3]/height
        X[:,i*8+4]=X[:,i*8+4]/width
        X[:,i*8+5]=X[:,i*8+5]/height
        X[:,i*8+6]=X[:,i*8+6]/width
        X[:,i*8+7]=X[:,i*8+7]/height
    return X

# unscale an array of tracks between 0 and 1. Here the array contains all tracks in an event stacked on one row
# arguments: array of tracks, x/y size of layers, number of tracks per event.
# returns: an array of unscaled tracks
def unnorm_GNet(X,width,height,nbTracks):
    X=X.astype(np.float32)
    for i in range(0,nbTracks):
        X[:,i*8+0]=X[:,i*8+0]*width
        X[:,i*8+1]=X[:,i*8+1]*height
        X[:,i*8+2]=X[:,i*8+2]*width
        X[:,i*8+3]=X[:,i*8+3]*height
        X[:,i*8+4]=X[:,i*8+4]*width
        X[:,i*8+5]=X[:,i*8+5]*height
        X[:,i*8+6]=X[:,i*8+6]*width
        X[:,i*8+7]=X[:,i*8+7]*height
    return X.astype(int)

# check if a hit belongs to a true_track. NB true_tracks and hits aready scaled between 0 and 1
# arguments: a hit (x,y), which layer the hit belongs to, true_tracks.
# returns: first a label saying whether the hit is noise or not, second a label saying which track it belongs too.
# tracks don't need to be ordered, but all hits in a track must have same label, here we use the order 
# of the track in true_tracks as the label. Noise are labeled as belonging to track 9999.
def is_hit_in_track(hit,layer,true_tracks):
    nonoise=0
    n_obj=9999
    for k in range(0,true_tracks.shape[0]):
        if true_tracks[k,2*layer]==hit[0] and true_tracks[k,2*layer+1]==hit[1]:
            nonoise=1
            n_obj=k
    return nonoise,n_obj
            


# get an array of hits in an event and number of real hits contained in the hits array, 
# the rest of the hits array is all zero. Also get a label whether the hit is noise or not and which true track
# the hit belongs to
# arguments: array containing hits in all layers, the x/y/z size of tracker, the max number of hits in an event,
# and an array of true_tracks in event
# returns: a list of hits in an event and how many hits we have in this event, label whether the hit is noise 
# or not and which true track the hit belongs to
def get_hits_GNet(event_lattice,width,height,depth,vmax,true_tracks):
    
    hits_l0=find_hits_normed(event_lattice[:,:,0],width,height)
    hits_l1=find_hits_normed(event_lattice[:,:,1],width,height)
    hits_l2=find_hits_normed(event_lattice[:,:,2],width,height)
    hits_l3=find_hits_normed(event_lattice[:,:,3],width,height)
    
    #add z info
    hits_l0=np.hstack((hits_l0,np.ones((hits_l0.shape[0],1))/4))
    hits_l1=np.hstack((hits_l1,2*np.ones((hits_l1.shape[0],1))/4))
    hits_l2=np.hstack((hits_l2,3*np.ones((hits_l2.shape[0],1))/4))
    hits_l3=np.hstack((hits_l3,np.ones((hits_l3.shape[0],1))))
    
    #print(hits_l3)
    
    #at present array has fixed sized vmax,3
    all_hits=np.zeros((vmax,3))
    size=np.zeros((1,1))
    
    # truth contains first a label saying whether the hit is noise or not
    # second a label saying which track it belongs too.
    # tracks don't need to be ordered, but all hits in a track must have same label
    # here we use the order of the track in true_tracks as the label
    # noise are labeled as belonging to track 9999
    truth=np.zeros((vmax,2))
    
    #hits already normed
    true_tracks_normed=norm(true_tracks.copy(),width,height)
    
    index=0
    for hit in hits_l0:
        all_hits[index]=hit
        nonoise,n_obj=is_hit_in_track(hit,0,true_tracks_normed)
        truth[index,0]=nonoise
        truth[index,1]=n_obj
        index=index+1
    for hit in hits_l1:
        all_hits[index]=hit
        nonoise,n_obj=is_hit_in_track(hit,1,true_tracks_normed)
        truth[index,0]=nonoise
        truth[index,1]=n_obj
        index=index+1
    for hit in hits_l2:
        all_hits[index]=hit
        nonoise,n_obj=is_hit_in_track(hit,2,true_tracks_normed)
        truth[index,0]=nonoise
        truth[index,1]=n_obj
        index=index+1
    for hit in hits_l3:
        all_hits[index]=hit
        nonoise,n_obj=is_hit_in_track(hit,3,true_tracks_normed)
        truth[index,0]=nonoise
        truth[index,1]=n_obj
        index=index+1
        
    size[0,0]=index
    
    return all_hits.reshape((1,vmax,3)),size,truth.reshape((1,vmax,2))

#create a model using simple garnet layers.
# arguments: vmax is max number of hit, quantize is whether to apply some transformation to data
# returns: model using garnet layers.
def make_model_old(vmax, quantize):
    x = keras.layers.Input(shape=(vmax, 3),name='hits')
    n = keras.layers.Input(shape=(1,), dtype='uint16',name='size')
    inputs = [x, n]

    v = GarNetStack([4, 4, 8], [8, 8, 16], [8, 8, 16], simplified=True, collapse='mean', input_format='xn', output_activation='tanh', name='gar_1', quantize_transforms=quantize)([x, n])
    v = Dense(64, activation='relu')(v)
    v = Dense(32, activation='relu')(v)
    out_beta=Dense(1,activation='sigmoid')(v)
    out_latent=Dense(2)(v)
    #out_latent = Lambda(lambda x: x * 10.)(out_latent)
    out=Concatenate()([out_beta, out_latent])
    
    return keras.Model(inputs=inputs, outputs=out)

#create a model using simple gravnet layers.
# arguments: vmax is max number of hit
# returns: model using gravnet layers.
def make_model(vmax):
    x = keras.layers.Input(shape=(vmax, 3),name='hits')
    n = keras.layers.Input(shape=(1,), dtype='uint16',name='size')
    #inputs = [x, n]
    inputs=x
    
    #v = BatchNormalization(momentum=0.6)(inputs)
    
    feat=[inputs]
    
    for i in range(2):#6
        #add global exchange and another dense here
        v = GlobalExchange()(inputs)
        v = Dense(64, activation='elu')(v)
        v = Dense(64, activation='elu')(v)
        v = BatchNormalization(momentum=0.6)(v)
        v = Dense(64, activation='elu')(v)
        v = GravNet_simple(n_neighbours=10, 
                 n_dimensions=4, 
                 n_filters=128, 
                 n_propagate=64)(v)
        v = BatchNormalization(momentum=0.6)(v)
        feat.append(Dense(32, activation='elu')(v))
    
    v = Concatenate()(feat)
    v = Dense(64, activation='elu')(v)
    out_beta=Dense(1,activation='sigmoid')(v)
    out_latent=Dense(2)(v)
    #out_latent = Lambda(lambda x: x * 10.)(out_latent)
    out=Concatenate()([out_beta, out_latent])
    
    return keras.Model(inputs=inputs, outputs=out)


#make plot of latent space representation of first event in data, useful to see clustering
#arguments: network prediction, truth (noise and obj number), add something to title (ie epoch nb N)
#number of tracks we're trying to ID, maximum amount of hits
def plot_latent_space(pred,truth,title_add,n_tracks,vmax):
    pred_latent_x=pred[0,:,1].reshape((vmax,1))
    pred_latent_y=pred[0,:,2].reshape((vmax,1))
    truth_objid=truth[0,:,1].reshape((vmax,1))

    fig = plt.figure(figsize=(20, 20))
    for i in range(n_tracks):
        plt.scatter(x=pred_latent_x[truth_objid==i],s=200,y=pred_latent_y[truth_objid==i],label='Track '+str(i))
    plt.scatter(x=pred_latent_x[truth_objid==9999],s=200,y=pred_latent_y[truth_objid==9999],label='Noise')
    plt.title('Learned Latent Space '+title_add)
    plt.ylabel('Coordinate 2')
    plt.xlabel('Coordinate 0')
    plt.legend(loc='upper left')
    plt.show()

#plot training history
#arguments: history, contains loss and val_loss as a function of epochs
def plot_history(history):
    fig = plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#train object condensation model
#arguments: hits in detector, number of real hits (the rest of the hits array is all zero),
#truth info (noise or not and track number)
#returns: trained object condensation model
def train_GNet_trackID(hits,size,truth):
    
    vmax=hits.shape[1]
    
    nbTrain=math.ceil(2*hits.shape[0]/3)
    
    hits_train=hits[:nbTrain,:]
    hits_test=hits[nbTrain:,:]
    
    size_train=size[:nbTrain,:]
    size_test=size[nbTrain:,:]
    
    y_train=truth[:nbTrain,:]
    y_test=truth[nbTrain:,:]
    
    print(hits_train.shape)
    print(hits_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    model = make_model(vmax)
    
    opti=Adam(learning_rate=0.0005)
    model.compile(loss=object_condensation_loss, optimizer=opti)
    
    
    pred_test=model.predict(hits_test[0].reshape((1,vmax,3)))
    plot_latent_space(pred_test,y_test[0].reshape((1,vmax,2)),'(Before Training)',2,vmax)
    
    for i in range(0,10):
        history=model.fit(hits_train,y_train,epochs=10, validation_data=(hits_test, y_test), verbose=1)
        
        pred_test=model.predict(hits_test[0].reshape((1,vmax,3)))
        plot_latent_space(pred_test,y_test[0].reshape((1,vmax,2)),'(Epoch '+str(i*50+50)+')',2,vmax)
        
        plot_history(history)
        
    pred_test=model.predict(hits_test[0].reshape((1,vmax,3)))
    plot_latent_space(pred_test,y_test[0].reshape((1,vmax,2)),'(After Training)',2,vmax)

    
    return model

#apply gravnet model, not yet finished
def apply_GNet_trackID(track_identifier,hits,size):
    pred_tracks = track_identifier.predict({"hits": hits, "size": size})
        
    return pred_tracks

#calculate metrics like track efficiency, not yet finished
def calculate_GNet_metrics(true_tracks,selected_tracks):
    TP=0
    FP=0
    FN=0
    
    
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
            
    eff=TP/true_tracks.size[0]
    FP_eff=FP/true_tracks.size[0]

