from Toy_Tracker_functions import norm, unnorm, find_hits_normed, genEvent
from garnet import GarNetStack
from Layers import GravNet_simple, GlobalExchange
from betaLosses import object_condensation_loss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import math
import time
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
    
    for i in range(12):#12 or 6
        #add global exchange and another dense here
        v = GlobalExchange()(inputs)
        v = Dense(64, activation='elu')(v)
        v = Dense(64, activation='elu')(v)
        v = BatchNormalization(momentum=0.6)(v)
        v = Dense(64, activation='elu')(v)
        v = GravNet_simple(n_neighbours=10, 
                 n_dimensions=2, #4
                 n_filters=128, 
                 n_propagate=64)(v)
        v = BatchNormalization(momentum=0.6)(v)
        feat.append(Dense(32, activation='elu')(v))
    
    v = Concatenate()(feat)
    v = Dense(64, activation='elu')(v)
    v = Dense(32, activation='elu')(v)
    out_beta=Dense(1,activation='sigmoid')(v)
    out_latent=Dense(2)(v)
    #out_latent = Lambda(lambda x: x * 10.)(out_latent)
    out=Concatenate()([out_beta, out_latent])
    
    return keras.Model(inputs=inputs, outputs=out)


#make plot of latent space representation of first event in data, useful to see clustering
#arguments: network prediction, truth (noise and obj number), add something to title (ie epoch nb N)
#where to save the plot, string at end of save name
#number of tracks we're trying to ID, maximum amount of hits
def plot_latent_space(pred,truth,title_add,n_tracks,vmax,saveDir,endName):
    pred_latent_x=pred[0,:,1].reshape((vmax,1))
    pred_latent_y=pred[0,:,2].reshape((vmax,1))
    pred_beta=pred[0,:,0].reshape((vmax,1))
    
    #underneath transparency of 0.2 the points are hard to see
    pred_beta[pred_beta<0.2]=0.2

    truth_objid=truth[0,:,1].reshape((vmax,1))
    
    #basic matplotlib color palette
    #assumes no more than 10 tracks per event, fine for now
    colors=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

    bgCol=0

    fig = plt.figure(figsize=(20, 20))

    #loop over tracks
    for i in range(n_tracks):
        scatter(pred_latent_x[truth_objid==i],pred_latent_y[truth_objid==i],colors[i],pred_beta[truth_objid==i],label='Track '+str(i),s=200)
        bgCol=i+1

    #plot noise
    scatter(pred_latent_x[truth_objid==9999],pred_latent_y[truth_objid==9999],colors[bgCol],pred_beta[truth_objid==9999],label='Noise',s=200)

    plt.title('Learned Latent Space '+title_add)
    plt.ylabel('Coordinate 1 [AU]')
    plt.xlabel('Coordinate 0 [AU]')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'latentSpace'+endName+'.png')

#make scatter plot with each point having transparency related to beta val
#arguments: x,y in latent space, color, beta array all other arguments
def scatter(x, y, color, alpha_arr, **kwarg):
    r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    plt.scatter(x, y, c=color, **kwarg)

#plot training history
#arguments: history, contains loss and val_loss as a function of epochs, 
#where to save the plot, string at end of save name
def plot_history(history,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'loss_epoch'+endName+'.png')

#plot track efficiency and purity as a function of epochs
#argument: purity, efficiency, epochs
#where to save the plot, string at end of save name
def plotMetrics_vEpochs(AvEff,AvPur,supEpochs,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(supEpochs, AvPur, marker='o', color='red',label='Purity',s=200)
    plt.scatter(supEpochs, AvEff, marker='o', color='blue',label='Efficiency',s=200)
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='lower center')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    plt.axhline(y = 0.9, color = 'grey', linestyle = '--') 
    plt.title('Metrics vs Training Epoch')
    plt.savefig(saveDir+'metrics_epoch'+endName+'.png')

#split dataset into training and testing (2/3 train 1/3 test) sets
#arguments: training data
#returns: training data split into train test
def get_train_test(hits,size,truth):
    nbTrain=math.ceil(2*hits.shape[0]/3)
    
    hits_train=hits[:nbTrain,:]
    hits_test=hits[nbTrain:,:]
    
    size_train=size[:nbTrain,:]
    size_test=size[nbTrain:,:]
    
    y_train=truth[:nbTrain,:]
    y_test=truth[nbTrain:,:]

    return hits_train,hits_test,size_train,size_test,y_train,y_test


#train object condensation model
#arguments: hits in detector, number of real hits (the rest of the hits array is all zero),
#truth info (noise or not and track number)
#where to save plots, string at end of plot save name
#path to load data during training, if '' then no data is reloaded
#returns: trained object condensation model
def train_GNet_trackID(hits,size,truth,saveDir,endName,reloadPath):
    
    vmax=hits.shape[1]
    
    hits_train,hits_test,size_train,size_test,y_train,y_test=get_train_test(hits,size,truth)
    
    print(hits_train.shape)
    print(hits_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    model = make_model(vmax)
    
    opti=Adam(learning_rate=0.0001)
    model.compile(loss=object_condensation_loss, optimizer=opti)
    
    
    #check what latent space looks like before training
    pred_test=model.predict(hits_test[0].reshape((1,vmax,3)))
    plot_latent_space(pred_test,y_test[0].reshape((1,vmax,2)),'(Before Training)',2,vmax,saveDir,endName+'_beforeTrain')
    

    nEpochs=40#40

    AvEff=[]
    AvPur=[]
    supEpochs=[]

    #do batches of super epochs for training
    for i in range(0,100):#10
        #train
        history=model.fit(hits_train,y_train,epochs=nEpochs, validation_data=(hits_test, y_test), verbose=1)
        
        #plot latent space and training loss history
        pred_test=model.predict(hits_test[0].reshape((1,vmax,3)))
        plot_latent_space(pred_test,y_test[0].reshape((1,vmax,2)),'(Epoch '+str(i*nEpochs+nEpochs)+')',2,vmax,saveDir,endName+'_supEpoch'+str(i))
        
        plot_history(history,saveDir,endName+'_supEpoch'+str(i))

        #test model by getting purity and efficiency of event
        #hardcoded for now, should think of changing this
        eff,pur=test_GNet(1000,model,20,20,5*20,24,False)
        AvEff.append(eff)
        AvPur.append(pur)
        supEpochs.append(i*nEpochs+nEpochs)
        plotMetrics_vEpochs(AvEff,AvPur,supEpochs,saveDir,endName)

        #reload data or not during super epochs
        #really helps with overfitting
        #avoids having to load in a lot of training data
        if reloadPath!='':
            print(' Reloading data, super epoch: '+str(i))
            hits, size, truth=load_data(reloadPath,4,vmax)
            hits_train,hits_test,size_train,size_test,y_train,y_test=get_train_test(hits,size,truth)
    
    
    return model

#load data from a given path, n files are loaded at random
#arguments: path to data, number of files to load from there, max number of hits
#returns: hits, size and truth arrays from path
def load_data(path,nFiles,vmax):
    fileNbs=np.random.randint(0,20,nFiles)

    hits=np.zeros((1,1,1))
    size=np.zeros((1,1,1))
    truth=np.zeros((1,1,1))

    for fileNb in fileNbs:
        #assumes no file has only 1 event
        if hits.shape[0]==1:
            hits=np.load(path+'hits_'+str(fileNb)+'.npy')
            size=np.load(path+'size_'+str(fileNb)+'.npy')
            truth=np.load(path+'truth_'+str(fileNb)+'.npy')
        else:
            hits=np.vstack((hits,np.load(path+'hits_'+str(fileNb)+'.npy')))
            size=np.vstack((size,np.load(path+'size_'+str(fileNb)+'.npy')))
            truth=np.vstack((truth,np.load(path+'truth_'+str(fileNb)+'.npy')))

    hits=hits[:,:vmax,:]
    size=size[:vmax,:]
    truth=truth[:,:vmax,:]

    return hits, size, truth

#apply gravnet model to hits from single event, returns set of tracks for event
#arguments: model, hits, size array,threshold to select condesation points
#returns: predicted tracks
def apply_GNet_trackID(track_identifier,hits,size,beta_thresh):
    pred = track_identifier.predict(hits)
    tracks=make_tracks_from_pred(hits,pred,beta_thresh)
    return tracks

#get all tracks in an event from object condensation prediction
#idea is there's one condensation point per track which has the highest 
#beta value predicted by model. we then group hits around this condensation
#point using the distance in latent space.
#in this case we select the closest hit in lc in each layer 
#arguments: all hits, prediction, threshold to select condesation points
#return: tracks in event
def make_tracks_from_pred(hits,pred,beta_thresh):

    vmax=hits.shape[1]
    all_tracks=np.zeros((1,8))

    pred_latent_coords=pred[0,:,1:3].reshape((vmax,2))
    pred_beta=pred[0,:,0].reshape((vmax))
    hits_event=hits[0,:,:].reshape((vmax,hits.shape[2]))
    
    #condensation points have high beta
    #we group other hits around these based on latent space distance
    cond_points_lc=pred_latent_coords[pred_beta>beta_thresh]
    other_lc=pred_latent_coords[pred_beta<beta_thresh]
    cond_points_hits=hits_event[pred_beta>beta_thresh]
    other_hits=hits_event[pred_beta<beta_thresh]

    #print(other_hits.shape)
    #print(cond_points_hits.shape)
        
    #loop over condensation points
    for j in range(0,cond_points_lc.shape[0]):
        dist_lc=np.zeros((other_lc.shape[0]))+1000
        #loop over other elements to assign distance
        for k in range(0,other_lc.shape[0]):
            dif_x=cond_points_lc[j,0]-other_lc[k,0]
            dif_y=cond_points_lc[j,1]-other_lc[k,1]
            dist_lc[k]=math.sqrt(dif_x**2+dif_y**2)

        track=np.zeros((1,8))
        #find best hit in each layer
        for k in range(1,5):
            #split hits and distance into layers
            #z coord normed, going from 0.25 to 1
            dist_lc_layer=dist_lc[other_hits[:,2]==k*1/4]
            other_hits_layer=other_hits[other_hits[:,2]==k*1/4]

            #print('layer '+str(k)+' '+str(k*1/4))
            #print(other_hits_layer.shape)
            #print(track.shape)

            #sort by distance from lowest to highest
            sort = np.argsort(dist_lc_layer)
            dist_lc_layer=dist_lc_layer[sort]
            other_hits_layer=other_hits_layer[sort]

            #if only condensation points in one layer or
            # or there's no noise in a layer or
            #if network is a bit rubbish it might not assign noise beta
            #under threshold in a given layer
            if(other_hits_layer.shape[0]>0):
                #first element has lowest distance
                track[0,(k-1)*2]=other_hits_layer[0,0]
                track[0,(k-1)*2+1]=other_hits_layer[0,1]
        
        #replace closest point in same layer as condensation point
        #with condensation point which is actually best hit
        l=int((cond_points_hits[j,2]-0.25)*8)
        track[0,l]=cond_points_hits[j,0]
        track[0,l+1]=cond_points_hits[j,1]
        
        if j==0:
            all_tracks=track
        else:
            all_tracks=np.vstack((all_tracks,track))

    return all_tracks

        

#calculate metrics like track efficiency, and purity for one event
#efficiency defined as percentage of true tracks that survive
#purity defined as ratio of false tracks over all predicted tracks
#arguments: true tracks and predicted tracks
#returns purity and efficiency
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
            
    eff=TP/true_tracks.shape[0]
    FP_eff=TP/(TP+FP)
    return eff, FP_eff

#test the object condensation method by generating n events
#and calculating efficiency, purity and mesuring prediciton times
#arguments: nb test events, GNet model, x/y/z size of array
#max number of hits, whether or not to print average eff,pur and times
#return: efficiency and purity averaged over nb test events.
def test_GNet(nb_events,model,width,height,depth,vmax,doPrint):

    AvEff=0
    AvPur=0
    
    AvEff_cr=0
    AvPur_cr=0

    AvTime_getEvent=0
    AvTime_getCandidates=0
    AvTime_apply=0

    for i in range(nb_events):
    
        #if (i%1000)==0:
        #    print('Generated '+str(i)+' events')

        #hard code these for now
        nbTracks=2
        resolution=0
        nbNoise=4
        allow_overlap=False

        #timing to get an event
        t0_getEvent = time.time()

        true_tracks,event_array=genEvent(depth,width,height,nbTracks,resolution,nbNoise,allow_overlap)

        t1_getEvent = time.time()
        AvTime_getEvent=AvTime_getEvent+(t1_getEvent-t0_getEvent)

        #timing to find track candidates (not requiring meaningful labels)
        t0_getCandidates = time.time()

        hits,size,truth=get_hits_GNet(event_array,width,height,depth,vmax,true_tracks)

        t1_getCandidates = time.time()
        AvTime_getCandidates=AvTime_getCandidates+(t1_getCandidates-t0_getCandidates)

        #timing to apply ID selecting only tracks with best response
        t0_apply = time.time()

        pred_tracks=apply_GNet_trackID(model,hits,size,0.1)

        t1_apply = time.time()
        AvTime_apply=AvTime_apply+(t1_apply-t0_apply)


        pred_tracks=unnorm(pred_tracks,width,height)
        eff,pur=calculate_GNet_metrics(true_tracks,pred_tracks)

        AvEff=AvEff+eff
        AvPur=AvPur+pur

    #average metrics, nb of tracks and times
    AvEff=AvEff/nb_events
    AvPur=AvPur/nb_events


    AvTime_getEvent=AvTime_getEvent/nb_events
    AvTime_getCandidates=AvTime_getCandidates/nb_events
    AvTime_apply=AvTime_apply/nb_events

    if doPrint==True:

        print('')
        print('Percentage of true tracks that survive '+str(AvEff))
        print('Fraction of true tracks in all predicted tracks '+str(AvPur))

        print('')
        print('Generating an event took on average '+str(AvTime_getEvent)+'s')
        print('Getting array of hits took on average '+str(AvTime_getCandidates)+'s')
        print('Applying the track ID took on average '+str(AvTime_apply)+'s')
        
    return AvEff,AvPur
