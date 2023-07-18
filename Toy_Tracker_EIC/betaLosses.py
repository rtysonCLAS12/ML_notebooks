"""
Modified version from
https://github.com/jkiesele/SOR/blob/master/modules/betaLosses.py
"""

import tensorflow as tf
import tensorflow.keras as keras

K = keras.backend


#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)

def create_pixel_loss_dict(truth, pred):
    '''
    input features as
    B x P x P x F
    with F = [x,y,z]
    
    truth as 
    B x P x P x T
    with T = [mask, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    '''
    
    #print('*** pred shape '+str(pred.shape))
    #print('*** truth shape '+str(truth.shape))
    
    outdict={}
    #truth = tf.Print(truth,[tf.shape(truth),tf.shape(pred)],'truth, pred ',summarize=30)
    
    #don't think i need to reshape here (to delete if right)
    #def resh(lastdim):
    #    return (tf.shape(pred)[0],tf.shape(pred)[1]*tf.shape(pred)[2],lastdim)

    #make it all lists
    #outdict['p_beta']   =  tf.reshape(pred[:,:,0:1], resh(1))
    #outdict['p_ccoords'] = tf.reshape(pred[:,:,1:3], resh(2))
    #outdict['t_mask'] =  tf.reshape(truth[:,:,0:1], resh(1)) 
    #outdict['t_objidx']= tf.reshape(truth[:,:,1:3], resh(1))
    
    #make it all lists
    outdict['p_beta']   =  pred[:,:,0:1]
    outdict['p_ccoords'] = pred[:,:,1:3]
    outdict['t_mask'] =  truth[:,:,0:1]
    outdict['t_objidx']= truth[:,:,1:2]
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.math.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    #outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    outdict['n_noise'] =  tf.cast(tf.shape(outdict['t_mask'])[1], dtype='float32') -outdict['n_nonoise']
    outdict['n_total'] = outdict['n_noise']+outdict['n_nonoise']
    
    
    return outdict

def calculate_charge(beta, q_min):
    beta = tf.clip_by_value(beta,0,1-K.epsilon()) #don't let gradient go to nan
    return tf.atanh(beta)+q_min


def sub_object_condensation_loss(d,q_min,Ntotal=4096):
    
    q = calculate_charge(d['p_beta'],q_min)
    
    L_att = tf.zeros_like(q[:,0,0])
    L_rep = tf.zeros_like(q[:,0,0])
    L_beta = tf.zeros_like(q[:,0,0])
    
    Nobj = tf.zeros_like(q[:,0,0])
    
    isobj=[]
    alpha=[]
    
    #for some reason this doesn't work
    #max_obj=tf.argmax(d['t_objidx'])+1
    #print('max number of objects',max_obj)
    
    for k in range(2):#maximum number of objects
        
        Mki      = tf.where(tf.abs(d['t_objidx']-float(k))<0.2, tf.zeros_like(q)+1, tf.zeros_like(q))
        
        #print('Mki',Mki.shape)
        
        iobj_k   = tf.reduce_max(Mki, axis=1) # B x 1
        
        
        Nobj += tf.squeeze(iobj_k,axis=1)
        
        
        kalpha   = tf.argmax(Mki*d['t_mask']*d['p_beta'], axis=1)
        
        isobj.append(iobj_k)
        alpha.append(kalpha)
        
        #print('kalpha',kalpha.shape)
        
        x_kalpha = tf.gather_nd(d['p_ccoords'],kalpha,batch_dims=1)
        x_kalpha = tf.expand_dims(x_kalpha, axis=1)
        
        #print('x_kalpha',x_kalpha.shape)
        
        q_kalpha = tf.gather_nd(q,kalpha,batch_dims=1)
        q_kalpha = tf.expand_dims(q_kalpha, axis=1)
        
        distance  = tf.sqrt(tf.reduce_sum( (x_kalpha-d['p_ccoords'])**2, axis=-1 , keepdims=True)+K.epsilon()) #B x V x 1
        F_att     = q_kalpha * q * distance**2 * Mki
        F_rep     = q_kalpha * q * tf.nn.relu(1. - distance) * (1. - Mki)
        
        L_att  += tf.squeeze(iobj_k * tf.reduce_sum(F_att, axis=1), axis=1)/(Ntotal)
        L_rep  += tf.squeeze(iobj_k * tf.reduce_sum(F_rep, axis=1), axis=1)/(Ntotal)
        
        
        beta_kalpha = tf.gather_nd(d['p_beta'],kalpha,batch_dims=1)
        L_beta += tf.squeeze(iobj_k * (1-beta_kalpha),axis=1)
        
        
    L_beta/=Nobj
    #L_att/=Nobj
    #L_rep/=Nobj
    
    L_suppnoise = tf.squeeze(tf.reduce_sum( (1.-d['t_mask'])*d['p_beta'] , axis=1) / (d['n_noise'] + K.epsilon()), axis=1)
    
    reploss = tf.reduce_mean(L_rep)
    attloss = tf.reduce_mean(L_att)
    betaloss = tf.reduce_mean(L_beta)
    supress_noise_loss = tf.reduce_mean(L_suppnoise)
    
    return reploss, attloss, betaloss, supress_noise_loss, Nobj, isobj, alpha
 

def object_condensation_loss(truth,pred):
    d = create_pixel_loss_dict(truth,pred)
    
    reploss, attloss, betaloss, supress_noise_loss, Nobj, isobj, alpha = sub_object_condensation_loss(d,q_min=0.1,Ntotal=d['n_total'])

    
    #payload_scaling = calculate_charge(d['p_beta'],0.1)
    
    #better purity, worse efficiency
    #loss = attloss + 0.5*reploss + 0.1*(supress_noise_loss+betaloss) 
    #good efficiency, worse purity
    loss = attloss + 0.5*(reploss + betaloss ) + 0.1*supress_noise_loss
    
    #loss = tf.Print(loss,[loss,
    #                          reploss,
    #                          attloss,
    #                          betaloss,
    #                          supress_noise_loss
    #                          ],
    #                          'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss' )
    return loss
    


