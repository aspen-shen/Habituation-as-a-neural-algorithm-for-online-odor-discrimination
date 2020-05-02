import numpy as np
import pandas as pd
import random


# Normalize the input dataset to have zero min and fixed mean value
def norm(df, set_mean):
    mins = df.min()
    df = df - df.min()
    df = set_mean * df/df.mean()
    return df

# Generate a random linear transformation matrix to transform PN to KC
def generate_transformation_matrix(n_kc,n_orn,n_response):
    R = np.zeros((n_kc, n_orn))
    for i in range(n_kc):
        random.seed(i)
        R[i,random.sample(list(range(n_orn)), n_response)] = 1
    return R

# Calculate the KC tag of an odor (i.e. the index of activated KCs)
def get_KC_tag(odor,w,R,thresh):
    '''
    odor: a vector of ORN responses for a given odor
    w: inhibitory synaptic strength from LN to PN
    R: random linear transformation matrix from PN to KC
    thresh: rectlinear threshold for KC activation
    '''
    p = odor - w
    p[p<0] = 0
    KC = np.matmul(R,p)
    KC[KC<=thresh] = 0
    threshold = np.quantile(KC,0.95)
    KC[KC<threshold] = 0
    tag = np.where(KC>0)[0]
    return tag

# Returns the inhibitory synaptic strength (from LN to PN) after habituation
def habituate(odor,w,alpha,beta,steps):
    '''
    odor : a vector of ORN responses for a given odor
    w: initial synaptic strength
    alpha: habituation parameter, determines how fast the odor gets habituated (usually set at 0.05)
    beta: dis-habituation parameter, determines how fast the odor gets dis-habituated (usually set at 0.01)
    steps: time step for habituation
    '''
    for i in range(steps):
        pp = odor - w
        pp[pp<0] = 0
        w += alpha*pp - beta*w
    return w

# Calculates jaccard similarity between two odor tags
def jaccard(tag1,tag2):
    if len(tag1)>0 or len(tag2)>0:
        return 100*len(np.intersect1d(tag1, tag2))/len(np.union1d(tag1, tag2))  
    else:
        return 0

# Performs background subtraction for odor mixture
def background_subtraction(df,alpha,beta,steps,R,thresh,linear=True):
    odors = df.columns
    n_odor = len(odors)
    mb_before = []
    mb_after = []
    # loop for all odor pairs in df
    for i in range(0,n_odor-1):
        for j in range(i+1,n_odor):
            odor1 = np.array(df[odors[i]].values)
            odor2 = np.array(df[odors[j]].values)
            if linear:
            	# calculate odor mixture with a linear model
                mixture = 0.8*odor1 + 0.2*odor2
            else:
            	# calculate odor mixture with a non-linear model and re-normalize the mixture data
                mixture = (odor1)**0.8 + (odor2)**0.2
                mixture = mixture + np.abs(np.min(mixture))
                mixture = mixture*10/np.mean(mixture)
            # calculate KC tag for mixture and components before habituation
            w = np.array([0.0]*len(odor1))
            KC1 = get_KC_tag(odor1,w,R,thresh)
            KC2 = get_KC_tag(odor2,w,R,thresh)
            KC_mix_before = get_KC_tag(mixture,w,R,thresh)
            # calculate jaccard similarity between odor 2 and the mixture before habituation
            mbb = jaccard(KC2,KC_mix_before)
            mb_before.append(mbb)
			#-------------------------------------------
			#habituate on odor1
			#-------------------------------------------
            w = habituate(odor1,w,alpha,beta,steps)
            # calculate KC tag for mixture after habituation
            KC_mix = get_KC_tag(mixture,w,R,thresh)
            # calculate jaccard similarity between KC tag of mixture after habituation and KC tag of odor 2
            mba = jaccard(KC2,KC_mix)
            mb_after.append(mba)
    to_plot = pd.DataFrame({'Before':mb_before,'After':mb_after})
    return to_plot

# calculate jaccard distance between one odor and a matrix of odors
from sklearn.metrics import jaccard_score
def jaccard_distance(KCA,KCs):
    distance = [1-jaccard_score(KCA,np.array(KCs[odor].values)) for odor in KCs.columns]
    return distance

# calculates mean average precision (mAP)
def mAP(dist1,dist2):
    '''
    dist1, dist2: two lists of jaccard distance
    '''
    aps = []
    ns = [10,20,30]
    for n in ns:
        truth = np.argsort(np.array(dist1))[1:n+1]
        test = np.argsort(np.array(dist2))[1:n+1]
        weight1 = np.array([i for i in range(len(test)) if test[i] in truth])
        weight1 = weight1 + 1
        rank1 = np.array(range(len(weight1))) + 1
        if len(weight1) == 0:
            ap1 = 0
        else:
            ap1 = np.mean(rank1/weight1)
        aps.append(ap1)
    return np.mean(aps)

# returns a list of digits 0, 1 the same length as KCs
# if the value is 1 at a certain position, then the corresponding KC is activated by this odor
def get_KC_binary(odor,w,R,thresh):
	p = odor - w
	p[p<0] = 0
	p = np.array(p)
	KC = np.matmul(R,p)
	KC[KC<=thresh] = 0
	threshold = np.quantile(KC,0.95)
	KC[KC<threshold] = 0
	KC[KC>=threshold] = 1
	return KC

# for odor A and odor B, calculate mAP between mixture(A, B) and odor A, and mixture(A, B) and odor B. 
# mixture contains 80% odor A and 20% odor B
# if hab=True, mAPs are calcuated after habituation to odor A; 
# if linear=True, the mixture is calcuated with a linear model
def get_nn(odorA,odorB,normed,KCs,R,thresh,alpha,beta,steps,hab=False,linear=True):
    A = np.array(normed[odorA].values)
    B = np.array(normed[odorB].values)
    if linear:
        mix = A*0.8 + B*0.2
    else:
        mix = A**0.8 + B**0.2
        mix = mix - np.min(mix)
        mix = mix*10/np.mean(mix)
    normed['mixture'] = mix
    w = np.array([0.0]*len(A))
    if hab:
        w = habituate(A,w,alpha,beta,steps)
    mixture = get_KC_binary(mix,w,R,thresh)
    KCs['mixture'] = mixture
    KCA = np.array(KCs[odorA].values)
    KCB = np.array(KCs[odorB].values)
    KC_mix = np.array(KCs['mixture'].values)
    Ann = jaccard_distance(KCA,KCs)
    Bnn = jaccard_distance(KCB,KCs)
    mix_nn = jaccard_distance(KC_mix,KCs)
    normed.drop(['mixture'],axis=1,inplace=True)
    return (mAP(Ann,mix_nn),mAP(Bnn,mix_nn))


# batch processing to get mAP for nearest neighbors
def batch_nn(df,KCs,R,thresh,alpha,beta,steps,linear=True):
    A_sim_before = []
    A_sim_after = []
    B_sim_before = []
    B_sim_after = []
    odors = df.columns
    for i in range(len(odors)-1):
        for j in range(i+1,len(odors)):
            odorA = odors[i]
            odorB = odors[j]
            a_before, b_before = get_nn(odorA,odorB,df,KCs,R,thresh,alpha,beta,steps,hab=False,linear=linear)
            a_after, b_after = get_nn(odorA,odorB,df,KCs,R,thresh,alpha,beta,steps,hab=True,linear=linear)
            A_sim_before.append(a_before)
            A_sim_after.append(a_after)
            B_sim_before.append(b_before)
            B_sim_after.append(b_after)
    out = pd.DataFrame({'A_sim_before':A_sim_before,'B_sim_before':B_sim_before,'A_sim_after':A_sim_after,'B_sim_after':B_sim_after})
    return out