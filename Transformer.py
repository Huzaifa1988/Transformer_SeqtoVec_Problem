#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy               as np
import pandas              as pd
import h5py                as h5py
import matplotlib.pyplot   as plt
import torch.nn            as nn
import torch.nn.functional as F
import pickle
import os.path
import torch
import math,copy,re
import warnings
import seaborn as sns

from   IPython.display import clear_output
from   numpy.random    import seed
from   sklearn.utils   import shuffle
####from aipolymer import *
from torch.utils.data import DataLoader, Dataset
from scipy.special import factorial

##%matplotlib inline

#%matplotlib inline
np.set_printoptions(precision = 3)
plt.rcParams["figure.figsize"] = (2,3)
plt.rcParams["figure.dpi"]     = 100

dev = torch.device("cpu")
#dev    = devCPU

warnings.simplefilter("ignore")
print(torch.__version__)


# In[2]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        #print(out)
        return out


# In[3]:


class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x


# In[4]:


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=64, n_heads=2):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output


# In[5]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=2, n_heads=2):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention   = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        attention_out = self.attention(key,query,value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out



class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()
        
        #self.embedding_layer = Embedding(vocab_size, embed_dim)
        #self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        
        self.num_layers  = 3
        self.hidden_size = embed_dim

        self.gru = nn.GRU(1, self.hidden_size, self.num_layers, bias=False, batch_first=True)
        ## x-> (batchsize, seq, input_size)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        #embed_out = self.embedding_layer(x)
        #out = self.positional_encoder(embed_out)
        out, h_n = self.gru(x, h0)
        
        for layer in self.layers:
            out = layer(out,out,out)

        return out  #32x10x512


# In[6]:


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length,num_layers=2, expansion_factor=2, n_heads=2):
        super(Transformer, self).__init__()
        
        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        
        #self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        #self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.fc = nn.Linear(seq_length*embed_dim, target_vocab_size, bias=False)
        
        
    
    def forward(self, src):
        """
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        #trg_mask = self.make_trg_mask(trg)
        #print(type(src))
        #print(src)
        enc_out = self.encoder(src)
        D1 = enc_out.shape
        outputs = self.fc(enc_out.reshape(D1[0],D1[1]*D1[2]))
   
        #outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs


# In[7]:


src_vocab_size = 4
target_vocab_size = 20
num_layers = 2
seq_length= 12


# let 0 be sos token and 1 be eos token
src = torch.tensor([[0, 2, 3, 2, 3, 3, 2, 3, 2, 2, 2, 1], 
                    [0, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 1]])
target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1], 
                       [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])

print(src.shape,target.shape)
model = Transformer(embed_dim=64, src_vocab_size=src_vocab_size, 
                    target_vocab_size=target_vocab_size, seq_length=seq_length,
                    num_layers=num_layers, expansion_factor=2, n_heads=2)
model


# In[8]:


def scale_sequence(x,scale):
    return scale*(1.-2.*x)
    #return scale*(2+x)

def truncate_normalize_free_energy(y,window):
    numz    = y.shape[1]
    print(numz)
    zmin    = numz//2-window
    zmax    = numz//2+window
    y       = y[:,zmin:zmax]
    norm    = np.max(np.abs(y))
    print ("F-norm = ",norm)
    norm = 1.0
    #ynorm   = y / norm
    ynorm = y
    return ynorm,norm

def preprocess(x,y,sequenceScale,window):
    x       = scale_sequence(x,sequenceScale)
    y, norm = truncate_normalize_free_energy(y,window)
    return x,y,norm

def read_hdf5(filename,N,w,key="pz"):
    """Read a hdf5 file from supplementary data files, 
       and return sequence (x) and free energy profile (y) data sets."""
    file    = h5py.File(filename, 'r')
    
    #rootKey = "w30/N"+str(N)+"/set0"
    rootKey = "w"+str(w)+"/N"+str(N)+"/set0"
    print(rootKey)
    
    s       = file[rootKey + "/sequence"][:]
    
    # Distribution data "p(z)" is stored in y. Shape is (n_data, sampling box size in z direction).
    pz      = file[rootKey + "/" + key][:]
    
    # Convert p(z) into a free energy profile assuming that the bulk value is normalized to 1.
    # The profile p(z) is the Rosenbluth-based estimate proportional to Omega_z in eq. S1.
    nz=pz.shape[1]
    print(nz)
    
    wherenot0 = np.sum((pz>0.),axis=1).astype(int)==pz.shape[1]
    s=s[wherenot0]
    pz=pz[wherenot0]
    mean=0.5*(pz[:,4]+pz[:,-4])
    ##mean=0.5*(pz[:,4]+pz[:,36])
    F = -np.log(pz/mean[:,None])
    return s,F

def Extract_fingerprint(seq):
        
        x_np = seq
        N    = (x_np*x_np).sum ( 1 )[:,None]
        Seq_A_count = ((x_np>0.5)*1.)
        Seq_A_count = np.sum(Seq_A_count, axis=1)   
        Seq_A_count = Seq_A_count.reshape(Seq_A_count.shape[0],1)
        
        Seq_B_count = N - Seq_A_count
        
        fact_A = factorial(Seq_A_count)
        fact_B = factorial(Seq_B_count)
        fact_N = factorial(N)
        
        pacc = (fact_A*fact_B)/fact_N
        
        #print( x_np[:5], N[:5], Seq_A_count[:5], Seq_B_count[:5] )
        
        change      = np.zeros((x_np.shape[0],1), dtype=np.float64)
        Selected    = np.zeros((x_np.shape[0],1), dtype=np.int8)
        
        for i in range(x_np.shape[0]):
            for j in range(int(N[i])-1):
                change[i] += ( np.absolute((x_np[i,j] - x_np[i,j+1]) ) )
        
            change[i] /= 2.0
            
        change = change.reshape(x_np.shape[0],1)
        Newx   = np.concatenate(( Seq_A_count, Seq_B_count, change, N, fact_A, fact_B, fact_N, pacc, Selected), axis=1)
        #New_X  = torch.from_numpy(Newx).float()
        i+=1
        
        return Newx


# In[9]:


def Shuf_train_test(X, Y, Seq, NN, Glob_Seq, w):
	
	X1_w    = X [:Seq][:]
	Y1_w    = Y [:Seq][:]
	#FP_X1_w = Extract_fingerprint(X1_w)

	X1_w_T    = X [Seq:][:]
	Y1_w_T    = Y [Seq:][:]
	#FP_X1_w_T = Extract_fingerprint(X1_w_T)
	
	#Z       = np.ones((X.shape[0], 1), dtype=np.float64, )*w
	#Z1_w    = Z [:Seq][:]
	#Z1_w_T  = Z [Seq:][:]
	
	#print (X1_w.shape,   Y1_w.shape)
	#print (X1_w_T.shape, Y1_w_T.shape)
	#print (Z1_w.shape,   Z1_w_T.shape)
	
	Cur_Seq   = NN
	halflen   = (Glob_Seq-Cur_Seq)
	Empty_20  = np.zeros((X1_w.shape[0], halflen), dtype=np.int8)
	#X1_w     = np.concatenate(( Empty_20, X1_w, FP_X1_w, Z1_w ), axis=1)
	X1_w      = np.concatenate(( Empty_20, X1_w,), axis=1)

	halflen     = (Glob_Seq-Cur_Seq)
	Empty_20    = np.zeros((X1_w_T.shape[0], halflen), dtype=np.int8)
	#X1_w_T      = np.concatenate(( Empty_20, X1_w_T, FP_X1_w_T, Z1_w_T), axis=1)
	X1_w_T      = np.concatenate(( Empty_20, X1_w_T,), axis=1)
	
	return X1_w, Y1_w, X1_w_T, Y1_w_T 	


# In[10]:


def Seq_Train_Test(FP_X, Seq):
    i     = 0
    Idx   = [] 
    x     = 0

    while i <  1:

        pacc = FP_X[x,-2]
        r    = np.random.uniform(0, 1, 1)

        if (pacc > r):

            Idx.append(x)
            FP_X [x,-1] = 1
            print (pacc , r, i, x)
            i += 1

        if (x == int(FP_X.shape[0]-1) ):
            x = 0
            
        x +=1

    i = 1
    while i <  Seq:

        pacc = FP_X[x,-2]
        r    = np.random.uniform(0, 1, 1)

        if (pacc > r):

            if (int(FP_X [x,-1]) == 0):
                Idx.append(x)
                FP_X[x,-1] = 1
                print (pacc , r, i, x)
                i += 1

        if (x == int(FP_X.shape[0]-1) ):
            x = 0

        x +=1

    print(Idx)
    
    return Idx, FP_X


# In[4]:


def Uniform_train_test(X, Y, Seq, NN, Glob_Seq, w):
    
    #X1_w    = X [:Seq][:]
    #Y1_w    = Y [:Seq][:]
    FP_X1_w = Extract_fingerprint(X)
    Idx_, update_FP_X_ = Seq_Train_Test(FP_X1_w, Seq)
    
    print((update_FP_X_[:,-1] > 0))

    X1_Tr = X [(update_FP_X_[:,-1] > 0)]
    Y1_Tr = Y [(update_FP_X_[:,-1] > 0)]
    FP_Tr = update_FP_X_[(update_FP_X_[:,-1] > 0)]

    print(X1_Tr.shape, Y1_Tr.shape)


    X1_Te = X [(update_FP_X_[:,-1] < 1)]
    Y1_Te = Y [(update_FP_X_[:,-1] < 1)]
    FP_Te = update_FP_X_[((update_FP_X_[:,-1] < 1))]

    print(X1_Te.shape, Y1_Te.shape )
    
    

    #X1_w_T    = X [Seq:][:]
    #Y1_w_T    = Y [Seq:][:]
    #FP_X1_w_T = Extract_fingerprint(X1_w_T)
    
    #Z       = np.ones((X.shape[0], 1), dtype=np.float64, )*w
    #Z1_w    = Z [:Seq][:]
    #Z1_w_T  = Z [Seq:][:]
    
    #print (X1_w.shape,   Y1_w.shape)
    #print (X1_w_T.shape, Y1_w_T.shape)
    #print (Z1_w.shape,   Z1_w_T.shape)
    
    Cur_Seq   = NN
    halflen   = (Glob_Seq-Cur_Seq)
    Empty_20  = np.zeros((X1_Tr.shape[0], halflen), dtype=float)
    #X1_w      = np.concatenate(( Empty_20, X1_w, FP_X1_w, Z1_w ), axis=1)
    X1_Tr      = np.concatenate(( Empty_20, X1_Tr, FP_Tr[:,:4] ), axis=1)

    halflen     = (Glob_Seq-Cur_Seq)
    Empty_20    = np.zeros((X1_Te.shape[0], halflen), dtype=float)
    #X1_w_T      = np.concatenate(( Empty_20, X1_w_T, FP_X1_w_T, Z1_w_T), axis=1)
    X1_Te      = np.concatenate(( Empty_20, X1_Te, FP_Te[:,:4]), axis=1)
    
    return X1_Tr, Y1_Tr, X1_Te, Y1_Te 


# In[11]:


window           = 10 
sequenceScale    = 1
Index_bulk       = 15
Glob_Seq         = 32
Seq              = 250
w                = 6


# In[12]:


Seq              = 250
NN               = 8
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_8, Y1_8, Norm_8 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_8.shape, Y1_8.shape)
print (X1_8[:5])
print (Y1_8[:5])
print (Norm_8)

X1_8, Y1_8, X1_8T, Y1_8T = Uniform_train_test(X1_8, Y1_8, Seq, NN, Glob_Seq, w)
print ( X1_8.shape, Y1_8.shape, X1_8T.shape, Y1_8T.shape,  )


# In[13]:


Seq              = 250
NN               = 10
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_10, Y1_10, Norm_10 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_10.shape, Y1_10.shape)
print (X1_10[:5], Y1_10[:5])
print (Norm_10)

X1_10, Y1_10, X1_10T, Y1_10T = Uniform_train_test(X1_10, Y1_10, Seq, NN, Glob_Seq, w)
print ( X1_10.shape, Y1_10.shape, X1_10T.shape, Y1_10T.shape,  )


# In[14]:


Seq              = 250
NN               = 12
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_12, Y1_12, Norm_12 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_12.shape, Y1_12.shape)
print (X1_12[:5], Y1_12[:5])
print (Norm_12)

X1_12, Y1_12, X1_12T, Y1_12T = Uniform_train_test(X1_12, Y1_12, Seq, NN, Glob_Seq, w)
print ( X1_12.shape, Y1_12.shape, X1_12T.shape, Y1_12T.shape,  )


# In[15]:


Seq              = 250
NN               = 14
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_14, Y1_14, Norm_14 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_14.shape, Y1_14.shape)
print (X1_14[:5], Y1_14[:5])
print (Norm_14)

X1_14, Y1_14, X1_14T, Y1_14T = Uniform_train_test(X1_14, Y1_14, Seq, NN, Glob_Seq, w)
print ( X1_14.shape, Y1_14.shape, X1_14T.shape, Y1_14T.shape,  )


# In[16]:


Seq              = 1
NN               = 16
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_16, Y1_16, Norm_16 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_16.shape, Y1_16.shape)
print (X1_16[:5], Y1_16[:5])
print (Norm_16)

X1_16, Y1_16, X1_16T, Y1_16T = Uniform_train_test(X1_16, Y1_16, Seq, NN, Glob_Seq, w)
print ( X1_16.shape, Y1_16.shape, X1_16T.shape, Y1_16T.shape,  )


# In[17]:


Seq              = 1
NN               = 18
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_18, Y1_18, Norm_18 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_18.shape, Y1_18.shape)
print (X1_18[:5], Y1_18[:5])
print (Norm_18)

X1_18, Y1_18, X1_18T, Y1_18T = Uniform_train_test(X1_18, Y1_18, Seq, NN, Glob_Seq, w)
print ( X1_18.shape, Y1_18.shape, X1_18T.shape, Y1_18T.shape,  )


# In[18]:


Seq              = 250
NN               = 20
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_20, Y1_20, Norm_20 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_20.shape, Y1_20.shape)
print (X1_20[:5], Y1_20[:5])
print (Norm_20)

X1_20, Y1_20, X1_20T, Y1_20T = Uniform_train_test(X1_20, Y1_20, Seq, NN, Glob_Seq, w)
print ( X1_20.shape, Y1_20.shape, X1_20T.shape, Y1_20T.shape,  )


# In[19]:


Seq              = 1
NN               = 22
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_22, Y1_22, Norm_22 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_22.shape, Y1_22.shape)
print (X1_22[:5], Y1_22[:5])
print (Norm_22)

X1_22, Y1_22, X1_22T, Y1_22T = Uniform_train_test(X1_22, Y1_22, Seq, NN, Glob_Seq, w)
print ( X1_22.shape, Y1_22.shape, X1_22T.shape, Y1_22T.shape,  )


# In[20]:


Seq              = 1
NN               = 24
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_24, Y1_24, Norm_24 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_24.shape, Y1_24.shape)
print (X1_24[:5], Y1_24[:5])
print (Norm_24)

X1_24, Y1_24, X1_24T, Y1_24T = Uniform_train_test(X1_24, Y1_24, Seq, NN, Glob_Seq, w)
print ( X1_24.shape, Y1_24.shape, X1_24T.shape, Y1_24T.shape,  )


# In[21]:


Seq              = 1
NN               = 26
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_26, Y1_26, Norm_26 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_26.shape, Y1_26.shape)
print (X1_26[:5], Y1_26[:5])
print (Norm_26)

X1_26, Y1_26, X1_26T, Y1_26T = Uniform_train_test(X1_26, Y1_26, Seq, NN, Glob_Seq, w)
print ( X1_26.shape, Y1_26.shape, X1_26T.shape, Y1_26T.shape,  )


# In[22]:


Seq              = 1
NN               = 28
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_28, Y1_28, Norm_28 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_28.shape, Y1_28.shape)
print (X1_28[:5], Y1_28[:5])
print (Norm_28)

X1_28, Y1_28, X1_28T, Y1_28T = Uniform_train_test(X1_28, Y1_28, Seq, NN, Glob_Seq, w)
print ( X1_28.shape, Y1_28.shape, X1_28T.shape, Y1_28T.shape,  )


# In[23]:


Seq              = 1
NN               = 30
X1f_20, Y1f_20 = read_hdf5("../data/all.hdf5", NN, w)

shuffleIdx     = shuffle(np.arange(X1f_20.shape[0]))
X1f_20         = X1f_20[shuffleIdx]
Y1f_20         = Y1f_20[shuffleIdx]

X1_30, Y1_30, Norm_30 = preprocess(X1f_20, Y1f_20 , sequenceScale, window)
print (X1_30.shape, Y1_30.shape)
print (X1_30[:5], Y1_30[:5])
print (Norm_30)

X1_30, Y1_30, X1_30T, Y1_30T = Uniform_train_test(X1_30, Y1_30, Seq, NN, Glob_Seq, w)
print ( X1_30.shape, Y1_30.shape, X1_30T.shape, Y1_30T.shape,  )


# In[26]:


print("Train data set")

X1_Train = np.concatenate(( X1_8,  X1_10, X1_12, X1_14, 
                            X1_16, X1_18, X1_20, X1_22,
                            X1_24, X1_26, X1_28, X1_30,), axis=0)

#X1_Train = torch.tensor(X1,device=dev).double()
#X1_Train = torch.from_numpy(X1).float()
print(X1_Train.shape)
print(type(X1_Train.shape))


Y1_Train = np.concatenate(( Y1_8,  Y1_10, Y1_12, Y1_14,
                            Y1_16, Y1_18, Y1_20, Y1_22,
                            Y1_24, Y1_26, Y1_28, Y1_30,), axis=0)

###Y1_Train, Y1norm = truncate_normalize_free_energy(Y1f,window)
print(Y1_Train.shape)
#Y1_Train = torch.from_numpy(Y1).float()
print(type(Y1_Train))


print("Test data set")

X1_T = np.concatenate(( X1_8T, X1_10T,  X1_12T, X1_14T, 
                        X1_16T, X1_18T, X1_20T, X1_22T,
                        X1_24T, X1_26T, X1_28T, X1_30T,), axis=0)

print(X1_8T.shape, X1_10T.shape, X1_12T.shape, X1_14T.shape, X1_16T.shape, X1_18T.shape, X1_20T.shape )

#X1_Train = torch.tensor(X1,device=dev).double()
X1_Test = np.array(X1_T)
print(X1_Test.shape)
print(type(X1_Test.shape))


Y1_T = np.concatenate((Y1_8T,  Y1_10T, Y1_12T, Y1_14T,
                       Y1_16T, Y1_18T, Y1_20T, Y1_22T,
                       Y1_24T, Y1_26T, Y1_28T, Y1_30T,), axis=0)

#X1_Train = torch.tensor(X1,device=dev).double()
Y1_Test = np.array(Y1_T)
print(Y1_Test.shape)
print(type(Y1_Test.shape))


# In[27]:


X1_Train = torch.from_numpy(X1_Train).float()
Y1_Train = torch.from_numpy(Y1_Train).float()

X1_Test = torch.from_numpy(X1_T[::4,:]).float()
Y1_Test = torch.from_numpy(Y1_T[::4,:]).float()


print(X1_Train.shape)
print(Y1_Train.shape)

print(X1_Test.shape)
print(Y1_Test.shape)


X1_TEST_10 = torch.from_numpy(X1_10T).float()
Y1_TEST_10 = torch.from_numpy(Y1_10T).float()

X1_TEST_12 = torch.from_numpy(X1_12T).float()
Y1_TEST_12 = torch.from_numpy(Y1_12T).float()

X1_TEST_14 = torch.from_numpy(X1_14T).float()
Y1_TEST_14 = torch.from_numpy(Y1_14T).float()

X1_TEST_16 = torch.from_numpy(X1_16T).float()
Y1_TEST_16 = torch.from_numpy(Y1_16T).float()

X1_TEST_18 = torch.from_numpy(X1_18T).float()
Y1_TEST_18 = torch.from_numpy(Y1_18T).float()

X1_TEST_20 = torch.from_numpy(X1_20T).float()
Y1_TEST_20 = torch.from_numpy(Y1_20T).float()

X1_TEST_22 = torch.from_numpy(X1_22T).float()
Y1_TEST_22 = torch.from_numpy(Y1_22T).float()

X1_TEST_24 = torch.from_numpy(X1_24T).float()
Y1_TEST_24 = torch.from_numpy(Y1_24T).float()

X1_TEST_26 = torch.from_numpy(X1_26T).float()
Y1_TEST_26 = torch.from_numpy(Y1_26T).float()

X1_TEST_28 = torch.from_numpy(X1_28T).float()
Y1_TEST_28 = torch.from_numpy(Y1_28T).float()

X1_TEST_30 = torch.from_numpy(X1_30T).float()
Y1_TEST_30 = torch.from_numpy(Y1_30T).float()


# In[28]:


class XY_Data(Dataset):
    def __init__(self, X, Y):
        #data loading
        self.input   = X
        self.output  = Y
        self.n_samples = X.shape[0]
        
    def __getitem__(self, index):
        # dataset[0]
        return self.input[index], self.output[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
TrainDataset = XY_Data(X1_Train, Y1_Train)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=TrainDataset, batch_size=batch_size, shuffle=True)


# In[29]:


# Hyper-parameters 
#hidden_size   = 128
#num_layers    = 4
# num_classes nothing but target Sequence size
#num_classes   = Y1_Train.shape[1]
num_epochs    = 701
learning_rate = 0.0005
l2 = 1e-5
#input_size = 1
#sequence_length = X1_Train.shape[1]
device = dev

src_vocab_size = 4
target_vocab_size = Y1_Train.shape[1]
num_layers = 3
seq_length= X1_Train.shape[1]
input_size = 1


# In[30]:


model = Transformer(embed_dim=64, src_vocab_size=src_vocab_size, 
                    target_vocab_size=target_vocab_size, seq_length=seq_length,
                    num_layers=num_layers, expansion_factor=2, n_heads=2)
model


# In[31]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = l2)

xTrain = X1_Train
yTrain = Y1_Train

xTrain   = xTrain.reshape(-1, seq_length, input_size).to(device)
out = model(xTrain)
out.shape
print(out)


# In[ ]:


mseHistory_GRU = list() # loss history

for epoch in range(num_epochs):
    
    
    with torch.no_grad():
            if ( epoch % 10 == 0 ):
                
                print ("Train")
                xTrain   = xTrain.reshape(-1, seq_length, input_size).to(device)
                #xTrain = xTrain.to(device)
                #yTrain = yTrain.to(device)
                #print(xTrain[:5])
                yPred    = model(xTrain)
                #print(yPred.shape)
                M        = yTrain.shape[0]*yTrain.shape[1]
                mseTrain = (yPred - yTrain).pow(2).sum()/M
                
                print("10")
                
                xTest      = X1_TEST_10
                yTest      = Y1_TEST_10
                xTest      = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred      = model(xTest)
                M          = yTest.shape[0]*yTest.shape[1]
                mseTest_10 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                print("12")
                xTest     = X1_TEST_12
                yTest     = Y1_TEST_12
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_12 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                print("14")
                xTest     = X1_TEST_14
                yTest     = Y1_TEST_14
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_14 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_16
                yTest     = Y1_TEST_16
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_16 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_18
                yTest     = Y1_TEST_18
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_18 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_20
                yTest     = Y1_TEST_20
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_20 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_22
                yTest     = Y1_TEST_22
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_22 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_24
                yTest     = Y1_TEST_24
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_24 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_26
                yTest     = Y1_TEST_26
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_26 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_28
                yTest     = Y1_TEST_28
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_28 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_TEST_30
                yTest     = Y1_TEST_30
                xTest     = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_30 = (yPred -  yTest).pow(2).sum()/M # mean square

                xTest     = X1_Test
                yTest     = Y1_Test
                xTest      = xTest.reshape(-1, seq_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest = (yPred -  yTest).pow(2).sum()/M # mean square
                
                mseRecord= np.array ( (epoch, float(mseTrain), float(mseTest_10), float(mseTest_12), float(mseTest_14), float(mseTest_16), float(mseTest_18), float(mseTest_20), float(mseTest_22), float(mseTest_24), float(mseTest_26), float(mseTest_28), float(mseTest_30), float(mseTest),) )

                #mseRecord= np.array ( (epoch, float(mseTrain), float(mseTest),) )
                
                print ( "rmse/kT ~", mseRecord[0], np.sqrt(mseRecord[1:] ))
                mseHistory_GRU.append(mseRecord)
                
                if ( epoch% 100 == 0 ):
                    
                    print(xTest.shape)
                    print(yTest.shape)
                    print(yPred.shape)
                
                    np_yPred = yPred.cpu().detach().numpy()
                    yPred_DF =pd.DataFrame(np_yPred)
                
                    np_yTest = yTest.cpu().detach().numpy()
                    yTest_DF =pd.DataFrame(np_yTest)
                    
                    xTest = xTest.squeeze(2)
                    
                    np_xTest = xTest.cpu().detach().numpy()
                    xTest_DF =pd.DataFrame(np_xTest)
                    
                    
                    fname = './GRU_N20_s250_Data_with_attention.hdf5'
                    path  = '/GRU/N20_s250/'
                    
                    yPred_DF.to_hdf( fname, path+'Pred'+str(epoch),mode='a')
                
                    yTest_DF.to_hdf( fname, path+'Target'+str(epoch),mode='a')
                    
                    xTest_DF.to_hdf( fname, path+'Sequence'+str(epoch),mode='a')
                    
                
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, seq_length=24]
        # resized: [N,seq_length=1,inputsize=24]
        images = images.reshape(-1, seq_length, input_size)
        #print(images.shape)
        labels = labels.to(device)
        #print("Train loop")
        #print(images.shape)
        #print(labels.shape)
        
        #images = images.double()
        
        # Forward pass
        outputs = model(images)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (i+1) % 100 == 0:
            #print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
DF_msehist =  pd.DataFrame(mseHistory_GRU)
DF_msehist.to_hdf('./GRU_N20_s250_mse_with_attention.hdf5','/GRU/N20_s250/mse/',mode='a')


# In[ ]:




