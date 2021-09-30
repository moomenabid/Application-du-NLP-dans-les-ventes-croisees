import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import random
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding,Input,Lambda,Reshape,Multiply,Dense
import keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

#Importing data
data=pd.read_csv("C:/Users/abidm/Desktop/for_the_drive/Algo_CAH_channel_filling-fordrive/data_family.csv", sep=",")
#data = pd.read_csv("C:/Users/user/Desktop/Projects/Channel_flling/Algo_CAH_channel_filling/data_family.csv", sep=",")
PcSpread=data.copy()

#data preprocessing -step1
PcSpread=PcSpread[['type','Customer.Num','Family','ParameterPanel','Gpch','Reportables']]
#We filter on public customers
PcSpread=PcSpread[PcSpread['type']=="Public"]
#Keeping relevant columns
PcSpread=PcSpread[['Customer.Num','Gpch','Reportables']]
#aggregate reportables for same couple(client,product)
PcSpread = PcSpread.groupby(['Customer.Num','Gpch'])['Reportables'].agg('sum')
PcSpread=PcSpread.reset_index()
#spread df
PcSpread=PcSpread.pivot(index='Customer.Num', columns='Gpch',values='Reportables')
#fillna
PcSpread=PcSpread.fillna(0)

#data preprocessing -step2
var_tst=MinMaxScaler().fit_transform(PcSpread)
var_tst=pd.DataFrame(data=var_tst,index=PcSpread.index, columns=PcSpread.columns) 

#boxplots
fig = plt.figure(figsize =(10, 7))
plt.boxplot(var_tst["TSH"])
plt.show()
                      
fig = plt.figure(figsize =(10, 7))
plt.boxplot(PcSpread["TSH"])
plt.show()

#filtering outliers (manually)
liste=[3170008566,3170403023,3170403318,3170403427
       ,3170403523,3170404157,3170404419,3170404549
       ,3170404645,3170404811,3170405264,3170405520
       ,3170405869,3170416724,3170431792,3170432227
       ,3170436708,3170441803
       ,3170403012,3170403551,3170403551,3170403831
       ,3170404057,3170404478,3170403043,3170404464]
PcSpread=PcSpread.loc[~PcSpread.index.isin(liste)]
var_tst=MinMaxScaler().fit_transform(PcSpread)
var_tst=pd.DataFrame(data=var_tst,index=PcSpread.index, columns=PcSpread.columns) 

#selecting products that have 
#high document frequency(present in more than 30 rows)
liste=[]
DF_Threshold=30
for col in PcSpread.columns.tolist():
    if len(PcSpread[(PcSpread[col]>0)][col])>=DF_Threshold:
        liste.append(col)
PcSpread=PcSpread[liste]

#getting bag of words using quantile 
#as weights in the sentence (client)
TF_max=6
for col in PcSpread.columns.tolist():
    for i in range(TF_max):
            weight=(i+1)
            qmin=(1/TF_max)*i
            qmax=(1/TF_max)*(i+1)
            vmin=PcSpread[(PcSpread[col]>0)][col].quantile(qmin)
            vmax=PcSpread[(PcSpread[col]>0)][col].quantile(qmax)
            #PcSpread2[(PcSpread2["AMP"]>=vmin) 
             #        & (PcSpread2["AMP"]<=vmax)]["AMP"]=weight
            PcSpread.loc[(PcSpread[col]>=vmin) & (PcSpread[col]<=vmax),col]=weight
            

#delete spaces in cols to make the right words in corpus later
PcSpread = PcSpread.rename(columns={col: col.replace(" ", "") for col in PcSpread.columns.tolist()})
PcSpread = PcSpread.rename(columns={col: col.replace("-", "") for col in PcSpread.columns.tolist()})
PcSpread = PcSpread.rename(columns={col: col.replace("/", "") for col in PcSpread.columns.tolist()})
PcSpread = PcSpread.rename(columns={col: col.replace("+", "") for col in PcSpread.columns.tolist()})
PcSpread.columns

#deleting sentences with very few words
sum_ax1_TF=PcSpread.sum(axis=1).to_frame()
sum_ax1_TF = sum_ax1_TF.rename(columns={0:'sum_ax1'})
TF_Threshold=20
PcSpread=PcSpread[sum_ax1_TF['sum_ax1']>=TF_Threshold]
sum_ax1_fltrd=sum_ax1_TF[sum_ax1_TF['sum_ax1']>=TF_Threshold]
len(PcSpread)
max(sum_ax1_fltrd['sum_ax1'])

#converting col type to int to be used in range later
PcSpread.dtypes.value_counts()
PcSpread = PcSpread.astype({col: int for col in PcSpread.columns.tolist()})
PcSpread.dtypes.value_counts()

#creating the corpus
sent=[]
for i, row in PcSpread.iterrows():
    one_sent=""
    for col in PcSpread.columns.tolist():
        for l in range(row[col]):
            one_sent=one_sent+" "+col
    one_sent=one_sent.strip()
    sent.append(one_sent)

#testing nbr words in a sentence is correct
var_tst=sum_ax1_fltrd.copy()
var_tst["sum_ax1"]=0
var_tst=var_tst.reset_index()

for i, row in var_tst.iterrows():
    word_list = sent[i].split() 
    number_of_words = len(word_list) 
    #var_tst[i]["sum_ax1"]=number_of_words
    var_tst.loc[i,"sum_ax1"]=number_of_words

var_tst=var_tst.set_index('Customer.Num')

sum_ax1_fltrd[sum_ax1_fltrd['sum_ax1']!=
              var_tst['sum_ax1']] #this gives empty df
                                        #=>corpus succ created

#getting targets
target_var = pd.read_csv("C:/Users/abidm/Desktop/for_the_drive/Algo_CAH_channel_filling-fordrive/target.csv", sep=",")
target_var = pd.read_csv("C:/Users/abidm/Desktop/for_the_drive/Algo_CAH_channel_filling-fordrive//target2.csv", sep=",")
target_var=target_var.set_index('Customer.Num')

#onehot
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(sent)
word2id = tokenizer.word_index#they're lower case
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
onehot_repr = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in sent]
VOC_SIZE = len(word2id)

#little test that one hot is fine
for i in range(len(sent)):
    if len(onehot_repr[i])!=len(sent[i].split()):
        print("we got a problem in i=",i)#shows nothing so fine
        
#padding+onehot
max_len=max([len(i) for i in onehot_repr])
onehot_repr=pad_sequences(onehot_repr,padding='pre',maxlen=max_len)
onehot_repr=onehot_repr.tolist()

#target create  cols "word_id" and y="onehot"
target_var['lower']=target_var['target'].apply(lambda x: x.lower() )
target_var['id']=range(len(target_var)) 
y=to_categorical(np.random.randint(VOC_SIZE,size=(len(target_var),1)), VOC_SIZE)

for i, row in target_var.iterrows():
    y[row['id']]=to_categorical(word2id[row['lower']], VOC_SIZE)

target_var['word_id']=0   
for i, row in target_var.iterrows():
    target_var.loc[i,'word_id']=word2id[row['lower']]
    #target_var.loc[i,'onehot'] = np_utils.to_categorical(row['id'], VOC_SIZE)
y_lst=y.tolist()#maybe need convert it to int since y is float
y_df = pd.DataFrame(y)
y_df=y_df.set_index(target_var.index)
#matrix used for clustering
ones_df=PcSpread.copy()
ones_df[ones_df!=0]=1

distance_matrix = euclidean_distances(ones_df)
distance_matrix=distance_matrix**2

#dendogram+creating groups
dendrogram = sch.dendrogram(sch.linkage(ones_df, method = 'ward',metric='euclidean'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

ones_df.to_csv("C:/Users/abidm/Desktop/for_the_drive/Algo_CAH_channel_filling-fordrive/ones_df.csv")
freq=ones_df.copy()
hc = AgglomerativeClustering(distance_threshold=6.75, affinity = 'euclidean', linkage = 'ward',n_clusters = None)
hc.fit(ones_df)
freq['frequency']=hc.labels_
df_freq=freq['frequency'].value_counts().to_frame()
df_freq['frequency'].values.sum()
freq=freq.rename(columns={'frequency':'groupe'})

#getting the groups to delete
del_list=[]
for groupe in freq['groupe'].unique().tolist():
    if (df_freq['frequency'][groupe]<=3):
        del_list.append(groupe)

#selecting the dataframe with the groups of frequency>=4
freq['id']=range(len(freq)) 
indexes=[freq.loc[i,'id'] for i in PcSpread.index if freq.loc[i,'groupe'] in del_list] 
ix=[i for i in PcSpread.index if freq.loc[i,'groupe'] not in del_list]  
PcSpread=PcSpread.loc[ix]
freq=freq.loc[ix]
df_freq=df_freq[df_freq['frequency']>=4]
#target_var=target_var.loc[ix]

for index in sorted(indexes, reverse=True):
    del sent[index]
    del onehot_repr[index]

#initializing freq['id']
freq['id']=range(len(freq)) 

#we need to delete els with freq<=3
trn_list=[]
val_list=[]
tst_list=[]
idx_list=[]
gr_len=0
val_len=0
tst_len=0

for groupe in freq['groupe'].unique():#getting the group
    
    #getting the group's dataframe and all the index list
    
    groupe_df=freq[freq['groupe']==groupe]
    idx_list=groupe_df.index.values.tolist()
    
    if (df_freq['frequency'][groupe]<=3):
        
        #we won't take customers with frequency less than 3
        pass
    
    elif (df_freq['frequency'][groupe]>=4) & (df_freq['frequency'][groupe]<=9):
        
        #getting new test list of customers andappendingit
        tst_len=round(len(groupe_df)/4)
        random.seed(41)
        tst_lst_jr=random.sample(idx_list, tst_len)
        tst_list=tst_list+tst_lst_jr
        idx_list = [elmnt for elmnt in idx_list if elmnt not in tst_lst_jr]
        
        #getting new validation list of customers andappendingit
        val_len=round(len(groupe_df)/4)
        random.seed(41)
        val_lst_jr=random.sample(idx_list, val_len)
        val_list=val_list+val_lst_jr
        idx_list = [elmnt for elmnt in idx_list if elmnt not in val_lst_jr]
        
        #getting new train list of customers andappendingit
        trn_list=trn_list+idx_list
    
    else:
        
        #same as the previous block but for freqs>10
        
        tst_len=round(len(groupe_df)/5)
        random.seed(41)
        tst_lst_jr=random.sample(idx_list, tst_len)
        tst_list=tst_list+tst_lst_jr
        idx_list = [elmnt for elmnt in idx_list if elmnt not in tst_lst_jr]
        
        val_len=round(len(groupe_df)/5)
        random.seed(41)
        val_lst_jr=random.sample(idx_list, val_len)
        val_list=val_list+val_lst_jr
        idx_list = [elmnt for elmnt in idx_list if elmnt not in val_lst_jr]
        
        trn_list=trn_list+idx_list

#train set,validation set, test set
x_trn=PcSpread.loc[trn_list]
x_val=PcSpread.loc[val_list]
x_tst=PcSpread.loc[tst_list] 
    
f_trn=freq.loc[trn_list]
f_val=freq.loc[val_list]
f_tst=freq.loc[tst_list]  

target_trn=target_var.loc[trn_list]
target_val=target_var.loc[val_list]
target_tst=target_var.loc[tst_list]   

y_df_trn=y_df.loc[trn_list]
y_df_val=y_df.loc[val_list]
y_df_tst=y_df.loc[tst_list]   

y_trn=y_df_trn.to_numpy()#these are floats
y_val=y_df_val.to_numpy()
y_tst=y_df_tst.to_numpy()

indexes=[freq.loc[i,'id'] for i in trn_list] 
sent_trn = [sent[i] for i in indexes]
onehot_trn=[onehot_repr[i] for i in indexes]
indexes=[freq.loc[i,'id'] for i in val_list] 
sent_val = [sent[i] for i in indexes]     
onehot_val=[onehot_repr[i] for i in indexes] 
indexes=[freq.loc[i,'id'] for i in tst_list] 
sent_tst = [sent[i] for i in indexes]
onehot_tst=[onehot_repr[i] for i in indexes]

onehot_trn = np.asarray(onehot_trn, dtype=np.int32)
onehot_val = np.asarray(onehot_val, dtype=np.int32)
onehot_tst = np.asarray(onehot_tst, dtype=np.int32)

check_prc_trn=f_trn['groupe'].value_counts().to_frame()
check_prc_val=f_val['groupe'].value_counts().to_frame()
check_prc_tst=f_tst['groupe'].value_counts().to_frame()
check_prc=pd.concat([check_prc_trn, check_prc_val,check_prc_tst], axis=1)

#model creation
def myMask(x):
    mask= K.greater(x,0) #will return boolean values
    mask= K.cast(mask, dtype=K.floatx()) 
    return mask
MAX_SEQUENCE_LENGTH=max_len
EMBEDDING_DIM=50 #10 0.37/100 0.42/50 43.7/   
VOC_SIZE=VOC_SIZE
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='dnn_input')
embedding_layer = Embedding(VOC_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, name = 'embedding_dnn')
embedded_sequences = embedding_layer(sequence_input)
y = Lambda(myMask, output_shape=(MAX_SEQUENCE_LENGTH,))(sequence_input)
y = Reshape(target_shape=(MAX_SEQUENCE_LENGTH,1))(y)
merge_layer = Multiply(name = 'masked_embedding_dnn')([embedded_sequences,y])
aggregation_layer=Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,),name ='aggregation_dnn')(merge_layer)
x=Dense(VOC_SIZE, activation='softmax',name ='hidden_dense_dnn')(aggregation_layer)
model = Model(sequence_input, x)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
VOC_SIZE

#testing model construction
print(len(model.predict(onehot_trn)[0]))

#train model
history=model.fit(onehot_trn, y_trn,batch_size=len(onehot_trn), epochs=200, verbose=1, validation_data=(onehot_val, y_val))

#params for learning curve
for k,v in history.history.items():
    print(k)
#learning curve accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#learning curve loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#testing model
score=model.evaluate(onehot_tst, y_tst)
model.metrics_names
score #we got accuracy=45.8









