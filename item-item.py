import math
import numpy as np
import pandas as pd
import random

#path where the full data set resides
path='C:/Users/Scorpio/Desktop/ML project/ml-100k/u.data'
path1='C:/Users/Scorpio/Desktop/ML project/ml-100k/u5.base'
path2='C:/Users/Scorpio/Desktop/ML project/ml-100k/u5.test'


data=pd.read_csv(path,sep="\t",header=None,
                 names=['userId','itemId','rating','timestamp'])




table=pd.pivot_table(data, values='rating',
                                    index=['userId'], columns=['itemId'])
print(" First few rows of Original useritem matrix without splitting into test and train\n")
print(table.head())
print(len(table.index))

print("------------------------------------------------------------------------------")



#path where the training data set resides



data=pd.read_csv(path1,sep="\t",header=None,names=['userId','itemId','rating','timestamp'])


#converting the data into useritem matrix form where cells represent the ratings.

table=pd.pivot_table(data, values='rating',index=['userId'], columns=['itemId'])
print(table.head())
print(len(table.index))

ratingmatrix=pd.pivot_table(data, values='rating',
                                   index=['itemId'], columns=['userId'])

print()
print(ratingmatrix.head()) 

from scipy.spatial.distance import correlation,cosine
'''
def similarity(user1,user2):
    user1=np.array(user1)-np.nanmean(user1)  
    user2=np.array(user2)-np.nanmean(user2)
    
    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
    
    if len(commonItemIds)==0:
         
        return 0
    else:
        user1=np.array([user1[i] for i in commonItemIds])
        user2=np.array([user2[i] for i in commonItemIds])
        return correlation(user1,user2)
    
'''

def rmse(testuseritemmatrix,full):     
    count=0
    rmse_sum=0
    for i in range(Rows):

            for j in range(Cols):

                if testuseritemmatrix[i][j] != -1 and testuseritemmatrix[i][j] != 0:
                    
                    

                    

                    rmse_sum= rmse_sum+ (testuseritemmatrix[i][j] - full[i][j]) * (testuseritemmatrix[i][j] - full[i][j] )

                    count=count+1
                    
    rmse= math.sqrt(rmse_sum / count)

    print("--------------------------------------------------------------------------------")

    print("The root mean square error is: ", rmse)

def mae(testuseritemmatrix,full):     
    count=0
    mae=0
    for i in range(Rows):

            for j in range(Cols):

                if testuseritemmatrix[i][j] != -1 and testuseritemmatrix[i][j] != 0:
                    
                    

                    

                    mae= mae + abs(testuseritemmatrix[i][j] - full[i][j])

                    count=count+1
                    
    maef= (mae / count)

    print("--------------------------------------------------------------------------------")

    print("The mae is: ",maef )







def similarity(cur,ite):
    cur=np.array(cur)
    ite=np.array(ite)
    mean_cur= np.nanmean(cur)
    mean_ite=np.nanmean(ite)
    norml_cur= cur-mean_cur
    norml_ite=ite-mean_ite
    bothrated=[]
    for i in range(len(cur)):
        if cur[i] > 0 and ite[i]>0:
            bothrated.append(i)
    if not bothrated:
        return 0,0
    else:
        cur_adjcos=[]
        ite_adjcos=[]
        cur_corre=[]
        ite_corre=[]
        for i in bothrated:
            cur_corre.append(cur[i])
            ite_corre.append(ite[i])
            cur_adjcos.append(norml_cur[i])
            ite_adjcos.append(norml_ite[i])
        return cosine(np.array(cur_adjcos),np.array(ite_adjcos)), correlation(np.array(cur_corre),np.array(ite_corre)) 





full=[]
K=5
time1=1
time2=1
full = [[0 for i in range(943)] for j in range(1682)]
full2 = [[0 for i in range(943)] for j in range(1682)]


for activeUser in ratingmatrix.index:
     
    similarities_adjcos =pd.DataFrame(index=ratingmatrix.index,columns=['Similarity'])
    similarities_corre = pd.DataFrame(index=ratingmatrix.index,columns=['Similarity'])
  
    for i in ratingmatrix.index:
        if i == activeUser:
            continue
        
        similarities_adjcos.loc[i],similarities_corre.loc[i]=similarity(ratingmatrix.loc[activeUser],
                                          ratingmatrix.loc[i])
    
    Kneighbours=pd.DataFrame.sort_values(similarities_adjcos,['Similarity'],ascending=[0])[:K]

    neighbourItemRatings= ratingmatrix.loc[Kneighbours.index]

    for i in ratingmatrix.columns:
         
        if(ratingmatrix.loc[activeUser, i] > 0):
            full[activeUser-1][i-1]=ratingmatrix.loc[activeUser, i]
            
        else:
            mean = np.nanmean(ratingmatrix.loc[activeUser])
            summation_numerator=0
            summation_denominator=0
            ratingmatrix.loc[activeUser, i]
            for j in neighbourItemRatings.index:
                
                if ratingmatrix.loc[j,i]>0:
                  
                    summation_numerator =summation_numerator + (ratingmatrix.loc[j,i]-np.nanmean(ratingmatrix.loc[j]))*Kneighbours.loc[j,'Similarity']
                    summation_denominator=summation_denominator + Kneighbours.loc[j,'Similarity']

            if summation_denominator==0:       
                predictedRating = mean
            else:
                predictedRating = mean + summation_numerator/summation_denominator
            full[activeUser-1][i-1]= predictedRating
    print(time1,"one item completed with cosine")
    time1=time1+1




    Kneighbours=pd.DataFrame.sort_values(similarities_corre,['Similarity'],ascending=[0])[:K]

    neighbourItemRatings= ratingmatrix.loc[Kneighbours.index]

    for i in ratingmatrix.columns:
         
        if(ratingmatrix.loc[activeUser, i] > 0):
            full2[activeUser-1][i-1]=ratingmatrix.loc[activeUser, i]
            
        else:
            mean = np.nanmean(ratingmatrix.loc[activeUser])
            summation_numerator=0
            summation_denominator=0
            ratingmatrix.loc[activeUser, i]
            for j in neighbourItemRatings.index:
                
                if ratingmatrix.loc[j,i]>0:
                  
                    summation_numerator =summation_numerator + (ratingmatrix.loc[j,i]-np.nanmean(ratingmatrix.loc[j]))*Kneighbours.loc[j,'Similarity']
                    summation_denominator=summation_denominator + Kneighbours.loc[j,'Similarity']

            if summation_denominator==0:       
                predictedRating = mean
            else:
                predictedRating = mean + summation_numerator/summation_denominator
            full2[activeUser-1][i-1]= predictedRating
    print(time2,"one item completed with correlation")
    time2=time2+1
    
                    
    





data=pd.read_csv(path2,sep="\t",header=None,names=['userId','itemId','rating','timestamp'])


table=pd.pivot_table(data, values='rating', index=['itemId'], columns=['userId'])

print(len(table.index))


Rows=len(table.index)
Cols=len(table.columns)

#getting the testing data into multidimensional python lists.


testuseritemmatrix = [[0 for i in range(943)] for j in range(1682)]
for i in table.index:
    
    for j in table.columns:
        
        if table.loc[i,j]> 0:
            testuseritemmatrix[i-1][j-1]= table.loc[i,j]
            
        else:
            testuseritemmatrix[i-1][j-1]= -1
     
    
rmse(testuseritemmatrix,full)
rmse(testuseritemmatrix,full2)

mae(testuseritemmatrix,full)
mae(testuseritemmatrix,full2)

#assessing the models performance using RMSE, on training data.


            

