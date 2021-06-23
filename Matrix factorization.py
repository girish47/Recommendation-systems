
import math
import numpy as np
import pandas as pd
import random





pathtrain1='C:/Users/Scorpio/Desktop/ML project/ml-100k/u1.base'
pathtrain2='C:/Users/Scorpio/Desktop/ML project/ml-100k/u2.base'
pathtrain3='C:/Users/Scorpio/Desktop/ML project/ml-100k/u3.base'
pathtrain4='C:/Users/Scorpio/Desktop/ML project/ml-100k/u4.base'
pathtrain5='C:/Users/Scorpio/Desktop/ML project/ml-100k/u5.base'



pathtest1='C:/Users/Scorpio/Desktop/ML project/ml-100k/u1.test'
pathtest2='C:/Users/Scorpio/Desktop/ML project/ml-100k/u2.test'
pathtest3='C:/Users/Scorpio/Desktop/ML project/ml-100k/u3.test'
pathtest4='C:/Users/Scorpio/Desktop/ML project/ml-100k/u4.test'
pathtest5='C:/Users/Scorpio/Desktop/ML project/ml-100k/u5.test'


trainpaths=[]

trainpaths.append(pathtrain1)
trainpaths.append(pathtrain2)
trainpaths.append(pathtrain3)
trainpaths.append(pathtrain4)
trainpaths.append(pathtrain5)



testpaths=[]

testpaths.append(pathtest1)
testpaths.append(pathtest2)
testpaths.append(pathtest3)
testpaths.append(pathtest4)
testpaths.append(pathtest5)

rmse_alldatasets=[]

def rmse(testuseritemmatrix,userlatentmatrix,itemlatentmatrix):     
        count=0
        rmse_sum=0
        for i in range(943):

                for j in range(1682):

                    if testuseritemmatrix[i][j] > 0:
                        
                        dotproduct=0

                        for n in range(nolafac):
                            dotproduct = dotproduct + userlatentmatrix[i][n] * itemlatentmatrix[n][j]
                            

                        rmse_sum= rmse_sum+ (testuseritemmatrix[i][j] - dotproduct) * (testuseritemmatrix[i][j] - dotproduct)

                        count=count+1
                        
        rmse= math.sqrt(rmse_sum / count)

        print("--------------------------------------------------------------------------------")

        print("The root mean square error is: ", rmse)
        return rmse




for tepath,trpath in zip(trainpaths,testpaths):




    data=pd.read_csv(tepath,sep="\t",header=None,names=['userId','itemId','rating','timestamp'])


    table=pd.pivot_table(data, values='rating', index=['userId'], columns=['itemId'])
    print("First few rows of the user-item matrix used for training")
    print(table.head())




    Rows=len(table.index)


    Cols=len(table.columns)


    #getting the data from the pivot table into multidimensional python lists.

    print(table.loc[1,1])
    useritemmatrix = [[0 for i in range(1682)] for j in range(943)]
    for i in table.index:
        for j in table.columns:
            if table.loc[i,j]> 0:
                useritemmatrix[i-1][j-1]= table.loc[i,j]
            else:
                useritemmatrix[i-1][j-1]= -1
         
        

    nolafac= 2


    for j in range(10):
        print(useritemmatrix[0][j])



    #setting up the userlatent matrix with random values initially
    userlatentmatrix=[]

    for i in range(943):
        temp1=[]

        for j in range(nolafac):

            temp1.append(round(random.uniform(0.1,0.9),2))

        userlatentmatrix.append(temp1)

     
    #setting up the itemlatent matrix with random values initially        
        
    itemlatentmatrix=[]

    for i in range(nolafac):

        temp2=[]

        for j in range(1682):

            temp2.append(round(random.uniform(0.1,0.9),2))

        itemlatentmatrix.append(temp2)




    pathtest1='C:/Users/Scorpio/Desktop/ML project/ml-100k/u1.test'
    pathtest2='C:/Users/Scorpio/Desktop/ML project/ml-100k/u2.test'
    pathtest3='C:/Users/Scorpio/Desktop/ML project/ml-100k/u3.test'
    pathtest4='C:/Users/Scorpio/Desktop/ML project/ml-100k/u4.test'
    pathtest5='C:/Users/Scorpio/Desktop/ML project/ml-100k/u5.test'





    data=pd.read_csv(trpath,sep="\t",header=None,names=['userId','itemId','rating','timestamp'])


    table=pd.pivot_table(data, values='rating', index=['userId'], columns=['itemId'])
    print("First few rows of the user-item matrix used for testing")
    print(table.head())

    Rows=len(table.index)
    Cols=len(table.columns)

    #getting the testing data into multidimensional python lists.

    testuseritemmatrix = [[0 for i in range(1682)] for j in range(943)]
    for i in table.index:
        
        for j in table.columns:
            
            if table.loc[i,j]> 0:
                testuseritemmatrix[i-1][j-1]= table.loc[i,j]
                
            else:
                testuseritemmatrix[i-1][j-1]= -1




















    #Matrix factorization training using stochastic gradient descent.

    epoch=502

    regularizer=0.01

    #stepping constant of gradient descent
    gamma=0.0001

    rmsevalues=[]



    for t in range(epoch):
        for i in range(943):
            for j in range(1682):

                if useritemmatrix[i][j] != -1:
                    #if the rating is available in a cell of useritemrating matrix of the training data

                    #find the corresponding dot product of the userlatent matrix row and itemlatent matrix column  
                    dotproduct=0
                    for n in range(nolafac):
                        dotproduct = dotproduct + userlatentmatrix[i][n] * itemlatentmatrix[n][j]
                        

                    error_ij = useritemmatrix[i][j] - dotproduct

                    #updating the userlatent matrix row and itemlatent matrix column using the stochastic gradient descent
                    for n in range(nolafac):
                        
                        userlatentgradient= 2* error_ij*itemlatentmatrix[n][j] - regularizer* userlatentmatrix[i][n]
                        userlatentmatrix[i][n] = userlatentmatrix[i][n] + gamma * userlatentgradient
                        itemlatentgradient= 2* error_ij*userlatentmatrix[i][n] - regularizer* itemlatentmatrix[n][j]
                        itemlatentmatrix[n][j]= itemlatentmatrix[n][j] + itemlatentgradient


        # used this to tweak and experiment with regularization and stepping constant of gradient descent.
        print("updated , first three values of row zero.")
        dotproduct1=0
        dotproduct2=0
        dotproduct3=0
        if t==10 or t==25 or t==50 or t==100 or t==250 or t==500:
            rmsevalue= rmse(testuseritemmatrix,userlatentmatrix,itemlatentmatrix)
            rmsevalues.append(rmsevalue)
        for n in range(len(userlatentmatrix[i])):
                        dotproduct1 = dotproduct1 + userlatentmatrix[0][n] * itemlatentmatrix[n][0]
                        dotproduct2 = dotproduct2 + userlatentmatrix[0][n] * itemlatentmatrix[n][1]
                        dotproduct3 = dotproduct3 + userlatentmatrix[0][n] * itemlatentmatrix[n][3]
        print(dotproduct1)
        print(dotproduct2)
        print(dotproduct3)
     
    print("------------------------------------------------------------------------------")
    print(rmsevalues)
    rmse_alldatasets.append(rmsevalues)
    
print(rmse_alldatasets)
avgrmse_alldatasets=[]
    
for i in range(len(rmse_alldatasets[0])):
    su=0
    av=0
    for j in range(len(rmse_alldatasets)):
        su= su+rmse_alldatasets[j][i]
    av=su/5
    avgrmse_alldatasets.append(av)

print(avgrmse_alldatasets)
