import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

#reading the data
data_training=pd.read_excel('G:/A S2/SEM 1/AI_Pak Ali Ridho/M6/TugasBayes/Datasetbayes.xlsx',sheet_name='Dataset')
data_testing=pd.read_excel('G:/A S2/SEM 1/AI_Pak Ali Ridho/M6/TugasBayes/Datasetbayes.xlsx',sheet_name='Datatest')

#splitting the data
train_x = data_training.drop(columns=['Class'])
train_y = data_training['Class']
test_x = data_testing.drop(columns=['Class'])
test_y = data_testing['Class']
test_x0=test_x[1:5]
test_x1=test_x[5:10]
test_x2=test_x[11:15]
test_x3=test_x[16:20]
test_x4=test_x[21:25]
test_x5=test_x[26:30]
test_x6=test_x[31:35]
test_x7=test_x[36:40]
test_x8=test_x[41:45]
test_x9=test_x[46:50]
test_y0=test_y[1:5]
test_y1=test_y[5:10]
test_y2=test_y[11:15]
test_y3=test_y[16:20]
test_y4=test_y[21:25]
test_y5=test_y[26:30]
test_y6=test_y[31:35]
test_y7=test_y[36:40]
test_y8=test_y[41:45]
test_y9=test_y[46:50]

#print (test_y9)

#visualize the data
data1=train_x.loc[[1],:]
data1=data1.to_numpy()
img = data1.reshape((11,11))
plt.imshow(img, cmap="Greys")
plt.show()

#converting the data
train_y=train_y.to_numpy()
test_y=test_y.to_numpy()


#training and testing
model = BernoulliNB(alpha=1.0e-10)
model.fit(train_x, train_y)
y_pred0 = model.predict(test_x0)
y_pred1 = model.predict(test_x1)
y_pred2 = model.predict(test_x2)
y_pred3 = model.predict(test_x3)
y_pred4 = model.predict(test_x4)
y_pred5 = model.predict(test_x5)
y_pred6 = model.predict(test_x6)
y_pred7 = model.predict(test_x7)
y_pred8 = model.predict(test_x8)
y_pred9 = model.predict(test_x9)
y_pred = model.predict(test_x)
akurasi=(accuracy_score(test_y, y_pred)*100)
print("The accuracy for the model is ",akurasi,"%")
print("Error rate: ",100-akurasi,"%")
print("Error rate 0: ",100-(accuracy_score(test_y0, y_pred0)*100),"%")
print("Error rate 1: ",100-(accuracy_score(test_y1, y_pred1)*100),"%")
print("Error rate 2: ",100-(accuracy_score(test_y2, y_pred2)*100),"%")
print("Error rate 3: ",100-(accuracy_score(test_y3, y_pred3)*100),"%")
print("Error rate 4: ",100-(accuracy_score(test_y4, y_pred4)*100),"%")
print("Error rate 5: ",100-(accuracy_score(test_y5, y_pred5)*100),"%")
print("Error rate 6: ",100-(accuracy_score(test_y6, y_pred6)*100),"%")
print("Error rate 7: ",100-(accuracy_score(test_y7, y_pred7)*100),"%")
print("Error rate 8: ",100-(accuracy_score(test_y8, y_pred8)*100),"%")
print("Error rate 9: ",100-(accuracy_score(test_y9, y_pred9)*100),"%")
plot_confusion_matrix(model, test_x, test_y)
plt.show()
