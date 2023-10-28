import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
work_path = os.path.dirname(os.path.abspath(__file__))
data_path = work_path+'/iris_dataset.csv'

data = pd.read_csv(data_path,header='infer',sep=',')
select_train = data.columns[:4]
X = data[select_train]
Y = data['target']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8)

lr = LogisticRegression(C=1e5)
lr.fit(X_train,Y_train)

Y_predict = lr.predict(X_test)
con_mat = confusion_matrix(Y_test,Y_predict)

plt.figure(figsize=(8,8))
sns.heatmap(con_mat,annot=True,cmap='Blues')
plt.xlabel('Y_predict')
plt.ylabel('Y_test')
plt.show()
