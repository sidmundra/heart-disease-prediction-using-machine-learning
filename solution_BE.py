
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification,roc_curve
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import log_loss
from matplotlib import pyplot
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
from mlxtend.classifier import StackingClassifier
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
import wx
import wx.xrc


data = pd.read_csv("heart.csv")

data.head(5)

col = data.columns

data.isnull().sum()

X = data.drop(['target'],1)
y=data['target']

X_train, X_test, y_train, y_test = tts(
    X,
    y,
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape



def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



def RF(X_train,y_train,X_test,y_test): 
  ada = RandomForestClassifier(n_estimators = 24,random_state=5)
  ada.fit(X_train,y_train)
  print("Random Forest:train set")
  y_pred = ada.predict(X_train)
  pred=ada.predict_proba(X_test)   
  print("Random Forest:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("Random Forest:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("Random Forest:Test set")
  y_pred = ada.predict(X_test)
  print("Random Forest:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("Random Forest:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = ada.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=ada.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(ada, classes=classes, support=True )
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

RF(X_train,y_train,X_test,y_test)

#KNN
def KNN(X_train,y_train,X_test,y_test):
  xgb=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
  xgb.fit(X_train,y_train)
  print("KNN:train set")
  y_pred = xgb.predict(X_train)
  pred=xgb.predict_proba(X_test)   
  print("KNN:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("KNN:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("KNN:Test set")
  y_pred = xgb.predict(X_test)
  print("KNN:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("KNN:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = xgb.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=xgb.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(xgb, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

KNN(X_train,y_train,X_test,y_test)
#Logistic Regression
def LR(X_train,y_train,X_test,y_test):
  sc=LogisticRegression()
  sc.fit(X_train,y_train)
  print("Logisitc Regresion:train set")
  y_pred = sc.predict(X_train)
  pred=sc.predict_proba(X_test)   
  print("Logisitc Regresion:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("Logisitc Regresion:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("Logisitc Regresion:Test set")
  y_pred = sc.predict(X_test)
  print("Logisitc Regresion:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("Logisitc Regresion:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #Confusion Matrix

  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
#ROC_AUC curve
  probs = sc.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=sc.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(sc, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

LR(X_train,y_train,X_test,y_test)

#NB Classifier
def NB(X_train,y_train,X_test,y_test):
  vc = GaussianNB()  
  vc.fit(X_train,y_train)
  print("Naive Bayes :train set")
  y_pred = vc.predict(X_train)
  pred=vc.predict_proba(X_test)   
  print("Naive Bayes :Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("Naive Bayes :Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("Naive Bayes :Test set")
  y_pred = vc.predict(X_test)
  print("Naive Bayes :Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("Naive Bayes :Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = vc.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=vc.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(vc, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

NB(X_train,y_train,X_test,y_test)

#SVM Classifier
def SV(X_train,y_train,X_test,y_test):
  qo=SVC(kernel = 'rbf')  
  qo=SVC(probability=True)
  qo.fit(X_train,y_train)
  print("SVM  :train set")
  y_pred = qo.predict(X_train)
  pred=qo.predict_proba(X_test)   
  print("SVM  :Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("SVM  :Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("SVM  :Test set")
  y_pred = qo.predict(X_test)
  print("SVM  :Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("SVM  :Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = qo.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=qo.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(qo, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()
SV(X_train,y_train,X_test,y_test)
#Decision Tree
def DTC(X_train,y_train,X_test,y_test):
  dtc=DecisionTreeClassifier(random_state=2)
  dtc.fit(X_train,y_train)
  print("DecisionTreeClassifier:train set")
  y_pred = dtc.predict(X_train)
  pred=dtc.predict_proba(X_test)   
  print("DecisionTreeClassifier:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("DecisionTreeClassifier:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("DecisionTreeClassifier:Test set")
  y_pred = dtc.predict(X_test)
  print("DecisionTreeClassifier:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("DecisionTreeClassifier:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #Confusion Matrix

  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
#ROC_AUC curve
  probs = dtc.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=dtc.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(dtc, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()
DTC(X_train,y_train,X_test,y_test)





###########################################################################
## Class MyFrame3
###########################################################################

class MyFrame3 ( wx.Frame ):
 def __init__( self, parent ):
    wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = "Heart Disease Prediction", pos = wx.DefaultPosition, size = wx.Size( 500,600 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
    self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
    self.SetBackgroundColour('White')
		
    fgSizer1 = wx.FlexGridSizer( 0, 2, 0, 0 )
    fgSizer1.SetFlexibleDirection( wx.BOTH )
    fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
    self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"Age", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText1.Wrap( -1 )
    fgSizer1.Add( self.m_staticText1, 0, wx.ALL, 5 )
		
    self.m_textCtrl1 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl1, 0, wx.ALL, 5 )
		
    self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"Sex", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText3.Wrap( -1 )
    fgSizer1.Add( self.m_staticText3, 0, wx.ALL, 5 )
		
    self.m_textCtrl3 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl3, 0, wx.ALL, 5 )
		
    self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"Cp", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText4.Wrap( -1 )
    fgSizer1.Add( self.m_staticText4, 0, wx.ALL, 5 )
		
    self.m_textCtrl4 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl4, 0, wx.ALL, 5 )
		
    self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"Trestbps", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText5.Wrap( -1 )
    fgSizer1.Add( self.m_staticText5, 0, wx.ALL, 5 )
		
    self.m_textCtrl5 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl5, 0, wx.ALL, 5 )
		
    self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"Chol", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText6.Wrap( -1 )
    fgSizer1.Add( self.m_staticText6, 0, wx.ALL, 5 )
		
    self.m_textCtrl6 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl6, 0, wx.ALL, 5 )
		
    self.m_staticText7 = wx.StaticText( self, wx.ID_ANY, u"Fbs", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText7.Wrap( -1 )
    fgSizer1.Add( self.m_staticText7, 0, wx.ALL, 5 )
		
    self.m_textCtrl7 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl7, 0, wx.ALL, 5 )
		
    self.m_staticText8 = wx.StaticText( self, wx.ID_ANY, u"Restecg", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText8.Wrap( -1 )
    fgSizer1.Add( self.m_staticText8, 0, wx.ALL, 5 )
		
    self.m_textCtrl8 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl8, 0, wx.ALL, 5 )
		
    self.m_staticText9 = wx.StaticText( self, wx.ID_ANY, u"Thalach", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText9.Wrap( -1 )
    fgSizer1.Add( self.m_staticText9, 0, wx.ALL, 5 )
		
    self.m_textCtrl9 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl9, 0, wx.ALL, 5 )
		
    self.m_staticText10 = wx.StaticText( self, wx.ID_ANY, u"Exang", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText10.Wrap( -1 )
    fgSizer1.Add( self.m_staticText10, 0, wx.ALL, 5 )
		
    self.m_textCtrl10 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl10, 0, wx.ALL, 5 )
		
    self.m_staticText11 = wx.StaticText( self, wx.ID_ANY, u"Oldpeak", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText11.Wrap( -1 )
    fgSizer1.Add( self.m_staticText11, 0, wx.ALL, 5 )
		
    self.m_textCtrl11 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl11, 0, wx.ALL, 5 )
		
    self.m_staticText14 = wx.StaticText( self, wx.ID_ANY, u"Slope", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText14.Wrap( -1 )
    fgSizer1.Add( self.m_staticText14, 0, wx.ALL, 5 )
		
    self.m_textCtrl13 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl13, 0, wx.ALL, 5 )
		
    self.m_staticText15 = wx.StaticText( self, wx.ID_ANY, u"Ca", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText15.Wrap( -1 )
    fgSizer1.Add( self.m_staticText15, 0, wx.ALL, 5 )
		
    self.m_textCtrl15 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl15, 0, wx.ALL, 5 )
		
    self.m_staticText16 = wx.StaticText( self, wx.ID_ANY, u"Thal", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText16.Wrap( -1 )
    fgSizer1.Add( self.m_staticText16, 0, wx.ALL, 5 )
		
    self.m_textCtrl16 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl16, 0, wx.ALL, 5 )
		
    self.m_staticText12 = wx.StaticText( self, wx.ID_ANY, u"Click to Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText12.Wrap( -1 )
    fgSizer1.Add( self.m_staticText12, 0, wx.ALL, 5 )
		
    self.m_button1 = wx.Button( self, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_button1, 0, wx.ALL, 5 )

  
		
    self.m_staticText16 = wx.StaticText( self, wx.ID_ANY, u"Prediction", wx.DefaultPosition, wx.DefaultSize, 0 )
    self.m_staticText16.Wrap( -1 )
    fgSizer1.Add( self.m_staticText16, 0, wx.ALL, 5 )
		
    self.m_textCtrl12 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    fgSizer1.Add( self.m_textCtrl12, 0, wx.ALL, 5 )
		
		
    self.SetSizer( fgSizer1 )
    self.Layout()
    # Connect Events
    self.m_button1.Bind( wx.EVT_BUTTON,self.submit)
    
    self.Centre( wx.BOTH )
  #pass
 def __del__( self ):
		pass
  # Virtual event handlers, overide them in your derived class
 def submit(self,event):
    # block not indented
    age = float(self.m_textCtrl1.GetValue())	
    sex = float(self.m_textCtrl3.GetValue())
    cp = float(self.m_textCtrl4.GetValue())
    trestbps = float(self.m_textCtrl5.GetValue())
    chol = float(self.m_textCtrl6.GetValue())
    fbs = float(self.m_textCtrl7.GetValue())
    restecg = float(self.m_textCtrl8.GetValue())
    thalach = float(self.m_textCtrl9.GetValue())
    exang = float(self.m_textCtrl10.GetValue())
    oldpeak = float(self.m_textCtrl11.GetValue())
    slope = float(self.m_textCtrl13.GetValue())
    ca =float(self.m_textCtrl15.GetValue())
    thal = float(self.m_textCtrl16.GetValue())

    col = data.columns

    col=col[:-1]

    output_data=[]
    output_data.append(age)
    output_data.append(sex)
    output_data.append(cp)
    output_data.append(trestbps)
    output_data.append(chol)
    output_data.append(fbs)
    output_data.append(restecg)
    output_data.append(thalach)
    output_data.append(exang)
    output_data.append(oldpeak)
    output_data.append(slope)
    output_data.append(ca)
    output_data.append(thal)

    output_data=pd.DataFrame([output_data],columns = col)

    sc1 = RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)
    sc1.fit(X_train,y_train)
    pred=sc1.predict(output_data)
    print("Prediction for newly added data : ",pred)

    if(pred==1):
      self.m_textCtrl12.SetValue(str("Heart Disease"))
    else:
      self.m_textCtrl12.SetValue(str("No disease"))

 




      
app1 = wx.App(False)
frame1 = MyFrame3(None)
frame1.Show(True)
app1.MainLoop()
  
  
