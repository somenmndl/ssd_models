#!/usr/bin/env python
# coding: utf-8

# In[14]:


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
import pandas as pd
import seaborn as sns


# In[15]:


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
        df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, )
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('Truth')
        plt.xlabel('Prediction')


# In[16]:


truth=["Person", "Not Person", "Not Person","Person","Person","Not Person","Person"]
prediction=["Not Person","Not Person","Not Person","Not Person","Not Person","Not Person","Not Person"]


# In[17]:


cm = confusion_matrix(truth,prediction)
print_confusion_matrix(cm,["Person","Not Person"])


# In[18]:


print(classification_report(truth, prediction))


# In[ ]:




