import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

def heatmap(dataframe):
    
    feats = dataframe.columns.to_list()[:10]
    matrix = dataframe[feats].corr()
    selected = set()
    for i in range(len(matrix.columns)):
        for j in range(i):
            if abs(matrix.iloc[i, j]) < 0.7:
                selected.add(matrix.columns[i])
    selected = list(selected)
                                
    plt.figure(figsize=[12, 10])
    sns.heatmap(dataframe[selected].corr(), annot=True, cmap="mako", linewidths=2, linecolor='white')
    plt.title('Heatmap')
    plt.show()
    plt.clf()
    
    
def confusion(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)
    threshs = [i*1/10 for i in range(1, 10)]
    for thresh in reversed(threshs):
        plt.figure(figsize=[8, 6])
        y_pred = np.where(model.predict_proba(X_test)[:, 1] > thresh, 1, 0)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', linewidths=2, linecolor='white',
        xticklabels=['Predicted malignant', 'Predicted benign'], yticklabels=['Actual malignant', 'Actual benign'])
        plt.text(x= 2.5, y=1.40, s=f'''
            Accuracy: {round(100*accuracy_score(y_test, y_pred))}%\n\n\n\n\n
            Precision: {round(100*precision_score(y_test, y_pred))}%\n\n\n\n\n
            Recall: {round(100*recall_score(y_test, y_pred))}%''')
        plt.text(x= 2.5, y=2.0, s=f'F1_score: {round(100*f1_score(y_test, y_pred))}%', fontsize=13)
        plt.text(x= .30, y=0.25, s='True negative')
        plt.text(x= .30, y=1.25, s='False negative')
        plt.text(x= 1.30, y=0.25, s='False positive')
        plt.text(x= 1.30, y=1.25, s='True positive')
        plt.title(f'Confusion matrix - prediction threshold: {thresh}')
        plt.show()
        plt.clf()