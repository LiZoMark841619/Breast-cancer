import matplotlib.pyplot as plt
import seaborn as sns
def heatmap_(dataframe):
    
    feats = dataframe.columns.to_list()
    matrix = dataframe[feats].corr()
    selected_feats = []
    for i in range(len(matrix.columns)):
        for j in range(i):
            if abs(matrix.iloc[i, j]) > 0.7:
                selected_feats.extend((matrix.columns[i], matrix.columns[j]))
                
    plt.figure(figsize=[12, 10])
    sns.heatmap(dataframe[selected_feats].corr(), annot=True, cmap="mako", linewidths=2, linecolor='white')
    plt.title('Heatmap')
    plt.savefig('good_heatmap.png')
    plt.show()
    plt.clf()