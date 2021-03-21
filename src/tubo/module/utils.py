import glob
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

def save_output(sub_data,evals_result=None,figure=None):
    sub_count = len(glob.glob('../output/*'))
    dir_name = ('../output/sub_{:0=3}'.format(sub_count))
    os.mkdir(dir_name)
    sub_data.to_csv(dir_name+'/sub_{:0=3}.csv'.format(sub_count),index=False)
    if(figure != None):
        figure.savefig(dir_name+'/img.jpg')
    if(evals_result!=None):
        with open(dir_name+'/evals_result.json', 'w') as outfile:
            json.dump(evals_result,outfile,indent=4)
    return

def save_output_binary(train,test,evals_result=None,figure=None):
    sub_count = len(glob.glob('../output_binary/*'))
    dir_name = ('../output_binary/sub_{:0=3}'.format(sub_count))
    os.mkdir(dir_name)
    train.to_csv(dir_name+'/sub_train_{:0=3}.csv'.format(sub_count),index=False)
    test.to_csv(dir_name+'/sub_test_{:0=3}.csv'.format(sub_count),index=False)
    if(evals_result != None):
        figure.savefig(dir_name+'/img.jpg')
    if(figure!=None):
        with open(dir_name+'/evals_result.json', 'w') as outfile:
            json.dump(evals_result,outfile,indent=4)
    return

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
