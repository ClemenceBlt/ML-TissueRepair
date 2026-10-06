import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# trace de la moyenne de loss et d'accuracy sur les folds a partir des fichiers json de chaque fold
# figures enregistrees dans dossier figures_generees

# Parametres
#classif_type = 'binary'
classif_type = 'mice'
# classif_type = "dayclassif"
# classif_type = 'binary_revgrad'
# classif_type = 'mice_revgrad'
suffix = 'SGDColorAug'
echeance = 'j03j10'
start_ep1 = 0
last_ep1 = 299  # premier entrainement
start_ep2 = 200
last_ep2 = 1199  # deuxieme entrainement, continue apres le premier
start_ep3 = 800
last_ep3 = 1499  # troisieme entrainement, continue apres le deuxieme
start_ep4 = 1500
last_ep4 = 2499  # quatrieme entrainement, continue apres le troisieme
start_ep5 = 2500
last_ep5 = 3499
kfolds = 5
dataset_name = '230330_miceTV'
batch_size = 16
learning_rate = 1e-3
w_decay = 1e-5
#net_name = "lenet_CV"
#net_name = "resnet"
#net_name = 'SqueezeNet'
#net_name = 'MobileNet'
net_name = 'vgg11'
# net_name = 'revgrad'

results1 = [0 for k in range(kfolds)]
results2 = [0 for k in range(kfolds)]
results3 = [0 for k in range(kfolds)]
results4 = [0 for k in range(kfolds)]
results5 = [0 for k in range(kfolds)]

# Recuperation des fichiers de l'entrainement

if classif_type == 'binary':  # classification binaire
    prefix = 'train_ch12_j03j10_' + net_name
    #criterion = "BinaryCrossEntropyLoss"  # BCE With Logits Loss
    criterion = "CrossEntropyLoss"
    fig_name = 'Binary Classification naloxone/small lipectomy'
elif classif_type == 'mice':  # classification des souris
    prefix = 'mice_j03j10_' + net_name
    criterion = "CrossEntropyLoss"
    fig_name = 'Mice Classification'
elif classif_type == 'dayclassif':  # classification des echeances
    prefix = 'dayclassif_' + net_name
    criterion = "CrossEntropyLoss"
    fig_name = 'Binary Classification D3/D10'
elif classif_type == 'binary_revgrad':
    prefix = 'revgrad'
    criterion = 'BinaryCrossEntropyLoss'
    fig_name = 'Binary Classification with RevGrad'
    abr = 'bc'
else:  # cas mice_revgrad
    prefix = 'revgrad'
    criterion = 'CrossEntropyLoss'
    fig_name = 'Mice Classification with RevGrad'
    abr = 'mc'

if net_name == 'revgrad':
    if suffix == '_':
        for k in range(kfolds):
            results1[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s.json'
                                          % (prefix, dataset_name, start_ep1, last_ep1, batch_size, learning_rate, k))
                                     .read())
            results2[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s.json'
                                          % (prefix, dataset_name, start_ep2, last_ep2, batch_size, learning_rate,
                                             k)).read())
            results3[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s.json'
                                          % (prefix, dataset_name, start_ep3, last_ep3, batch_size,
                                             learning_rate, k)).read())
            #results4[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s.json'
            #                              % (prefix, dataset_name, start_ep4, last_ep4, batch_size,
            #                              learning_rate, k)).read())
    else:
        for k in range(kfolds):
            results1[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep1, last_ep1, batch_size,
                                             learning_rate, k, suffix)).read())
            results2[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep2, last_ep2, batch_size,
                                             learning_rate, k, suffix)).read())
            results3[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep3, last_ep3, batch_size,
                                             learning_rate, k, suffix)).read())
            #results4[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_validfld%s_%s.json'
            #                              % (prefix, dataset_name, start_ep4, last_ep4, batch_size,
            #                              learning_rate, k, suffix)).read())
else:
    if suffix == '_':
        for k in range(kfolds):
            results1[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s.json'
                                          % (prefix, dataset_name, start_ep1, last_ep1, batch_size, learning_rate, w_decay,
                                             k)).read())
            """
            results2[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s.json'
                                          % (prefix, dataset_name, start_ep2, last_ep2, batch_size, learning_rate, w_decay,
                                             k)).read())
            results3[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s.json'
                                          % (prefix, dataset_name, start_ep3, last_ep3, batch_size,
                                             learning_rate, w_decay, k)).read())
            results4[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s.json'
                                          % (prefix, dataset_name, start_ep4, last_ep4, batch_size,
                                          learning_rate, w_decay, k)).read())
            """
    else:
        for k in range(kfolds):
            """
            if k == 0:
                results1[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                                                  % (prefix, dataset_name, start_ep1, 499, batch_size,
                                                     learning_rate, w_decay, k, suffix)).read())
                #results2[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                #                          % (prefix, dataset_name, start_ep2, 1499, batch_size,
                #                             learning_rate, w_decay, k, suffix)).read())
            else:
            """
            results1[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                                              % (prefix, dataset_name, start_ep1, last_ep1, batch_size,
                                                 learning_rate, w_decay, k, suffix)).read())
            """
            results2[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep2, last_ep2, batch_size,
                                             learning_rate, w_decay, k, suffix)).read())
            results3[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep3, last_ep3, batch_size,
                                             learning_rate, w_decay, k, suffix)).read())
            results4[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep4, last_ep4, batch_size,
                                             learning_rate, w_decay, k, suffix)).read())
            results5[k] = json.loads(open('./json_files/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_testfld%s_%s.json'
                                          % (prefix, dataset_name, start_ep5, last_ep5, batch_size,
                                             learning_rate, w_decay, k, suffix)).read())
            """
                

# Trace des fonctions objectifs
def plot_loss(resu1, folds_number, losscriterion, lr, bs, wdecay, datasetname, figname, pref, sfx, startep1, lastep1,
              startep2=0, lastep2=-2, resu2=[], startep3=0, lastep3=-2, resu3=[], startep4=0, lastep4=-2, resu4=[],
              startep5=0, lastep5=-2, resu5=[]):
    somme_train = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5) + 1))
    mean_train = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5) + 1))

    if pref == 'revgrad':
        trainloss = 'train_loss_' + abr
        validloss = 'valid_loss_' + abr
    else:
        trainloss = 'train_loss'
        validloss = 'test_loss'  # valid_loss

    for i in range(lastep1 - startep1 + 1):
        somme_train[i] = 0
        for k in range(folds_number):
            somme_train[i] += resu1[k]['fold' + str(k)][trainloss][i]
        mean_train[i] = somme_train[i]/folds_number
    for i in range(lastep2-startep2+1):
        somme_train[startep2 + i] = 0
        for k in range(folds_number):
            if k==0:
                somme_train[startep2 + i] += resu2[k]['fold' + str(k)][trainloss][i]
            else:
                somme_train[startep2 + i] += resu1[k]['fold' + str(k)][trainloss][startep2 + i]
        mean_train[startep2 + i] = somme_train[startep2 + i]/folds_number
    """

    for i in range(lastep3-startep3+1):
        somme_train[startep3 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_train[startep3 + i] += resu3[k]['fold' + str(k)][trainloss][i]
            else:
                somme_train[startep3 + i] += resu1[k]['fold' + str(k)][trainloss][startep3 + i]
        mean_train[startep3 + i] = somme_train[startep3 + i]/folds_number

    for i in range(lastep4 - startep4 + 1):
        somme_train[startep4 + i] = 0
        for k in range(folds_number):
            if k==0:
                somme_train[startep4 + i] += resu4[k]['fold' + str(k)][trainloss][i]
            else:
                somme_train[startep4 + i] += resu2[k]['fold' + str(k)][trainloss][i]
        mean_train[startep4 + i] = somme_train[startep4 + i] / folds_number

    for i in range(lastep5 - startep5 + 1):
        somme_train[startep5 + i] = 0
        for k in range(folds_number):
            if k==0:
                somme_train[startep5 + i] += resu5[k]['fold' + str(k)][trainloss][i]
            else:
                somme_train[startep5 + i] += resu2[k]['fold' + str(k)][trainloss][startep5 - startep4 + i]
        mean_train[startep5 + i] = somme_train[startep5 + i] / folds_number
    """
    somme_test = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5) + 1))
    mean_test = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5) + 1))

    for i in range(lastep1 - startep1 + 1):
        somme_test[i] = 0
        for k in range(folds_number):
            somme_test[i] += resu1[k]['fold' + str(k)][validloss][i]
        mean_test[i] = somme_test[i]/folds_number
    for i in range(lastep2-startep2+1):
        somme_test[startep2 + i] = 0
        for k in range(folds_number):
            if k==0:
                somme_test[startep2 + i] += resu2[k]['fold' + str(k)][validloss][i]
            else:
                somme_test[startep2 + i] += resu1[k]['fold' + str(k)][validloss][startep2 + i]
        mean_test[startep2 + i] = somme_test[startep2 + i]/folds_number
    """

    for i in range(lastep3-startep3+1):
        somme_test[startep3 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_test[startep3 + i] += resu3[k]['fold' + str(k)][validloss][i]
            else:
                somme_test[startep3 + i] += resu1[k]['fold' + str(k)][validloss][startep3 + i]
        mean_test[startep3 + i] = somme_test[startep3 + i]/folds_number

    for i in range(lastep4-startep4+1):
        somme_test[startep4 + i] = 0
        for k in range(folds_number):
            if k==0:
                somme_test[startep4 + i] += resu4[k]['fold' + str(k)][validloss][i]
            else:
                somme_test[startep4 + i] += resu2[k]['fold' + str(k)][validloss][i]
        mean_test[startep4 + i] = somme_test[startep4 + i]/folds_number

    for i in range(lastep5-startep5+1):
        somme_test[startep5 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_test[startep5 + i] += resu5[k]['fold' + str(k)][validloss][i]
            else:
                somme_test[startep5 + i] += resu2[k]['fold' + str(k)][validloss][startep5 - startep4 + i]
        mean_test[startep5 + i] = somme_test[startep5 + i]/folds_number
    """
    if folds_number == 5:
        title = 'Mean of the 5 folds'
    else:
        title = 'Fold ' + str(folds_number-1)

    plt.rcParams["figure.figsize"] = [43.1, 31]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ftsz = 156
    linewd = 8
    ax.plot(mean_test, color='b', linewidth=linewd)
    ax.plot(mean_train, color='r', label=title, linewidth=linewd)
    
    ax.set_xlabel('Number of epochs', fontsize=ftsz)
    #ax.set_ylabel('Loss function \n (%s)' % losscriterion)
    ax.set_ylabel('Loss function', fontsize=ftsz)
    ax.tick_params(labelleft=True, labelbottom=True, size=28, width=linewd)
    ax.spines['bottom'].set_linewidth(linewd)
    ax.spines['top'].set_linewidth(linewd)
    ax.spines['left'].set_linewidth(linewd)
    ax.spines['right'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-30)
    plt.yticks(fontsize=ftsz-30)
    
    # ax.set_ylim([0.15, 0.69])
    """
    #ax.legend()
    #fig.text(0.5, 0.99, '%s' % figname, ha='center', va='top', size='large')
    if pref == 'revgrad':
        fig.text(0.02, 0.95, '%s, %s, %s \nlearning rate=%s, batch size=%s'
                 % (echeance, net_name, losscriterion, lr, bs), ha='left', va='top')
    else:
        fig.text(0.02, 0.95, '%s, %s \nlearning rate=%s, batch size=%s, w_decay=%s'
                 % (# echeance,  # ajouter echeance si pas dayclassif
                     net_name, losscriterion, lr, bs, wdecay), ha='left', va='top')
    fig.text(0.01, 0.04, dataset_name, ha='left', size="small")
    fig.text(0.01, 0.01, sfx, ha='left', size="small")
    fig.text(0.9, 0.92, "training loss", ha="right", va="bottom", size="medium", color='r')
    fig.text(0.9, 0.885, "validation loss", ha="right", va="bottom", size="medium", color='b')
    # ha : 'center', 'right', 'left'
    # va : 'top', 'bottom', 'center', 'baseline', 'center_baseline'
    """
    plt.tight_layout()
    if pref == 'revgrad':
        if sfx == '_':
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_loss.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4), bs, lr))
        else:
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_%s_loss.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4), bs, lr, sfx))
    else:
        if sfx == '_':
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_loss.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5), bs, lr,
                           wdecay), dpi=300)
        else:
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_%s_loss_260604.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5), bs, lr,
                           wdecay, sfx), dpi=300)
            #plt.show()
    plt.close()

#plot_loss(results1, kfolds, criterion, learning_rate, batch_size, w_decay, dataset_name, fig_name, prefix, suffix,
#          start_ep1, last_ep1, start_ep2, last_ep2, results2)#, start_ep3, last_ep3, results3, start_ep4, last_ep4,
          #results4, start_ep5, last_ep5, results5)
#plot_loss(results1, kfolds, criterion, learning_rate, batch_size, w_decay, dataset_name, fig_name, prefix, suffix,
#          start_ep1, last_ep1)


def plot_losswithstd(resu1, lr, bs, wdecay, datasetname, pref, sfx, startep1, lastep1, lastep2=0, resu2=[]):
    '''
    Loss with standard deviation curves.
    '''

    """
    # Putting values of resu2 into resu1 to work on only one table
    for elmt in ['train_loss', 'test_loss']:
        #resu1[0]['fold0'][elmt] += resu2[0]['fold0'][elmt]
        resu1[0]['fold0'][elmt] = resu1[0]['fold0'][elmt][:lastep1+1]  # when more values saved than what we want to plot
    """


    # All loss values of the 5 folds successively
    allfolds_train = []
    allfolds_valid = []
    for k in range(5):
        allfolds_train += resu1[k]['fold' + str(k)]['train_loss']
        allfolds_valid += resu1[k]['fold' + str(k)]['test_loss']

    xvalues = np.concatenate((np.array(range(0, max(lastep1, lastep2)+1)), np.array(range(0, max(lastep1, lastep2)+1)), np.array(range(0, max(lastep1, lastep2)+1)),
                              np.array(range(0, max(lastep1, lastep2)+1)), np.array(range(0, max(lastep1, lastep2)+1))),axis=0)
    df_train = pd.DataFrame({'Epochs':xvalues, 'train_loss':allfolds_train})
    df_valid = pd.DataFrame({'Epochs':xvalues, 'valid_loss':allfolds_valid})
        
    plt.rcParams["figure.figsize"] = [8.62,6.2]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ftsz = 30
    linewd = 1.5

    sns.lineplot(x='Epochs', y='valid_loss', data=df_valid, estimator=np.mean, color='b', errorbar='sd')
    sns.lineplot(x='Epochs', y='train_loss', data=df_train, estimator=np.mean, color='r', errorbar='sd')

    ax.set_xlabel('Number of epochs', fontsize=ftsz)
    ax.set_ylabel('Loss function', fontsize=ftsz)
    ax.tick_params(labelleft=True, labelbottom=True, size=5, width=linewd)
    ax.spines['bottom'].set_linewidth(linewd)
    ax.spines['top'].set_linewidth(linewd)
    ax.spines['left'].set_linewidth(linewd)
    ax.spines['right'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-5)
    plt.yticks(fontsize=ftsz-5)
    
    plt.tight_layout()
    fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_%s_loss_260604.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2), bs, lr, wdecay, sfx), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    

#plot_losswithstd(results1, learning_rate, batch_size, w_decay, dataset_name, prefix, suffix, start_ep1, last_ep1)#, last_ep2, results2)


def plot_accwithstd(resu1, lr, bs, wdecay, datasetname, pref, sfx, startep1, lastep1, lastep2=0, resu2=[]):
    '''
    Accuracy with standard deviation curves.
    '''

    """
    # Putting values of resu2 into resu1 to work on only one table
    for elmt in ['train_acc', 'test_acc']:
        #resu1[0]['fold0'][elmt] += resu2[0]['fold0'][elmt]
        resu1[0]['fold0'][elmt] = resu1[0]['fold0'][elmt][:lastep1+1]
    """
        
    # All loss values of the 5 folds successively
    allfolds_train = []
    allfolds_valid = []
    for k in range(5):
        allfolds_train += resu1[k]['fold' + str(k)]['train_acc']
        allfolds_valid += resu1[k]['fold' + str(k)]['test_acc']

    xvalues = np.concatenate((np.array(range(0, max(lastep1, lastep2)+1)), np.array(range(0, max(lastep1, lastep2)+1)), np.array(range(0, max(lastep1, lastep2)+1)),
                              np.array(range(0, max(lastep1, lastep2)+1)), np.array(range(0, max(lastep1, lastep2)+1))),axis=0)
    df_train = pd.DataFrame({'Epochs':xvalues, 'train_acc':allfolds_train})
    df_valid = pd.DataFrame({'Epochs':xvalues, 'valid_acc':allfolds_valid})
        
    plt.rcParams["figure.figsize"] = [8.62,6.2]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ftsz = 30
    linewd = 1.5

    sns.lineplot(x='Epochs', y='valid_acc', data=df_valid, estimator=np.mean, color='b', errorbar='sd')
    sns.lineplot(x='Epochs', y='train_acc', data=df_train, estimator=np.mean, color='r', errorbar='sd')

    ax.set_xlabel('Number of epochs', fontsize=ftsz)
    ax.set_ylabel('Accuracy', fontsize=ftsz)
    ax.tick_params(labelleft=True, labelbottom=True, size=5, width=linewd)
    ax.spines['bottom'].set_linewidth(linewd)
    ax.spines['top'].set_linewidth(linewd)
    ax.spines['left'].set_linewidth(linewd)
    ax.spines['right'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-5)
    plt.yticks(fontsize=ftsz-5)
    
    plt.tight_layout()
    fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_%s_acc_260604.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2), bs, lr, wdecay, sfx), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    

plot_accwithstd(results1, learning_rate, batch_size, w_decay, dataset_name, prefix, suffix, start_ep1, last_ep1)#, last_ep2, results2)

# Trace de la precision
def plot_accuracy(resu1, folds_number, losscriterion, lr, bs, wdecay, datasetname, figname, pref, sfx, startep1, lastep1,
                  startep2=0, lastep2=-2, resu2=[], startep3=0, lastep3=-2, resu3=[], startep4=0, lastep4=-2, resu4=[],
                  startep5=0, lastep5=-2, resu5=[]):
    somme_train = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5)+1))
    mean_train = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5)+1))

    if pref == 'revgrad':
        trainacc = 'train_acc_' + abr
        validacc = 'valid_acc_' + abr
    else:
        trainacc = 'train_acc'
        validacc = 'test_acc'  # valid_acc

    for i in range(lastep1 - startep1 + 1):
        somme_train[i] = 0
        for k in range(folds_number):
            somme_train[i] += resu1[k]['fold' + str(k)][trainacc][i]
        mean_train[i] = somme_train[i] / folds_number
    """
    for i in range(lastep2-startep2+1):
        somme_train[startep2 + i] = 0
        for k in range(folds_number):
            if k==0:
                somme_train[startep2 + i] += resu2[k]['fold' + str(k)][trainacc][i]
            else:
                somme_train[startep2 + i] += resu1[k]['fold' + str(k)][trainacc][startep2 + i]
        mean_train[startep2 + i] = somme_train[startep2 + i]/folds_number

    for i in range(lastep3-startep3+1):
        somme_train[startep3 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_train[startep3 + i] += resu3[k]['fold' + str(k)][trainacc][i]
            else:
                somme_train[startep3 + i] += resu1[k]['fold' + str(k)][trainacc][startep3 + i]
        mean_train[startep3 + i] = somme_train[startep3 + i]/folds_number

    for i in range(lastep4-startep4+1):
        somme_train[startep4 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_train[startep4 + i] += resu4[k]['fold' + str(k)][trainacc][i]
            else:
                somme_train[startep4 + i] += resu2[k]['fold' + str(k)][trainacc][i]
        mean_train[startep4 + i] = somme_train[startep4 + i]/folds_number

    for i in range(lastep5-startep5+1):
        somme_train[startep5 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_train[startep5 + i] += resu5[k]['fold' + str(k)][trainacc][i]
            else:
                somme_train[startep5 + i] += resu2[k]['fold' + str(k)][trainacc][startep5 - startep4 + i]
        mean_train[startep5 + i] = somme_train[startep5 + i]/folds_number
    """
    somme_test = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5)+1))
    mean_test = list(range(startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5)+1))

    for i in range(lastep1-startep1+1):
        somme_test[i] = 0
        for k in range(folds_number):
            somme_test[i] += resu1[k]['fold' + str(k)][validacc][i]
        mean_test[i] = somme_test[i] / folds_number
    """
    for i in range(lastep2-startep2+1):
        somme_test[startep2 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_test[startep2 + i] += resu2[k]['fold' + str(k)][validacc][i]
            else:
                somme_test[startep2 + i] += resu1[k]['fold' + str(k)][validacc][startep2 + i]
        mean_test[startep2 + i] = somme_test[startep2 + i]/folds_number

    for i in range(lastep3 - startep3 + 1):
        somme_test[startep3 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_test[startep3 + i] += resu3[k]['fold' + str(k)][validacc][i]
            else:
                somme_test[startep3 + i] += resu1[k]['fold' + str(k)][validacc][startep3 + i]
        mean_test[startep3 + i] = somme_test[startep3 + i] / folds_number

    for i in range(lastep4 - startep4 + 1):
        somme_test[startep4 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_test[startep4 + i] += resu4[k]['fold' + str(k)][validacc][i]
            else:
                somme_test[startep4 + i] += resu2[k]['fold' + str(k)][validacc][i]
        mean_test[startep4 + i] = somme_test[startep4 + i] / folds_number

    for i in range(lastep5 - startep5 + 1):
        somme_test[startep5 + i] = 0
        for k in range(folds_number):
            if k == 0:
                somme_test[startep5 + i] += resu5[k]['fold' + str(k)][validacc][i]
            else:
                somme_test[startep5 + i] += resu2[k]['fold' + str(k)][validacc][startep5 - startep4 + i]
        mean_test[startep5 + i] = somme_test[startep5 + i] / folds_number
    """
    if folds_number == 5:
        title = 'Mean of the 5 folds'
    else:
        title = 'Fold ' + str(folds_number-1)

    plt.rcParams["figure.figsize"] = [43.1, 31]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ftsz = 156
    linewd = 8
    ax.plot(mean_test, color='b', linewidth=linewd)
    ax.plot(mean_train, color='r', label=title, linewidth=linewd)

    ax.set_xlabel('Number of epochs', fontsize=ftsz)
    ax.set_ylabel('Accuracy', fontsize=ftsz)
    ax.tick_params(labelleft=True, labelbottom=True, size=28, width=linewd)
    ax.spines['bottom'].set_linewidth(linewd)
    ax.spines['top'].set_linewidth(linewd)
    ax.spines['left'].set_linewidth(linewd)
    ax.spines['right'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-30)
    plt.yticks(fontsize=ftsz-30)
    
    #ax.set_ylim([0.56, 0.94])
    """
    ax.legend()
    fig.text(0.5, 0.99, '%s' % figname, ha='center', va='top', size='large')
    if pref == 'revgrad':
        fig.text(0.02, 0.95, '%s, %s, %s \nlearning rate=%s, batch size=%s'
                 % (echeance, net_name, losscriterion, lr, bs), ha='left', va='top')
    else:
        fig.text(0.02, 0.95, '%s, %s \nlearning rate=%s, batch size=%s, w_decay=%s'
                 % (# echeance,
                     net_name, losscriterion, lr, bs, wdecay), ha='left', va='top')
    fig.text(0.01, 0.04, dataset_name, ha='left', size="small")
    fig.text(0.01, 0.01, sfx, ha='left', size="small")
    fig.text(0.9, 0.92, "training accuracy", ha="right", va="bottom", size="medium", color='r')
    fig.text(0.9, 0.885, "validation accuracy", ha="right", va="bottom", size="medium", color='b')
    """
    plt.tight_layout()
    if pref == 'revgrad':
        if sfx == '_':
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_acc.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4), bs, lr))
        else:
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_%s_acc.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4), bs, lr, sfx))
    else:
        if sfx == '_':
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_acc.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5), bs,
                           lr, wdecay))
        else:
            fig.savefig('./figures_generees/%s_%s_ep%s_%s_bs%s_lr%s_wd%s_%s_acc_250227.png'
                        % (pref, datasetname, startep1, max(lastep1, lastep2, lastep3, lastep4, lastep5), bs,
                           lr, wdecay, sfx), dpi=300)
            #plt.show()
    plt.close()


#plot_accuracy(results1, kfolds, criterion, learning_rate, batch_size, w_decay, dataset_name, fig_name, prefix, suffix,
#              start_ep1, last_ep1, start_ep2, last_ep2, results2, start_ep3, last_ep3, results3, start_ep4,
#              last_ep4, results4, start_ep5, last_ep5, results5)
#plot_accuracy(results1, kfolds, criterion, learning_rate, batch_size, w_decay, dataset_name, fig_name, prefix, suffix,
#           start_ep1, last_ep1, start_ep2, last_ep2, results2)#, start_ep3, 1499, results3)
#plot_accuracy(results1, kfolds, criterion, learning_rate, batch_size, w_decay, dataset_name, fig_name, prefix, suffix,
#           start_ep1, last_ep1)
