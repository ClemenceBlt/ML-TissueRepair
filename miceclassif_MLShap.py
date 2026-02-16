import numpy as np
import matplotlib.pyplot as plt
import time

import seaborn as sn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, classification_report)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.cluster import KMeans

import pickle
import xgboost
import shap


# Fichier pour entrainer XGBoost et faire tourner SHAP dessus pour la classification des souris.
# A utiliser sur calmip/osirim et avec GPU.
# Processus se fait en trois temps : d'abord on entraine avec XGBoost (en enregistrant le modele entraine), puis on
# utilise ce modele enregistre pour calculer les SHAP et enfin on trace les SHAP prealablement enregistres. En plus de
# ces trois etapes, on peut en ajouter une quatrieme avant XGBoost de gridsearch pour trouver les meilleurs parametres.

# Fichiers qui doivent deja exister pour que ce programme fonctionne :
# - Xfiltered[...].npy : matrice qui contient pour chaque image de l'ensemble d'entrainement les valeurs de chaque
# feature selectionne par Boruta
# - Xtestfiltered[...].npy : matrice pareille que ci-dessus mais pour l'ensemble de test
# - ytrain[...].npy : vecteur qui contient les labels des images d'entrainement (numeros des souris)
# - ytest[...].npy : vecteur qui contient les labels des images de test
# - pyfeats_features_names_230623.pkl : noms des features extraits par PyFeats
# - featmask_miceclassif_ML[...].npy : masque des variables selectionnees.


# parametres :
suffix = '_'
dataset_name = '240110_set1_miceTV'
dataset_name_pref = dataset_name[:-2]  # sans le TV/Te final

if suffix == '_':
    prefix_name = 'miceclassif_ML_%s' % dataset_name
else:
    prefix_name = 'miceclassif_ML_%s_%s' % (dataset_name, suffix)

npy_path_Xtrainfiltered = './npy_files/Xtrainfiltered_%s.npy' % prefix_name
npy_path_Xtestfiltered = './npy_files/Xtestfiltered_%sTe.npy' % (prefix_name[:-2])
npy_path_ytrain = './npy_files/ytrain_%s.npy' % (prefix_name)
npy_path_ytest = './npy_files/ytest_%sTe.npy' % (prefix_name[:-2])
# npy_path_featmask = './npy_files/featmask_%s.npy' % prefix_name
npy_path_featmask = './npy_files/featmask_miceclassif_ML_230330_22miceTV_1141510.npy'
# pkl_path_model = './pkl_files/trainedXGBoost_calmip_%s.pkl' % prefix_name
pkl_path_model_gpu = './pkl_files/trainedXGBoost_occidatagpu_%s.pkl' % (prefix_name)
# chemin ou on enregistre le modele XGBoost entraine sur calmip avec GPU
npy_path_background = './npy_files/background_%s.npy' % prefix_name
npy_path_shapvalues = './npy_files/SHAPvalues_%s_%s.npy' % (prefix_name[:-2], dataset_name)

f = open('./text_files/%s_matconf.txt' % prefix_name, 'w')
# Adapt the suffixe to the current task: xgboost, matconf, saveshap, plotshap.

f.write('miceclassif_MLShap.py lance depuis (osirim_)run_miceclassif_MLShap.sh \n')
print('miceclassif_MLShap.py lance depuis (osirim_)run_miceclassif_MLShap.sh')

# Recuperation tableau Xtrainfiltered
Xtrainfiltered = np.load(npy_path_Xtrainfiltered)
print('Train features loaded from', npy_path_Xtrainfiltered)
print('Len loaded X train filtered:', len(Xtrainfiltered))
print('Number of extracted features:', len(Xtrainfiltered[0]))
print("Head of Xtrainfiltered:", Xtrainfiltered[0][:5])

f.write('Train features loaded from {val} \n'.format(val=npy_path_Xtrainfiltered))
f.write('Len loaded X train filtered: {val} \n'.format(val=len(Xtrainfiltered)))
f.write('Number of extracted features: {val} \n'.format(val=len(Xtrainfiltered[0])))
f.write('Head of Xtrainfiltered: {val} \n'.format(val=Xtrainfiltered[0][:5]))

# Recuperation tableau Xtestfiltered
Xtestfiltered = np.load(npy_path_Xtestfiltered)
print('Test features loaded from', npy_path_Xtestfiltered)
print('Len loaded X test filtered:', len(Xtestfiltered))
print('Number of extracted features:', len(Xtestfiltered[0]))
print("Head of Xtestfiltered:", Xtestfiltered[0][:5])

f.write('Test features loaded from {val} \n'.format(val=npy_path_Xtestfiltered))
f.write('Len loaded X test filtered: {val} \n'.format(val=len(Xtestfiltered)))
f.write('Number of extracted features: {val} \n'.format(val=len(Xtestfiltered[0])))
f.write('Head of Xtestfiltered: {val} \n'.format(val=Xtestfiltered[0][:5]))

# Recuperation tableau ytrain
ytrain = np.load(npy_path_ytrain)
print('Train features labels loaded from', npy_path_ytrain)
print('Len loaded y_train:', len(ytrain))
print('Train set\'s transition from D3 nalox to small lipectomy:', ytrain[19720:19728])
f.write('Train features labels loaded from {val} \n'.format(val=npy_path_ytrain))
f.write('Len loaded y_train: {val} \n'.format(val=len(ytrain)))
f.write('Train set\'s transition from D3 nalox to small lipectomy: {val} \n'.format(val=ytrain[19720:19728]))

# Recuperation tableau ytest
ytest = np.load(npy_path_ytest)
print('Test features labels loaded from', npy_path_ytest)
print('Len loaded y_test:', len(ytest))
print('Test set\'s transition from D3 nalox to small lipectomy:', ytest[1542:1550])
f.write('Test features labels loaded from {val} \n'.format(val=npy_path_ytest))
f.write('Len loaded y_test: {val} \n'.format(val=len(ytest)))
f.write('Test set\'s transition from D3 nalox to small lipectomy: {val} \n'.format(val=ytest[1542:1550]))

# Load names of all extracted features
pklfile_names = './pkl_files/pyfeats_features_names_230623.pkl'
with open(pklfile_names, 'rb') as gg:
    features_names = pickle.load(gg)
features_names = np.array(features_names)  # sinon le [mask] ne fonctionne pas
print('Features names loaded from', pklfile_names)
print('First 5 features names:', features_names[:5])
print('Total number of features:', len(features_names))
f.write('Features names loaded from {val} \n'.format(val=pklfile_names))
f.write('First 5 features names: {val} \n'.format(val=features_names[:5]))
f.write('Total number of features: {val} \n'.format(val=len(features_names)))

# Load mask of selected features and create list of selected features names
featmask = np.load(npy_path_featmask)
print('Mask of selected features loaded from', npy_path_featmask)
f.write('Mask of selected features loaded from {val} \n'.format(val=npy_path_featmask))
unique, counts = np.unique(featmask, return_counts=True)
featnb = dict(zip(unique, counts))[True]
print("Number of selected features:", featnb)
f.write('Number of selected features: {val} \n'.format(val=featnb))
selectedfeat_names = features_names[featmask]
# print("Size of selectedfeat_names:", len(selectedfeat_names))



# --- Training with XGBoost: gridsearch for the parameters ---
"""
# Range of parameters
params_halving = {'booster': ['gbtree'],
                  'objective': ['multi:softmax'],
                  'verbosity': [0],
                  'n_estimators': [5, 100],
                  'max_depth': [2, 6],
                  'learning_rate': [0.01, 0.5],
                  'gamma': [0.0, 0.9],
                  'subsample': [0.1, 1.0],
                  'scale_pos_weight': [0.2, 5]
                  # , 'use_label_encoder': [False]
                  }
scorer = make_scorer(accuracy_score)  # choix metrique pour le gridsearch
print("Grid search function definition")
model = HalvingGridSearchCV(xgboost.XGBClassifier(), params_halving, scoring=scorer, refit=scorer, cv=5, verbose=1)
"""

"""
# Training with XGBoost with GPU acceleration
startxg_gpu = time.time()

param = {"num_class": 22, "booster": 'gbtree', "gamma": 0.0, "learning_rate": 0.5, "max_depth": 6, "n_estimators": 100,
         "objective": 'multi:softmax', "scale_pos_weight": 0.2, "subsample": 1.0, "verbosity": 0}
dtrain = xgboost.DMatrix(Xtrainfiltered, ytrain)
print("Start training XGBoost using Xtrainfiltered")  
model = xgboost.train(param, dtrain, num_boost_round=500)

endxg_gpu = time.time()
print("XGBoost with GPU training time (min):", (endxg_gpu-startxg_gpu)/60)
f.write('XGBoost with GPU training time (min): {val} \n'.format(val=(endxg_gpu-startxg_gpu)/60))

# Save trained model to pickle
with open(pkl_path_model_gpu, 'wb') as ff:
    pickle.dump(model, ff)

print("XGBoost model saved in", pkl_path_model_gpu)
f.write('XGBoost model saved in {val} \n'.format(val=pkl_path_model_gpu))
"""
"""
# Gridsearch: get parameters and model
params = model.best_params_
print("Best parameters according to GridSearch:", params)
f.write("Best parameters according to GridSearch: {val} \n".format(val=params))
model = model.best_estimator_
"""

#"""
# Load saved trained model
with open(pkl_path_model_gpu, 'rb') as gg:
    model_loaded = pickle.load(gg)
print("XGBoost model successfully loaded from", pkl_path_model_gpu)
f.write("XGBoost model successfully loaded from {val} \n".format(val=pkl_path_model_gpu))
#"""

#"""
dtrain = xgboost.DMatrix(Xtrainfiltered, ytrain) 
dtest = xgboost.DMatrix(Xtestfiltered, ytest)  
#"""

#"""
# Accuracy at the end of training
ypredtrain = model_loaded.predict(dtrain)
train_acc = accuracy_score(ytrain, ypredtrain)
print("Accuracy sur l'ensemble d'entrainement :", train_acc)
f.write('Accuracy sur l ensemble d entrainement : {val} \n'.format(val=train_acc))

# Accuracy on test set
ypred = model_loaded.predict(dtest)
test_acc = accuracy_score(ytest, ypred)
print("Accuracy sur l'ensemble de test :", test_acc)
f.write('Accuracy sur l ensemble de test : {val} \n'.format(val=test_acc))
#"""

#"""
# Confusion matrix
#set_confmat = "train"
set_confmat = "test"  
ftscale = 11

if set_confmat=="train":
    array = confusion_matrix(ytrain, ypredtrain)
    accuracy = train_acc
    suffixname = "TV"
    ftsz=48
    #nbsize = 6.5 
else:
    array = confusion_matrix(ytest, ypred)
    accuracy = test_acc
    suffixname = "Te"
    ftsz=65
    #nbsize = 8.5

#print(array.tolist())
df_cm = pd.DataFrame(array)
sn.set_theme(font_scale=ftscale)  # taille du texte des axes
# print(df_cm)

plt.rcParams["figure.figsize"] = [34.5, 31]
fig, ax = plt.subplots()
sn.heatmap(df_cm, cmap='Blues', annot=False, fmt='0.0f', annot_kws={"size": ftsz}, cbar=False, square=True,
           xticklabels=True, yticklabels=True)
# annot_kws : taille des chiffres dans la matrice, cbar : legende des couleurs
#"""
"""
fig.text(0.5, 0.995, 'Confusion matrix - Mice classification with XGBoost', ha='center', va='top', size='medium')
fig.text(0.5, 0.945, prefix_name, ha='center', va='top', size='small')
"""
#"""
#fig.text(0.90, 0.55, "Accuracy:\n%0.5s" % accuracy, wrap=True, ha='center', va='top', size='medium')
#fig.text(0.944, 0.625, "Acc:\n%s" % round(accuracy, 2), wrap=True, ha='center', va='top', size='medium')  # version sans texte

#ax.set_xlabel("Predicted label")
#ax.set_ylabel("True label")
fig.tight_layout(pad=0.01, rect=[-0.063, 0, 1.0, 1.0])
plt.savefig('./figures_generees/%s_confmat_%s_sansnum.png' % (prefix_name, suffixname), dpi=300)
#plt.show()
#plt.close()

report = classification_report(ytest, ypred)
print(report)
#"""


# --- Explanation with SHAP ---

Xallfiltered = np.concatenate((Xtrainfiltered, Xtestfiltered), axis=0) 

"""
# Background definition with k-medoid calculation

startkmean = time.time()
print("Start k-means")
kmeans = KMeans(n_clusters=1000, n_init='auto')
# n_init pose probleme sur calmip mais ok en local (pb de version de scikit-learn)
# function to calculate the k-means

# rng = np.random.default_rng()  # noyau aleatoire different a chaque fois car on ne donne pas de seed
# Xallfiltered_sample = rng.choice(Xallfiltered, size=50000, replace=False, axis=0)
# selection aleatoire d'un certain nombre de lignes de la matrice donnee en argument
# on fait tourner les k-medoides sur une selection aleatoire de 54000 images parmi les 88054 car au-dela de 54000
# ca demande trop de memoire si en local

clusters = kmeans.fit(Xallfiltered)  # kmed.fit(Xallfiltered_sample)
# k-means fitted to the data

endkmean = time.time()
print("Total time needed for k-means (min):", (endkmean-startkmean)/60)
f.write("Total time needed for k-means (min): {val} \n".format(val=(endkmean-startkmean)/60))

kmeans_clusters = np.array(clusters.cluster_centers_)

# From k-means to k-medoids
kmedoid_clusters = np.empty((np.shape(kmeans_clusters)))
# tableau pour accueillir les centres des k-medoides (qu'on approxime avec les elements les plus proches des centres
# des k-means)
# Boucle pour trouver pour chaque centre de cluster de k-means, l'element des donnees (de la matrice Xallfiltered) le
# plus proche
print("Approximation de k-means vers k-medoid")
for i in range(len(kmeans_clusters)):  # parcours des clusters (lignes)
    print(i)
    distmin = np.linalg.norm(kmeans_clusters[i] - Xallfiltered[0])
    elmtmin = Xallfiltered[0]
    for j in range(1, len(Xallfiltered)):  # parcours des images
        dist = np.linalg.norm(kmeans_clusters[i] - Xallfiltered[j])
        if dist < distmin:
            distmin = dist
            elmtmin = Xallfiltered[j]
    kmedoid_clusters[i] = elmtmin

background = kmedoid_clusters
# on prend comme background les centres des k-medoides
print("Cluster centers shape:", np.shape(background))
f.write("Cluster centers shape: {val} \n".format(val=np.shape(background)))


# Saving background to npy file
np.save(npy_path_background, background)
print('background saved in %s' % npy_path_background)
f.write('background saved in {val} \n'.format(val=npy_path_background))
"""

"""
# Loading the saved background
background_loaded = np.load(npy_path_background)
print('Background loaded from', npy_path_background)
"""
"""
# SHAP values with GPU
print("Start calculating shap values")
start1 = time.time()
shap_values = model_loaded.predict(dtrain, pred_contribs=True)
end1 = time.time()
print("Time for shap in min:", (end1-start1)/60)
"""

"""
model_loaded.set_param({"device": "cuda"})

# Shap values calculation
# this is the code used when calculating SHAP values with GPU on calmip/osirim
explainer = shap.TreeExplainer(model_loaded)  #, data=background_loaded)
"""
"""
print("Start calculating shap values")
start1 = time.time()
shap_values = explainer.shap_values(Xallfiltered)  #, check_additivity=False)
end1 = time.time()
print("Time for shap in min:", (end1-start1)/60)
f.write("Time for shap in min: {val} \n".format(val=(end1-start1)/60))

# Save SHAP values
np.save(npy_path_shapvalues, shap_values)
print("SHAP values successfully saved in", npy_path_shapvalues)
f.write("SHAP values successfully saved in {val} \n".format(val=npy_path_shapvalues))
"""

""""
# Load SHAP values
shap_values = np.load(npy_path_shapvalues)
print("SHAP values successfully loaded from", npy_path_shapvalues)
f.write("SHAP values successfully loaded from {val} \n".format(val=npy_path_shapvalues))

# visualize the first prediction's explanation
print("shap_values", type(shap_values), len(shap_values), len(shap_values[0]), len(shap_values[0][0]))
f.write("shap_values: {val1} ({val2}, {val3}, {val4}) \n".format(val1=type(shap_values), val2=len(shap_values),
                                                                 val3=len(shap_values[0]), val4=len(shap_values[0][0])))
# liste de dimension (22, 93498, 1057) ie (nb de souris, nb d'images, nb de variables par image)
print("Xallfiltered", type(Xallfiltered), np.shape(Xallfiltered))
f.write("Xallfiltered: {val1} {val2} \n".format(val1=type(Xallfiltered), val2=np.shape(Xallfiltered)))
# array numpy de dimension (93498, 1057) ie (nb d'images, nb de variables par image)
print("explainer.expected_value", type(explainer.expected_value), len(explainer.expected_value))
f.write("explainer.expected_value: {val1} {val2} \n".format(val1=type(explainer.expected_value),
                                                            val2=len(explainer.expected_value)))
# liste de longueur 22 (nb de souris)
"""

"""
# Summary of feature importance
print("Start drawing plots")
for mice in range(22):
    shap.summary_plot(shap_values[mice], features=Xallfiltered, feature_names=selectedfeat_names, max_display=20,
                      show=False)  

    fig, ax = plt.gcf(), plt.gca()
    plt.gcf().set_size_inches(6.5,6)
    ftsz=15
    axvalues_ftsz=ftsz-4
    ticksz = 5
    linewd=1
    ax.set_xlabel("SHAP value", fontsize=ftsz)
    ax.spines['bottom'].set_linewidth(linewd)
    ax.tick_params(size=ticksz, width=linewd)
    plt.xticks(fontsize=axvalues_ftsz)
    # Color bar:
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=axvalues_ftsz)
    cb_ax.set_ylabel("Feature value", fontsize=ftsz)

    plt.tight_layout()
    plt.savefig('./figures_generees/%s_SHAPlocal_mice%s_250312.png' % (prefix_name, mice), dpi=300) 
    plt.close()
    print("Mouse", mice, "drawn")

print("Plots drawn")

# shap.dependence_plot("rank(0)", shap_values[mice], features=Xallfiltered, feature_names=selectedfeat_names,
#                      show=False)

"""
f.close()
