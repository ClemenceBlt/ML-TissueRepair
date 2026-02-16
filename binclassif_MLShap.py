import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import seaborn as sn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, accuracy_score,
                             classification_report)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.cluster import KMeans
import xgboost
import pickle
import shap


# Fichier pour entrainer XGBoost et faire tourner SHAP dessus pour la classification binaire.
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


# parametres :
jobid_toread = '230323_trainmosTV'
suffix = '_'
dataset_name = '230323_trainmosTV'
testset_name = '230323_trainmosTe'

if suffix == '_':
    prefix_name = 'binary_classif_ML_%s' % dataset_name  #, jobid_toread)
else:
    prefix_name = 'binary_classif_ML_%s_%s_%s' % (dataset_name, jobid_toread, suffix)

#npy_path_Xtrainfiltered = './npy_files/Xtrainfiltered_binary_classif_ML_%s_%s.npy' % (dataset_name, jobid_toread)
npy_path_Xtrainfiltered = './npy_files/Xfiltered_binary_classif_ML_%s_0.npy' % dataset_name
#npy_path_Xtestfiltered = './npy_files/Xtestfiltered_binary_classif_ML_%s_%s.npy' % (testset_name, testset_name)
npy_path_Xtestfiltered = './npy_files/Xtestfiltered_binary_classif_ML_%s_1173408.npy' % jobid_toread
#npy_path_ytrain = './npy_files/ytrain_binary_classif_ML_%s_%s.npy' % (dataset_name, jobid_toread)
npy_path_ytrain = './npy_files/ytrain_binary_classif_ML_%s_0.npy' % jobid_toread
#npy_path_ytest = './npy_files/ytest_binary_classif_ML_%s_%s.npy' % (testset_name, testset_name)
npy_path_ytest = './npy_files/ytest_binary_classif_ML_%s_1173408.npy' % jobid_toread
npy_path_featmask = './npy_files/featmask_binary_classif_ML_230323_trainmosTV_0.npy'
#pkl_path_model_gpu = './pkl_files/trainedXGBoost_osirimgpu_binary_classif_ML_%s_%s.pkl' % (jobid_toread[:-2], jobid_toread)
pkl_path_model_gpu = './pkl_files/trainedXGBoost_calmipgpu_%s_0.pkl' % prefix_name
npy_path_background = './npy_files/background_%s.npy' % prefix_name
npy_path_shapvalues = './npy_files/SHAPvalues_%s.npy' % prefix_name

f = open('./text_files/%s_plotshap.txt' % prefix_name, 'w')
# Adapt the suffixe to the current task: xgboost, matconf, saveshap, plotshap.

f.write('binclassif_MLShap.py lance depuis run_binclassif_MLShap.sh \n')
print('binclassif_MLShap.py lance depuis run_binclassif_MLShap.sh')

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
print('Train features loaded from', npy_path_ytrain)
print('Len loaded y_train:', len(ytrain))
print("Passage j03 nalox a ptlip pour train :", ytrain[18624:18628])
# ytrain = [0,...,0,1,...,1,0,...,0,1,...,1] car alternance j03 nalox, j03 ptlip, j10 nalox et j10 ptlip
f.write('Train features loaded from {val} \n'.format(val=npy_path_ytrain))
f.write('Len loaded y_train: {val} \n'.format(val=len(ytrain)))
f.write('Train set\'s transition from D3 nalox to small lipectomy: {val} \n'.format(val=ytrain[18624:18628]))

# Recuperation tableau ytest
ytest = np.load(npy_path_ytest)
print('Test features loaded from', npy_path_ytest)
print('Len loaded y_test:', len(ytest))
print("Passage j03 nalox a ptlip pour test :", ytest[3098:3102])
f.write('Test features loaded from {val} \n'.format(val=npy_path_ytest))
f.write('Len loaded y_test: {val} \n'.format(val=len(ytest)))
f.write('Test set\'s transition from D3 nalox to small lipectomy: {val} \n'.format(val=ytest[3098:3102]))

# Load names of all extracted features
pklfile_names = './pkl_files/pyfeats_features_names_230623.pkl'
with open(pklfile_names, 'rb') as gg:
    features_names = pickle.load(gg)
features_names = np.array(features_names)  # sinon le [mask] ne fonctionne pas
print('Features names loaded from', pklfile_names)
print('First 5 features names:', features_names[:5])
print('Number of features:', len(features_names))
f.write('Features names loaded from {val} \n'.format(val=pklfile_names))
f.write('First 5 features names: {val} \n'.format(val=features_names[:5]))
f.write('Number of features: {val} \n'.format(val=len(features_names)))

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
                  'num_class': [2],
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

param = {'booster': 'gbtree', 'gamma': 0.0, 'learning_rate': 0.5, 'max_depth': 6, 'n_estimators': 100,
         'num_class': 2, 'objective': 'multi:softmax', 'scale_pos_weight': 0.2, 'subsample': 1.0, 'verbosity': 0}
dtrain = xgboost.DMatrix(Xtrainfiltered, ytrain)
# DMatrix = a data structure more memory efficient and faster than the usual Pandas dataframe
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
# model = model.best_estimator_
"""

#"""
# Load saved trained model
with open(pkl_path_model_gpu, 'rb') as gg:
    model_loaded = pickle.load(gg)
print("XGBoost model successfully loaded from", pkl_path_model_gpu)
#"""

"""
dtrain = xgboost.DMatrix(Xtrainfiltered, ytrain) 
dtest = xgboost.DMatrix(Xtestfiltered, ytest)  
"""

"""
# Accuracy at the end of training
ypredtrain = model_loaded.predict(dtrain)
train_acc = accuracy_score(ytrain, ypredtrain)
# ytrain contient de 1 et de 0 et ypredtrain doit contenir la meme chose (et pas des probabilites) sinon le calcul
# de l'accuracy ne se fera pas.
print("Accuracy sur l'ensemble d'entrainement :", train_acc)
f.write('Accuracy sur l ensemble d entrainement : {val} \n'.format(val=train_acc))

# Accuracy on test set
ypred = model_loaded.predict(dtest)
test_acc = accuracy_score(ytest, ypred)
print("Accuracy sur l'ensemble de test :", test_acc)
f.write('Accuracy sur l ensemble de test : {val} \n'.format(val=test_acc))

report = classification_report(ytest, ypred)
# report = classification_report(ytrain, ypredtrain)
print(report)
"""

"""
# Confusion matrix
ftscale = 20
ftsz=250

array = confusion_matrix(ytrain, ypredtrain)
#array = confusion_matrix(ytest, ypred) 
#df_cm = pd.DataFrame(array, index=["naloxone", "small lipectomy"], columns=["naloxone", "small lipectomy"])
# df_cm = pd.DataFrame(array, index=["regenerating", "scarring"], columns=["regenerating", "scarring"])
df_cm = pd.DataFrame(array, index=["rgn", "scar."], columns=["rgn", "scar."])

sn.set_theme(font_scale=ftscale)  # taille du texte des axes
print(df_cm)

plt.rcParams["figure.figsize"] = [40, 31]
fig, ax = plt.subplots()
sn.heatmap(df_cm, cmap='Blues', annot=True, fmt='0.0f', annot_kws={"size": ftsz}, cbar=False, square=True)
# annot_kws : taille des chiffres dans la matrice, cbar : legende des couleurs
"""
"""
fig.text(0.5, 0.995, 'Confusion matrix - Binary classification with XGBoost', ha='center', va='top', size='medium')
fig.text(0.5, 0.945, prefix_name, ha='center', va='top', size='small')

if suffix != '_':
    fig.text(0.725, 0.88, suffix, ha='left', size="small")
fig.text(0.75, 0.5, report, wrap=True, ha='center', va='top', size='small')
fig.text(0.75, 0.7, "Accuracy:\n%0.5s" % test_acc, wrap=True, ha='center', va='top', size='medium')
# version avec texte
"""
"""
# fig.text(0.59, 0.55, "Accuracy:\n%0.5s" % test_acc, wrap=True, ha='center', va='top', size='medium')
fig.text(0.91, 0.705, "Acc:\n%s" % round(train_acc, 2), wrap=True, ha='center', va='top',
         size='medium')  # version sans texte  
# version sans texte

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

#fig.tight_layout(pad=0.5, rect=[-0.33, -0.02, 0.9, 0.9])  # pour la version avec le texte
# fig.tight_layout(pad=1.05, rect=[-0.42, -0.045, 1, 0.95])  # pour la version sans le texte
fig.tight_layout(pad=0.01, rect=[-0.13, 0, 1.0, 1.0])

plt.savefig('./figures_generees/%s_confmat_TV_250228.png' % prefix_name, dpi=300)  
#plt.show()
plt.close()
"""


# --- Explanation with SHAP ---

Xallfiltered = np.concatenate((Xtrainfiltered, Xtestfiltered), axis=0) 

"""
# Background definition with k-medoid calculation

startkmean = time.time()
print("Start k-means")
# kmed = KMedoids(n_clusters=1000, metric='euclidean', method='pam')
kmeans = KMeans(n_clusters=1000, n_init='auto')
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
"""

"""
# SHAP values with GPU
print("Start calculating shap values")
start1 = time.time()
shap_values = model_loaded.predict(dtrain, pred_contribs=True)
end1 = time.time()
print("Time for shap in min:", (end1-start1)/60)
"""

#"""
model_loaded.set_param({"device": "cuda"})

# Shap values calculation
# this is the code used when calculating SHAP values with GPU on calmip/osirim

# Explainer definition
explainer = shap.TreeExplainer(model_loaded)  #, data=background)
#"""
"""
# Shap values calculation
# Random Forest : shap_values = np array de dim (nb de classes (2), taille du dataset (lentest), nb de features (893))
# XGBoost : shap_values de dimension (taille dataset (97576), nb de features (911))

print("Start shap values calculation")
startshap = time.time()
shap_values = explainer.shap_values(Xallfiltered)  #, check_additivity=False)
endshap = time.time()
print("Time for shap in min:", (endshap-startshap)/60)
f.write("Time for shap in min: {val} \n".format(val=(endshap-startshap)/60))

# Save SHAP values
np.save(npy_path_shapvalues, shap_values)
print("SHAP values successfully saved in", npy_path_shapvalues)
f.write("SHAP values successfully saved in {val} \n".format(val=npy_path_shapvalues))
"""

#"""
# Load SHAP values
shap_values = np.load(npy_path_shapvalues)
print("SHAP values successfully loaded from", npy_path_shapvalues)
f.write("SHAP values successfully loaded from {val} \n".format(val=npy_path_shapvalues))

# visualize the first prediction's explanation
print("shap_values", type(shap_values), len(shap_values), len(shap_values[0]), len(shap_values[0][0]))
f.write("shap_values: {val1} ({val2}, {val3}, {val4}) \n".format(val1=type(shap_values), val2=len(shap_values),
                                                                 val3=len(shap_values[0]), val4=len(shap_values[0][0])))
# liste de dimension (2, 97576, 911) ie (nb de classes, nb d'images, nb de variables par image)
print("Xallfiltered", type(Xallfiltered), np.shape(Xallfiltered))
f.write("Xallfiltered: {val1} {val2} \n".format(val1=type(Xallfiltered), val2=np.shape(Xallfiltered)))
# array numpy de dimension (97576, 911) ie (nb d'images, nb de variables par image)
print("explainer.expected_value", type(explainer.expected_value), len(explainer.expected_value))
f.write("explainer.expected_value: {val1} {val2} \n".format(val1=type(explainer.expected_value),
                                                            val2=len(explainer.expected_value)))
# liste de longueur 2 (nb de classes : nalox et PL)
#"""

#"""
# Shap values plot
shap.summary_plot(shap_values[0], features=Xallfiltered, feature_names=selectedfeat_names, max_display=22,
                  show=False)

fig, ax = plt.gcf(), plt.gca()
plt.gcf().set_size_inches(6.5,7)
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
plt.savefig('./figures_generees/%s_SHAPlocal_binary_22_250312.png' % prefix_name, dpi=300)
plt.close()

#"""

f.close()
