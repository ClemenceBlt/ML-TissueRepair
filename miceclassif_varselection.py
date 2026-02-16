import numpy as np
import time
import sys
import argparse
import os
import pickle
import torch
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# from allclassif_iterativeFeatDel import jobid_toread

# Selection de variables avec Boruta pour la classification des souris (22) a partir des features des images d'un
# ensemble de donnees qu'on charge depuis des fichiers pickle (.pkl).
# Genere : - 1 fichier texte de resume de l'entrainement
#          - 3 fichiers npy contenant les valeurs des features selectionnees par Boruta (Xfiltered), les valeurs de
#          ytrain (labels) et le masque des features conserves (featmask).

parser = argparse.ArgumentParser()
# parser.add_argument('-jobid', '--jobid', default='240110_set4_miceTe', type=str)
parser.add_argument('-suf', '--suf', default='_', type=str)
args = parser.parse_args()

big_start = time.time()

# parametres :
# jobid_toread = args.jobid
suffix = args.suf

torch.manual_seed(0)
# dossier d'images utilise :
dataset_name = '240110_set4_miceTe'  # nom du dataset
jobid_toread = dataset_name

if suffix == '_':
    prefix_name = 'miceclassif_ML_%s' % dataset_name  # ou % (dataset_name, jobid_toread) le cas echeant)
else:
    prefix_name = 'miceclassif_ML_%s_%s_%s' % (dataset_name, suffix)

"""
# A commenter pour la creation de Xtestfiltered pour laquelle les noms des chemins seront definis plus bas
npy_path_Xfiltered = './npy_files/Xtrainfiltered_%s.npy' % prefix_name
npy_path_ytrain = './npy_files/ytrain_%s.npy' % prefix_name
npy_path_featmask = './npy_files/featmask_%s.npy' % prefix_name
"""
f = open('./text_files/%s.txt' % prefix_name, 'w')

print('miceclassif_varselection.py')
f.write('miceclassif_varselection.py\n')

print('used environment:', sys.prefix)
f.write('used environment: {val} \n'.format(val=sys.prefix))

print('parameters', args)
f.write('parameters: {val} \n'.format(val=args))

print('Dataset name:', dataset_name)
f.write('Dataset name: {val} \n'.format(val=dataset_name))


# -- Creation des listes X_train (features des images) et y_train (labels des images : numero de souris) --
"""
# Extraction des noms de features a commenter pour la generation de Xtest pour laquelle les noms sont inutiles
# Pickle file loading

# First, extract only names of extracted features
pklfile_names = './pyfeats_features_names_230623.pkl'
with open(pklfile_names, 'rb') as gg:
    features_names = pickle.load(gg)

# print('Features names:', features_names)
# f.write('Features names: {val} \n'.format(val=features_names))
print('Number of features:', len(features_names))
f.write('Number of features: {val} \n'.format(val=len(features_names)))
"""


# Then extract the features

mice_numbers = ["nalox_j03_17", "nalox_j03_19_OD", "nalox_j03_19_OG", "nalox_j03_20", "nalox_j03_21",
                "nalox_j03_30_ODD", "ptlip_j03_20", "ptlip_j03_21", "ptlip_j03_26", "ptlip_j03_28", "nalox_j10_19_OG",
                "nalox_j10_20", "nalox_j10_22_OG", "nalox_j10_27_ODD", "nalox_j10_27_OGG", "nalox_j10_30_ODD",
                "ptlip_j10_20_OG", "ptlip_j10_25_OD", "ptlip_j10_25_OG", "ptlip_j10_27", "ptlip_j10_27_OD",
                "ptlip_j10_27_OGG"]
# version avec 22 souris


def extract_features(jobidtoread, mice_numbers_tab):
    '''
    Fonction pour recuperer l'information contenue dans les fichiers pkl et les utiliser dans le programme en cours.
    Parcourt le dossier contenant tous les fichiers pkl et met les informations dans des tableaux.
    :param jobidtoread: accole a la fin de output_chdb_, permet de savoir de quel dossier de fichiers pkl partir
    :param mice_numbers_tab: tableau dont la valeur a l'indice i est le nom de la souris n°i
    :return: Xtrain = matrice qui a la ligne i contient les features de l'image i
             ytrain = liste qui contient les labels des images (numero de la souris)
    '''
    Xtrain = []
    ytrain = []

    for echeance in ['j03', 'j10']:
        for classe in ['nalox', 'ptlip']:
            img_tiles = os.listdir(os.path.join('./pkl_files_outputchdb/output_chdb_%s' % jobidtoread, echeance, classe))
            for img_tile_name in img_tiles:
                # print(img_tile_name)
                for ch in ['ch1', 'ch2']:
                    list_pkl_files = os.listdir(os.path.join('./pkl_files_outputchdb/output_chdb_%s' % jobidtoread,
                                                             echeance, classe, img_tile_name, ch))
                    for pklfile in list_pkl_files:
                        pklfile_path = os.path.join('./pkl_files_outputchdb/output_chdb_%s' % jobidtoread, echeance,
                                                    classe, img_tile_name, ch, pklfile)
                        with open(pklfile_path, 'rb') as g:
                            train_features_i = pickle.load(g)
                        img_name = pklfile[9:-4]
                        # on enleve le mot features au debut et le .pkl a la fin
                        # ex : nalox_j03_ztt_manip21_ODD_mos225_tile180_ch1-0074
                        # ou   nalox_j03_ztt_manip30_ODD_tile2x7_tile8 - Ch2 - C2 Z9 T1

                        Xtrain.append(train_features_i[img_name])

                        # Determination du numero de souris en fonction du nom du fichier courant
                        mice_ear = img_name[22:26]  # lettres de l'oreille en incluant l'underscore d'apres
                        pos_underscore = mice_ear.find('_')  # permet de savoir si 2 ou 3 lettres pour l'oreille
                        mice_nb = img_name[19:22 + pos_underscore]
                        label_str = img_name[:10] + mice_nb
                        # label_str vaut par exemple nalox_j03_19_OD. Ce sont les prefixes du tableau mice_numbers

                        # Gestion des labels qui sont differents, pour une meme souris, entre l'image basique et la
                        # mosaique
                        if label_str == "nalox_j03_17_OD":
                            label_str = "nalox_j03_17"
                        elif label_str == "nalox_j03_20_OG":
                            label_str = "nalox_j03_20"
                        elif label_str == "nalox_j03_21_ODD":
                            label_str = "nalox_j03_21"
                        elif label_str == "ptlip_j03_28_OG":
                            label_str = "ptlip_j03_28"

                        if label_str not in mice_numbers_tab:
                            print("Label pas dans ma liste !")

                        label_int = [i for i, x in enumerate(mice_numbers_tab) if x == label_str][0]
                        # numero de la souris en cours, selon le tableau mice_numbers_tab

                        ytrain.append(label_int)

    return Xtrain, ytrain


X_train, y_train = extract_features(jobid_toread, mice_numbers)
# a noter que la variable X_train vaut en fait X_test quand on fait tourner ce code pour generer Xtestfiltered
# (le jobid_toread renvoie en effet alors vers le dossier des images de test)
imgnb = len(X_train)
print('len X:', imgnb)
f.write('len X: {val} \n'.format(val=imgnb))
print('len y :', len(y_train))
f.write('len y: {val1} \n'.format(val1=len(y_train)))


"""
# -- Feature importance with Boruta --

# Random forest classifier:
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=3)#, verbose=3)
# contrairement a ce qui est suggere dans la doc, changer max_depth ne modifie pas significativement le nb de features
# selectionnes
print('Definition of random forest classification function done.')

# Boruta feature selection method:
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=3, random_state=1)
# modifier max_iter de 100 a 200 donne des resultats semblables (sur 1990 images)
print('Definition of feature selection function with Boruta done.')

# Find all relevant features:
X_train_array = np.array(X_train)
y_train_array = np.array(y_train)
print('Starting feature selection with Boruta.')
feat_selector.fit(X_train_array, y_train_array)
print('Feature selection with Boruta done.')

# Check selected features:
selected_features = feat_selector.support_
print('Selected features (first 10 candidates):', selected_features[:10])
f.write('Selected features (first 10 candidates): {val} \n'.format(val=selected_features[:10]))

# Check ranking of features
selector_ranking = feat_selector.ranking_
print('Feature ranking (first 10 candidates):', selector_ranking[:10])
f.write('Feature ranking (first 10 candidates): {val} \n'.format(val=selector_ranking[:10]))

nb_selected_feat = feat_selector.n_features_
print('Number of selected features:', nb_selected_feat)
f.write('Number of selected features: {val} \n'.format(val=nb_selected_feat))

# Print support and ranking for each feature
# (from https://towardsdatascience.com/feature-selection-with-boruta-in-python-676e3877e596)
print("\n--- Support and Ranking for each feature ---")
f.write('\n--- Support and Ranking for each feature ---\n')
for i in range(len(selected_features)):
    if selected_features[i]:
        print("Passes the test:", features_names[i], " - Ranking:", selector_ranking[i])
        f.write('Passes the test: {val1} - Ranking: {val2}. \n'.format(val1=features_names[i],
                                                                       val2=selector_ranking[i]))
    # else a commenter si on ne veut pas avoir trop de messages
    '''
    else:
        print("Does not pass the test:", features_names[i], " - Ranking:", feat_selector.ranking_[i])
        f.write('Does not pass the test: {val1} - Ranking: {val2}. \n'.format(val1=features_names[i],
                                                                              val2=feat_selector.ranking_[i]))
    '''
    

# Call transform to filter it down to selected features
X_train_filtered = feat_selector.transform(X_train_array)
print('Len X train filtered:', len(X_train_filtered))
print('Len X train filtered [0]:', len(X_train_filtered[0]))
f.write('Len X train filtered: {val} \n'.format(val=len(X_train_filtered)))
f.write('Len X train filtered [0]: {val} \n'.format(val=len(X_train_filtered[0])))

# Enregistrement X_train_filtered, y_train et selected_features
np.save(npy_path_Xfiltered, X_train_filtered)
print('X_train_filtered saved in %s' % npy_path_Xfiltered)
f.write('X_train_filtered saved in {val} \n'.format(val=npy_path_Xfiltered))

np.save(npy_path_ytrain, y_train)
print('y_train saved in %s' % npy_path_ytrain)
f.write('y_train saved in {val} \n'.format(val=npy_path_ytrain))

np.save(npy_path_featmask, selected_features)
print('selected_features saved in %s' % npy_path_featmask)
f.write('selected_features saved in {val} \n'.format(val=npy_path_featmask))
"""

#"""
# --- Partie pour Xtestfiltered ---

npy_path_Xtestfiltered = './npy_files/Xtestfiltered_%s.npy' % prefix_name
npy_path_ytest = './npy_files/ytest_%s.npy' % prefix_name  # ytest par defaut
npy_path_featmask = './npy_files/featmask_miceclassif_ML_230330_22miceTV_1141510.npy'
# le featmask a ete cree precedemment par l'application de ce meme programme

mask_selected_feat = np.load(npy_path_featmask)  # chargement du masque des features

# Comptage du nombre de features selectionnes par Boruta :
unique, counts = np.unique(mask_selected_feat, return_counts=True)
featnb = dict(zip(unique, counts))[True]
print("Number of selected features:", featnb)
f.write('Number of selected features: {val} \n'.format(val=featnb))

X_test_filtered = np.empty((imgnb, featnb))
# de dimension nb d'images x nb de features apres selection par Boruta
for i in range(imgnb):  # parcours des lignes (images)
    jj = 0  # colonnes de X_test_filtered qui se remplissent progressivement
    for j in range(len(mask_selected_feat)):  # parcours des colonnes (features)
        if mask_selected_feat[j]:
            X_test_filtered[i][jj] = X_train[i][j]  # ajout a X_test_filtered seulement si le feature est selectionne
            jj += 1
        # on ne prend de X_train que les features qui ont ete selectionnes
        # a noter que la variable X_train vaut en fait X_test quand on fait tourner ce code pour generer Xtestfiltered

# Enregistrement X_test_filtered
np.save(npy_path_Xtestfiltered, X_test_filtered)
print('Dimensions of X_test_filtered:', np.shape(X_test_filtered))
f.write('Dimensions of X_test_filtered: {val} \n'.format(val=np.shape(X_test_filtered)))
print('X_test_filtered saved in %s' % npy_path_Xtestfiltered)
f.write('X_test_filtered saved in {val} \n'.format(val=npy_path_Xtestfiltered))

# Enregistrement de ytest
np.save(npy_path_ytest, y_train)
print('y_test saved in %s' % npy_path_ytest)
f.write('y_test saved in {val} \n'.format(val=npy_path_ytest))
#"""


big_end = time.time()

print('Total time in s.:', big_end - big_start)
print('Total time in min:', (big_end - big_start)/60)
print('Total time in hours:', (big_end - big_start)/3600)
f.write('Total time in s.: {val:0.3f} \n'.format(val=big_end - big_start))
f.write('Total time in min: {val:0.3f} \n'.format(val=(big_end - big_start) / 60))
f.write('Total time in hours: {val:0.3f} \n'.format(val=(big_end - big_start) / 3600))

f.close()
