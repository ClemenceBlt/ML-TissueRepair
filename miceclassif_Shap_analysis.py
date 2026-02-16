import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Analyse des SHAP de classification des souris avec les deux metriques choisies : position moyenne d'un feature parmi
# les classements SHAP et nombre d'apparition d'un feature dans un top 20. Calcul et trace des graphes.

# Key name for all the files to load
prefix_name = 'miceclassif_ML_240110_mice_240110_miceTV'

# Useful files paths
# npy_path_featmask = './npy_files/featmask_%s.npy' % prefix_name  # mask of selected features
npy_path_featmask = './npy_files/featmask_miceclassif_ML_230330_22miceTV_1141510.npy'
npy_path_shapvalues = './npy_files/SHAPvalues_%s.npy' % prefix_name  # SHAP values
npy_path_matfeat = './npy_files/SHAPmatfeat_%s.npy' % prefix_name  # matrice that will be created here

# Load names of all extracted features
pklfile_names = './pkl_files/pyfeats_features_names_230623.pkl'
with open(pklfile_names, 'rb') as gg:
    features_names = pickle.load(gg)
features_names = np.array(features_names)

# Load mask of selected features and create list of selected features names
featmask = np.load(npy_path_featmask)
selectedfeat_names = features_names[featmask]
print("Number of selected features:", len(selectedfeat_names))

# Load SHAP values
shap_values = np.load(npy_path_shapvalues)
print("SHAP values successfully loaded from", npy_path_shapvalues)
print("shap_values", type(shap_values), len(shap_values), len(shap_values[0]), len(shap_values[0][0]))
# array numpy de dimension (22, 93498, 1057) ie (nb de souris, nb d'images, nb de variables par image)

# Pour trouver le numero d'un feature donne par son nom
# print(list(selectedfeat_names).index('LBP_R_1_P_8_energy'))
# print(selectedfeat_names[93])


def createmat_idxfeatures(shapvalues):
    '''
    Building a matrix with 22 lines (the number of mice). On each line, the list of the indices of the features ordered
    from the most important to the least important for prediction.
    :param shapvalues: shap values (array of size (22, 93498, 1057))
    :return: matrix of size 22x1057 with the features ordered by importance for each mouse
    '''
    nb_img = len(shapvalues[0])  # number of images (93498)
    nb_feat = len(shapvalues[0][0])  # number of features (1057)
    print("Nombre d'images, nombre de variables :", nb_img, nb_feat)
    matidx = []
    for micenb in range(22):  # parcours des souris
        print("Current mouse:", micenb)
        sumshap = []
        # liste qui contiendra, pour chaque feature, les sommes des valeurs absolues des shap values de ttes les images
        for j in range(nb_feat):  # parcours des features (colonnes)
            sommej = 0
            for i in range(nb_img):  # parcours des images (lignes)
                sommej += abs(shapvalues[micenb][i][j])
            sumshap.append(sommej)

        featidx_sorted = sorted(range(len(sumshap)), key=lambda x: sumshap[x])
        # ajouter [-20:] a la fin si on ne veut garder que le top 20
        # pour la souris en cours, liste des indices des features classes par sommes des shap values croissantes
        # (les premiers features sont ceux qui ont la plus petite somme, donc les moins importants pour la prediction)
        featidx_sorted.reverse()  # liste ordonnee du feature le plus important au moins important
        matidx.append(featidx_sorted)

    return np.array(matidx)


"""
matmainfeatures = createmat_idxfeatures(shap_values)

# Save matrix on local file
np.save(npy_path_matfeat, matmainfeatures)
"""

#"""
# Load matrix
shap_matrix = np.load(npy_path_matfeat)
print("SHAP values matrix successfully loaded from", npy_path_matfeat)
print("SHAP values matrix type, dim:", type(shap_matrix), np.shape(shap_matrix))
#"""


"""
# Checking that the matrix we just built corresponds to the SHAP plots
for mice in range(22):
    print("-- Mouse", mice, '--')
    for ii in range(20):
        print(selectedfeat_names_modif[shap_matrix[mice][ii]])
"""
"""
# Conversion en dataframe Panda
# en fait non utilise
dfmatfeat = pd.DataFrame(shap_matrix, index=[i for i in range(22)], columns=[j for j in range(1, 21)])
#print(dfmatfeat)
uniquefeat = pd.Series({i: dfmatfeat[i].unique() for i in dfmatfeat})
#print(uniquefeat[20])
"""


def dico_features(shapmatrix):
    '''
    Creates dictionnary with keys the features and values the list of positions achieved by this feature.
    :return: dictionnary of all features with list of positions
    '''
    dico_feat = {}
    nbmice = len(shapmatrix)
    nbfeat = len(shapmatrix[0])
    for i in range(nbmice):  # parcours des lignes (souris)
        if i not in {2, 5, 11, 18, 19}:  # on ne veut pas prendre en compte les SHAP des souris 2, 5, 11, 18 et 19 (car 240110_mice)
        # if i not in {2, 5, 11, 18, 19, 10, 12, 13, 14, 15, 16, 17, 20, 21}:
        # if i not in {2, 5, 11, 18, 19, 0, 1, 3, 4, 6, 7, 8, 9}:  # J10
        # if i in {0, 1, 3, 4, 16, 17, 20, 21}:  # J3NJ10PL
        # if i in {6, 7, 8, 9, 10, 12, 13, 14, 15}:  # J3PLJ10N
            for j in range(nbfeat):  # parcours des colonnes (features)
                featnb = shapmatrix[i][j]
                if featnb not in dico_feat:
                    dico_feat[featnb] = []
                    # creation nouvelle cle si on tombe sur un feature non encore rencontre
                    # chaque cle = liste des places
                dico_feat[featnb].append(j+1)
                # ajout de la position du feature (de 1 a 1057)

    return dico_feat


dicofeat = dico_features(shap_matrix)
# print("Nombre de features parmi les 1057 qui apparaissent dans au moins un top 20 des 22 souris :", len(dicofeat))
# pertinent seulement si on ne considere que les tops 20
# print("Features with list of positions:", dicofeat)


def top_n(lili, nn):
    '''
    Counts the number of elements of a list that are smaller than a given number.
    :param lili: list to check
    :param nn: threshold
    :return: number of elements of list lili that are <= nn
    '''
    cpt = 0
    for elmt in lili:
        if elmt <= nn:
            cpt += 1
    return cpt


def calc_occ_meanpos(dico_feat, selectedfeatnames, ntop):
    '''
    Calculation of the average position.
    :param dico_feat: dictionnary with all the features and their positions in the SHAP
    :param selectedfeatnames: names of the features
    :param ntop: top n considered
    :return: (dictionnary with keys the features numbers and values the number of top n where they appear,
              dictionnary with keys the features names and values the number of top n where they appear,
              dictionnary with keys the features numbers and values the average position,
              dictionnary with keys the features names and values the average position)
    '''
    dico_occ = {}
    dico_occ_names = {}
    dico_meanpos = {}
    dico_meanpos_names = {}
    for featnb in dico_feat:
        list_positions = dico_feat[featnb]  # liste des positions calculee par fonction dico_features
        lenlist = len(list_positions)
        nbintopn_nd3 = top_n(list_positions[:4], ntop)  # count for naloxone day 3
        nbintopn_pld3 = top_n(list_positions[4:8], ntop)  # count for small lipectomy day 3
        nbintopn_nd10 = top_n(list_positions[8:13], ntop)  # count for naloxone day 10
        nbintopn_pld10 = top_n(list_positions[13:], ntop)  # count for small lipectomy day 10
        nbintopn = nbintopn_nd3 + nbintopn_pld3 + nbintopn_nd10 + nbintopn_pld10
        # nbintopn = top_n(list_positions, ntop)  # le nombre de fois ou le feature est dans un top n

        dico_occ[featnb] = [nbintopn, nbintopn_nd3, nbintopn_pld3, nbintopn_nd10, nbintopn_pld10]
        dico_occ_names[selectedfeatnames[featnb]] = [nbintopn, nbintopn_nd3, nbintopn_pld3, nbintopn_nd10,
                                                     nbintopn_pld10]

        sumpos = sum(list_positions)
        dico_meanpos[featnb] = sumpos / lenlist  # position moyenne du feature courant
        dico_meanpos_names[selectedfeatnames[featnb]] = sumpos / lenlist

    return dico_occ, dico_occ_names, dico_meanpos, dico_meanpos_names


nfortop = 20

dicoocc, dicoocc_names, dicomeanpos, dicomeanpos_names = calc_occ_meanpos(dicofeat, selectedfeat_names, nfortop)

#"""
# Tri par ordre decroissant d'occurrences
dicoocc_sorted = dict(sorted(dicoocc.items(), key=lambda item: item[1], reverse=True))
dicooccnames_sorted = dict(sorted(dicoocc_names.items(), key=lambda item: item[1], reverse=True))
print("Features with number of occurrences (sorted): [total, nalox D3, small lip D3, nalox D10, small lip D10]")
# print(dicoocc_sorted)
print(dicooccnames_sorted)
# Tri par ordre croissant de position moyenne
dicomeanpos_sorted = dict(sorted(dicomeanpos.items(), key=lambda item: item[1]))
dicomeanposnames_sorted = dict(sorted(dicomeanpos_names.items(), key=lambda item: item[1]))
print("Features with average position (sorted):")
# print(dicomeanpos_sorted)
print(dicomeanposnames_sorted)
#"""

plt.rcParams["figure.figsize"] = [22, 16]
plt.rcParams["figure.autolayout"] = True


def plot_first_features(m, dicomeanposnames_s, dicooccnames_s, ntop):
    '''
    Plots average position and number of occurrences only of the first m features
    :param m: number of features to plot
    :param dicomeanposnames_s:
    :param dicooccnames_s:
    :param ntop: n choosen for top for occurrence number
    '''
    meanpos_toplot = {k: dicomeanposnames_s[k] for k in list(dicomeanposnames_s)[:m]}
    occurrences_toplot = {k: dicooccnames_s[k] for k in list(dicooccnames_s)[:m]}

    plt.rcParams["figure.figsize"] = [18.5, 11]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()

    ftsz=20.5
    axvalues_ftsz=16
    lgd_ftsz=18
    linewd = 1.5
    ticklen = 5

    ax1 = fig.add_subplot(1, 2, 1)
    plt.bar(range(len(meanpos_toplot)), list(meanpos_toplot.values()), align='center', color='tab:orange')
    plt.xticks(range(len(meanpos_toplot)), list(meanpos_toplot.keys()), rotation=90, ha='center')
    ax1.tick_params(axis='x', which='major', size=ticklen, width=linewd)
    ax1.tick_params(axis='y', which='major', size=ticklen, width=linewd)
    ax1.spines['bottom'].set_linewidth(linewd)
    ax1.spines['left'].set_linewidth(linewd)
    plt.xticks(fontsize=axvalues_ftsz)
    plt.yticks(fontsize=axvalues_ftsz)
    ax1.set_xlim(xmin=-0.5, xmax=m-0.5)
    ax1.set_ylim(ymin=1, ymax=48)
    plt.ylabel('Average position in the SHAP plots', fontsize=ftsz)
    #plt.ylabel("Position moyenne dans les classements de SHAP", fontsize=34)

    ax2 = fig.add_subplot(1, 2, 2)
    sizex = len(occurrences_toplot)
    values_x = range(sizex)
    # Building of dictionnary with the 4 classes for the bar plot
    listof5 = list(occurrences_toplot.values())
    li_nd3 = []
    li_pld3 = []
    li_nd10 = []
    li_pld10 = []
    for (_, b, c, d, e) in listof5:
        li_nd3.append(b)
        li_pld3.append(c)
        li_nd10.append(d)
        li_pld10.append(e)
    #dico4classes = {'naloxone D3': li_nd3, 'naloxone D10': li_nd10,
    #                'small lipectomy D3': li_pld3, 'small lipectomy D10': li_pld10}
    dico4classes = {'regenerating D3': li_nd3, 'regenerating D10': li_nd10,
                    'scarring D3': li_pld3, 'scarring D10': li_pld10}
    #dico4classes = {'Régénération J3': li_nd3, 'Régénération J10': li_nd10,
    #                'Cicatrisation J3': li_pld3, 'Cicatrisation J10': li_pld10}
    bottom = np.zeros(sizex)

    # colors = {'naloxone D3': 'crimson', 'naloxone D10': 'royalblue',
    #          'small lipectomy D3': 'pink', 'small lipectomy D10': 'lightblue'}
    colors = {'regenerating D3': 'crimson', 'regenerating D10': 'mediumblue',
             'scarring D3': 'lightcoral', 'scarring D10': 'skyblue'}  # couleurs pour que ca rende bien en N&B
    #colors = {'Régénération J3': 'crimson', 'Régénération J10': 'royalblue',
    #          'Cicatrisation J3': 'pink', 'Cicatrisation J10': 'lightblue'}

    for classname, occnb in dico4classes.items():
        ax2.bar(values_x, occnb, label=classname, bottom=bottom, color=colors[classname])
        bottom += occnb

    # plt.bar(range(len(occurrences_toplot)), list(occurrences_toplot.values()), align='center', color='tab:orange')
    plt.xticks(values_x, list(occurrences_toplot.keys()), rotation=90, ha='center')
    ax2.tick_params(axis='x', which='major', size=ticklen, width=linewd)
    ax2.tick_params(axis='y', which='major', size=ticklen, width=linewd)
    ax2.spines['bottom'].set_linewidth(linewd)
    ax2.spines['left'].set_linewidth(linewd)
    plt.xticks(fontsize=axvalues_ftsz)
    plt.yticks(fontsize=axvalues_ftsz)
    ax2.legend(fontsize=lgd_ftsz)
    plt.ylabel('Number of occurrences in the SHAP top %s' % ntop, fontsize=ftsz)
    #plt.ylabel("Nombre d\'occurences dans les tops 20 de SHAP", fontsize=34)
    ax2.set_xlim(xmin=-0.5, xmax=m-0.5)
    #plt.yticks(range(0, 17))
    ax2.set_ylim(ymin=0, ymax=16.2)

    plt.tight_layout()
    fig.savefig('./figures_generees/SHAP_240110mice_meanocc_22_4classes.png', dpi=300)
    plt.close()
    #plt.show()


plot_first_features(22, dicomeanposnames_sorted, dicooccnames_sorted, nfortop)


def plot_pos_occ(dicomeanposnames_s, dicooccnames_s, ntop):
    # Plots the features on the graph of the number of occurences in the tops 20 depending on the average position

    # occurrences_toplot = dict(filter(lambda kv: kv[1] != 0, dicooccnames_s.items()))
    # only the features that are in at least one top 20
    occurrences_toplot = dicooccnames_s
    xfeat, yocc = zip(*occurrences_toplot.items())

    meanpos_toplot = {key: dicomeanposnames_s[key] for key in occurrences_toplot.keys()}
    # only the features that were selected before
    _, ymean = zip(*meanpos_toplot.items())

    plt.rcParams["figure.figsize"] = [19, 10]
    plt.rcParams["figure.autolayout"] = True
    fig1, ax1 = plt.subplots()

    ftsz=24
    axvalues_ftsz=20
    feat_ftsz=15
    linewd = 2
    ticklen = 7

    # plt.scatter(ymean, yocc, color='tab:green')
    for i in range(len(xfeat)):
        x = ymean[i]
        y = yocc[i][0]
        if i in [2,4,13,14,15,21,28,29,31,36,59,200]:  # point a gauche du nom
            horizal='left'
            vertial='center'
            nametoplot = " " + xfeat[i]
        elif i in [18,20,25,33,51]:  # point a gauche, texte descendu
            horizal='left'
            vertial='top'
            nametoplot = " " + xfeat[i]
        elif i in [19,42]:  # point a gauche, texte remonte
            horizal='left'
            vertial='bottom'
            nametoplot = " " + xfeat[i]
        elif i in [69]:  # point a droite, texte remonte
            horizal = 'right'
            vertial = 'bottom'
            nametoplot = xfeat[i]
        else:  # point a droite
            horizal = 'right'
            vertial = 'center'
            nametoplot = xfeat[i] + " "
        if x <= 85: 
            plt.plot(x, y, 'o', color='tab:green')
            plt.text(x, y, nametoplot, fontsize=feat_ftsz, rotation=40, horizontalalignment=horizal,
                     verticalalignment=vertial, rotation_mode="anchor")

    ax1.tick_params(axis='x', which='major', size=ticklen, width=linewd)
    ax1.tick_params(axis='y', which='major', size=ticklen, width=linewd)
    ax1.spines['bottom'].set_linewidth(linewd)
    ax1.spines['left'].set_linewidth(linewd)
    plt.xticks(fontsize=axvalues_ftsz)
    plt.yticks(fontsize=axvalues_ftsz)

    plt.xlabel('Average position in the SHAP plots', fontsize=ftsz)
    plt.ylabel('Number of occurrences in the SHAP top %s' % ntop, fontsize=ftsz)
    ax1.set_xlim(xmin=-6)#, xmax=99.5)
    ax1.set_ylim(ymin=-1.05, ymax=18.1)
    plt.yticks([2*i for i in range(0,9)])

    plt.tight_layout()
    plt.savefig('./figures_generees/SHAP_240110mice_meanocc_x0to85.png', dpi=300)
    plt.close()
    #plt.show()


# plot_pos_occ(dicomeanposnames_sorted, dicooccnames_sorted, nfortop)
