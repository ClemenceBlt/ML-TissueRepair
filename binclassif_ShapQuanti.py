import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from matplotlib.ticker import PercentFormatter
import seaborn as sn

# Pour la classification binaire naloxone/petite lipectomie, analyse quantitative des SHAP avec le trace des graphes de
# dependance partielle.

# Parametres :
dataset_name = '230323_trainmosTV'
suffix = '_'

if suffix == '_':
    prefix_name = 'binclassif_ML_%s' % dataset_name
else:
    prefix_name = 'binclassif_ML_%s_%s' % (dataset_name, suffix)

npy_path_shapvalues = './npy_files/SHAPvalues_binary_classif_ML_%s.npy' % dataset_name
npy_path_Xtrainfiltered = './npy_files/Xfiltered_binary_classif_ML_%s_0.npy' % dataset_name
npy_path_Xtestfiltered = './npy_files/Xtestfiltered_binary_classif_ML_%s_1173408.npy' % dataset_name
npy_path_featmask = './npy_files/featmask_binary_classif_ML_230323_trainmosTV_0.npy'

# Utile seulement si on veut colorer en fonction de la classe :
npy_path_ytrain = './npy_files/ytrain_binary_classif_ML_%s_0.npy' % dataset_name
npy_path_ytest = './npy_files/ytest_binary_classif_ML_%s_1173408.npy' % dataset_name

# Recuperation tableau Xtrainfiltered
Xtrainfiltered = np.load(npy_path_Xtrainfiltered)
print('Train features loaded from', npy_path_Xtrainfiltered)
print('Len loaded X train filtered:', len(Xtrainfiltered))
print('Number of extracted features:', len(Xtrainfiltered[0]))

# Recuperation tableau Xtestfiltered
Xtestfiltered = np.load(npy_path_Xtestfiltered)
print('Test features loaded from', npy_path_Xtestfiltered)
print('Len loaded X test filtered:', len(Xtestfiltered))
print('Number of extracted features:', len(Xtestfiltered[0]))

# Recuperation tableau ytrain
ytrain = np.load(npy_path_ytrain)
print('Train features loaded from', npy_path_ytrain)
print('Len loaded y_train:', len(ytrain))

# Recuperation tableau ytest
ytest = np.load(npy_path_ytest)
print('Test features loaded from', npy_path_ytest)
print('Len loaded y_test:', len(ytest))

# Load names of all extracted features
pklfile_names = './pkl_files/pyfeats_features_names_230623.pkl'
with open(pklfile_names, 'rb') as gg:
    features_names = pickle.load(gg)
features_names = np.array(features_names)  # sinon le [mask] ne fonctionne pas
print('Features names loaded from', pklfile_names)
print('Number of features:', len(features_names))

# Load mask of selected features and create list of selected features names
featmask = np.load(npy_path_featmask)
print('Mask of selected features loaded from', npy_path_featmask)
unique, counts = np.unique(featmask, return_counts=True)
nboffeat = dict(zip(unique, counts))[True]
print("Number of selected features:", nboffeat)
selectedfeat_names = features_names[featmask]

# Load SHAP values
shapvalues = np.load(npy_path_shapvalues)
print("SHAP values successfully loaded from", npy_path_shapvalues)


Xallfiltered = np.concatenate((Xtrainfiltered, Xtestfiltered), axis=0)
idxtraintest = len(Xtrainfiltered)


"""
# Visualize the first prediction's explanation
print("shap_values", type(shapvalues), len(shapvalues), len(shapvalues[0]), len(shapvalues[0][0]))
# liste de dimension (2, 97576, 911) ie (nb de classes, nb d'images, nb de variables par image)
print("Xallfiltered", type(Xallfiltered), np.shape(Xallfiltered))

# Shap values plot
shap.summary_plot(shapvalues[0], features=Xallfiltered, feature_names=selectedfeat_names, max_display=22, show=True)
plt.close()
"""

# -- Valeur des variables --
featnb = 93
print("Variable consideree :", featnb, selectedfeat_names[featnb])


def dp_binary(feat_nb, shap_values, Xall_filtered, selected_feat_names, datasetname, prefixname):
    '''
    Trace et enregistrement du graphe de dependance partielle de la classification binaire pour la prediction de la
    naloxone.
    :param feat_nb: numero de la variable consideree
    :param shap_values: valeurs de SHAP pour la classification binaire calculees precedemment
    :param Xall_filtered: tableau contenant les images
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer le fichier cree
    :return: enregistre le graphe
    '''
    fig1, ax1 = plt.subplots(1, 1)
    ftsz=30
    linewd=2
    dotsz=2
    # version classique : 1, version big : 2
    shap.dependence_plot(feat_nb, shap_values[0], features=Xall_filtered, feature_names=selected_feat_names,
                         interaction_index=None, dot_size=dotsz, alpha=0.6, show=False, ax=ax1)
    # alternative pour le 1er agt : "rank(%s)" % featrank avec featrank=1 par exemple
    ax1.set_xlabel(selected_feat_names[feat_nb], fontsize=ftsz)
    ax1.set_ylabel("SHAP value for \n%s" % selected_feat_names[feat_nb], fontsize=ftsz)

    plt.gcf().set_size_inches(19, 10)
    ax1.set_xlim(xmin=7.2, xmax=7.75)
    ax1.set_ylim(ymin=-8, ymax=8.1)
    ax1.tick_params(size=ftsz-20, width=linewd)
    ax1.spines['bottom'].set_linewidth(linewd)
    ax1.spines['left'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-8)
    plt.yticks(fontsize=ftsz-8)
    #plt.title("Dependence plot - Binary classification (%s)" % datasetname)
    plt.tight_layout()
    plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_view1_big.png"
                % (feat_nb, prefixname, feat_nb), dpi=300)
    #plt.show()
    plt.close()

dp_binary(featnb, shapvalues, Xallfiltered, selectedfeat_names, dataset_name, prefix_name)


def dp_binary_traintest(feat_nb, shap_values, Xall_filtered, Xtrain_filtered, Xtest_filtered, idx, selected_feat_names,
                        datasetname, prefixname):
    '''
    Graphe de dependance partielle de la classification binaire (pour la prédiction de la naloxone) ou on distingue les
    images issues de l'ensemble d'entrainement de celui de test.
    Commentaire : shap_values a ete calcule a partir de Xallfiltered (shap_values = explainer.shap_values(Xallfiltered)).
    La ligne i de shap_values[0] indique les valeurs de SHAP de l'image de la ligne i de Xallfiltered, il n'y a pas de
    melange des lignes (puisque d'apres la documentation : "Each row [of shap_values] sums to the difference between the
    model output **for that sample** and the expected value of the model output"). Donc, vu la construction de
    Xallfiltered, shap_values[0][:len(Xtrainfiltered)] contient les valeurs de SHAP pour Xtrainfiltered et
    shap_values[0][len(Xtrainfiltered):] contient celles pour Xtestfiltered.
    :param feat_nb: numero de la variable consideree
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param Xall_filtered: tableau contenant toutes les images (entrainement et test)
    :param Xtrain_filtered: tableau contenant les images d'entrainement
    :param Xtest_filtered: tableau contenant les images de test
    :param idx: nb d'images de l'ensemble d'entrainement (c'est l'indice ou on coupe shap_values)
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer les fichiers crees
    :return: enregistre le graphe demande
    '''
    fig1, ax1 = plt.subplots(1, 1)
    print("Tailles donnees entrainement :", len(Xtrain_filtered), len(shap_values[0][:idx]))
    print("Tailles donnees test :", len(Xtest_filtered), len(shap_values[0][idx:]))

    # Ensemble d'entrainement :
    shap.dependence_plot(feat_nb, shap_values[0][:idx], features=Xall_filtered[:idx], feature_names=selected_feat_names,
                         interaction_index=None, dot_size=1, alpha=0.6, show=False, ax=ax1, color="mediumslateblue")
    ax1.text(0.01, 0.95, 'Training set', ha='left', va='top', transform=ax1.transAxes, size=15, color="mediumslateblue")
    # Ensemble de test :
    shap.dependence_plot(feat_nb, shap_values[0][idx:], features=Xall_filtered[idx:], feature_names=selected_feat_names,
                         interaction_index=None, dot_size=1, alpha=0.6, show=False, ax=ax1, color="Limegreen")
    ax1.text(0.01, 0.90, 'Test set', ha='left', va='top', transform=ax1.transAxes, size=15, color="Limegreen")
    # v1 : dot_size=2, alpha=1
    # v2 : dot_size=1, alpha=0.6

    plt.gcf().set_size_inches(19, 10)
    ax1.set_xlim(xmin=0, xmax=47000)
    ax1.set_ylim(ymin=-2.6, ymax=1.4)
    plt.title("Dependence plot - Binary classification (%s)" % datasetname)
    plt.tight_layout()
    #plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_view2_TnTe_v1.png"
    #            % (feat_nb, prefixname, feat_nb), dpi=300)
    plt.show()
    plt.close()

#dp_binary_traintest(featnb, shapvalues, Xallfiltered, Xtrainfiltered, Xtestfiltered, idxtraintest, selectedfeat_names,
#                    dataset_name, prefix_name)


def dp_binary_naloxptlip(feat_nb, shap_values, Xtrain_filtered, Xtest_filtered, y_train, y_test, selected_feat_names,
                         datasetname, prefixname):
    '''
    Graphe de dependance partielle de la classification binaire (pour la prédiction de la naloxone) ou on distingue les
    images naloxone et petite lipectomie.
    :param feat_nb: numero de la variable consideree
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param Xtrain_filtered: tableau contenant les images d'entrainement
    :param Xtest_filtered: tableau contenant les images de test
    :param y_train: tableau contenant les labels des images d'entrainement
    :param y_test: tableau contenant les labels des images de test
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer les fichiers crees
    :return: enregistre le graphe demandee
    '''
    # Definition des indices, tableaux d'images et tableaux de valeurs de SHAP utiles pour la suite
    idx = len(Xtrain_filtered)  # nb d'images d'entrainement
    idx_trainnalox = [i for i in range(len(y_train)) if y_train[i]==0]  # indices dans Xtrain (ou ytrain) des images nalox
    idx_trainptlip = [i for i in range(len(y_train)) if y_train[i]==1]  # indices dans Xtrain (ou ytrain) des images ptlip
    idx_testnalox = [i for i in range(len(y_test)) if y_test[i]==0]  # indices dans Xtest (ou ytest) des images nalox
    idx_testptlip = [i for i in range(len(y_test)) if y_test[i]==1]  # indices dans Xtest (ou ytest) des images ptlip
    Xtrainnalox = np.array([Xtrain_filtered[i] for i in idx_trainnalox])  # images nalox d'entrainement
    Xtrainptlip = np.array([Xtrain_filtered[i] for i in idx_trainptlip])  # images ptlip d'entrainement
    Xtestnalox = np.array([Xtest_filtered[i] for i in idx_testnalox])  # images nalox de test
    Xtestptlip = np.array([Xtest_filtered[i] for i in idx_testptlip])  # images ptlip de test
    shapvalues_trainnalox = np.array([shap_values[0][:idx][i] for i in idx_trainnalox])  # valeurs de SHAP des images d'entrainement nalox
    shapvalues_trainptlip = np.array([shap_values[0][:idx][i] for i in idx_trainptlip])  # valeurs de SHAP des images d'entrainement ptlip
    shapvalues_testnalox = np.array([shap_values[0][idx:][i] for i in idx_testnalox])  # valeurs de SHAP des images de test nalox
    shapvalues_testptlip = np.array([shap_values[0][idx:][i] for i in idx_testptlip])  # valeurs de SHAP des images de test ptlip

    fig1, ax1 = plt.subplots(1, 1)
    ftsz=30
    linewd=2
    dotsz=2
    # version classique : 1 pour train, 2 pour test
    # version big : 2 pour train, 3 pour test
    alphavl=0.6
    #"""
    # Trace des graphes nalox et ptlip pour l'ensemble d'entrainement
    shap.dependence_plot(feat_nb, shapvalues_trainnalox, features=Xtrainnalox, feature_names=selected_feat_names,
                         interaction_index=None, dot_size=dotsz, alpha=alphavl, show=False, ax=ax1, color="mediumslateblue")
    shap.dependence_plot(feat_nb, shapvalues_trainptlip, features=Xtrainptlip, feature_names=selected_feat_names,
                         interaction_index=None, dot_size=dotsz, alpha=alphavl, show=False, ax=ax1, color="Orange")
    #ax1.text(0.03, 0.96, 'Training set - naloxone', ha='left', va='top', transform=ax1.transAxes, size=ftsz-5,
    #         color="mediumslateblue")
    #ax1.text(0.03, 0.90, 'Training set - small lipectomy', ha='left', va='top', transform=ax1.transAxes, size=ftsz-5,
    #         color="Orange")
    ax1.text(0.03, 0.96, 'regenerating', ha='left', va='top', transform=ax1.transAxes, size=ftsz-5,
             color="mediumslateblue")
    ax1.text(0.03, 0.90, 'scarring', ha='left', va='top', transform=ax1.transAxes, size=ftsz-5,
             color="Orange")
    """
    # Trace des graphes nalox et ptlip pour l'ensemble de test
    shap.dependence_plot(feat_nb, shapvalues_testnalox, features=Xtestnalox, feature_names=selected_feat_names,
                         interaction_index=None, dot_size=dotsz, alpha=alphavl, show=False, ax=ax1, color="Red")
    shap.dependence_plot(feat_nb, shapvalues_testptlip, features=Xtestptlip, feature_names=selected_feat_names,
                         interaction_index=None, dot_size=dotsz, alpha=alphavl, show=False, ax=ax1, color="Limegreen")
    #ax1.text(0.03, 0.96, 'Test set - naloxone', ha='left', va='top', transform=ax1.transAxes, size=lgdsize,
    #         color="Red")
    #ax1.text(0.03, 0.90, 'Test set - small lipectomy', ha='left', va='top', transform=ax1.transAxes, size=lgdsize,
    #         color="Limegreen")
    ax1.text(0.03, 0.96, 'regenerating', ha='left', va='top', transform=ax1.transAxes, size=ftsz-5,
             color="Red")
    ax1.text(0.03, 0.90, 'scarring', ha='left', va='top', transform=ax1.transAxes, size=ftsz-5,
             color="Limegreen")
    """
    ax1.set_xlabel(selected_feat_names[feat_nb], fontsize=ftsz)
    ax1.set_ylabel("SHAP value for \n%s" % selected_feat_names[feat_nb], fontsize=ftsz)

    plt.gcf().set_size_inches(19,10)
    ax1.set_xlim(xmin=-0.0005, xmax=0.07)
    ax1.set_ylim(ymin=-6.1, ymax=10.1)
    ax1.tick_params(size=ftsz-20, width=linewd)
    ax1.spines['bottom'].set_linewidth(linewd)
    ax1.spines['left'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-8)
    plt.yticks(fontsize=ftsz-8)
    #plt.title("Dependence plot - Binary classification (%s)" % datasetname)
    plt.tight_layout()
    plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_view1_Tn_NP_big.png"
                % (feat_nb, prefixname, feat_nb), dpi=300)
    #plt.show()
    plt.close()

#dp_binary_naloxptlip(featnb, shapvalues, Xtrainfiltered, Xtestfiltered, ytrain, ytest, selectedfeat_names, dataset_name,
#                    prefix_name)


def boxplots_binary_naloxptlip(feat_nb, shap_values, y_train, y_test, idx, selectedfeatnames, datasetname, prefixname):
    '''
    Analyse des graphes des donnees presentees par la fonction dp_binary_naloxptlip : repartition des images de nalox
    et de ptlip dans le cadre de la prediction de la nalox. Reprise des boxplots de la fonction
    histsign_binary_naloxptlip avec les ensembles d'entrainement et de test compares.
    :param feat_nb: numero de la variable consideree
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param y_train: tableau contenant les labels des images d'entrainement
    :param y_test: tableau contenant les labels des images de test
    :param idx: nb d'images de l'ensemble d'entrainement (c'est l'indice ou on coupe shap_values)
    :param selectedfeatnames: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer le fichier cree
    :return: enregistre le graphe demande
    '''
    plt.rcParams["figure.figsize"] = [25, 13]
    plt.rcParams["figure.autolayout"] = True

    # -- Ensemble d'entrainement --
    # Indices des images nalox :
    idx_train_n = [i for i in range(len(y_train)) if y_train[i] == 0]
    # Indices des images ptlip :
    idx_train_p = [i for i in range(len(y_train)) if y_train[i] == 1]
    # Valeurs de SHAP pour la prediction de la naloxone pour la variable feat_nb :
    shap_train_predn = shap_values[0][:idx][:, feat_nb]
    # Valeurs de SHAP pour la prediction de la naloxone avec seulement les images nalox :
    shap_train_predn_imgn = np.array([shap_train_predn[i] for i in idx_train_n])
    # Valeurs de SHAP pour la prediction de la naloxone avec seulement les images ptlip :
    shap_train_predn_imgp = np.array([shap_train_predn[i] for i in idx_train_p])

    # -- Ensemble de test --
    # Indices des images nalox :
    idx_test_n = [i for i in range(len(y_test)) if y_test[i] == 0]
    # Indices des images ptlip :
    idx_test_p = [i for i in range(len(y_test)) if y_test[i] == 1]
    # Valeurs de SHAP pour la prediction de la naloxone pour la variable feat_nb :
    shap_test_predn = shap_values[0][idx:][:, feat_nb]
    # Valeurs de SHAP pour la prediction de la naloxone avec seulement les images nalox :
    shap_test_predn_imgn = np.array([shap_test_predn[i] for i in idx_test_n])
    # Valeurs de SHAP pour la prediction de la naloxone avec seulement les images ptlip :
    shap_test_predn_imgp = np.array([shap_test_predn[i] for i in idx_test_p])

    sn.set_theme(style='ticks')
    f, (ax_train_boxn, ax_train_boxp, ax_test_boxn, ax_test_boxp)\
       = plt.subplots(4, sharex='all', gridspec_kw={'height_ratios': (0.25, 0.25, 0.25, 0.25)})
    #f, (ax_train_boxn, ax_train_boxp)\
    #   = plt.subplots(2, sharex='all', gridspec_kw={'height_ratios': (0.5, 0.5)})

    colortrainnalox = "mediumslateblue"
    colortrainptlip = "Orange"
    colortestnalox = "Red"
    colortestptlip = "Limegreen"

    linewd = 4
    sn.boxplot(x=shap_train_predn_imgn, fill=False, color=colortrainnalox, ax=ax_train_boxn, linewidth=linewd,
               showmeans=True, meanline=True, meanprops=dict(color=colortrainnalox, linewidth=linewd))
    sn.boxplot(x=shap_train_predn_imgp, fill=False, color=colortrainptlip, ax=ax_train_boxp, linewidth=linewd,
               showmeans=True, meanline=True, meanprops=dict(color=colortrainptlip, linewidth=linewd))
    sn.boxplot(x=shap_test_predn_imgn, fill=False, color=colortestnalox, ax=ax_test_boxn, linewidth=linewd,
               showmeans=True, meanline=True, meanprops=dict(color=colortestnalox, linewidth=linewd))
    sn.boxplot(x=shap_test_predn_imgp, fill=False, color=colortestptlip, ax=ax_test_boxp, linewidth=linewd,
               showmeans=True, meanline=True, meanprops=dict(color=colortestptlip, linewidth=linewd))

    # Reglage des axes :
    ax_train_boxn.axvline(x=0, ymin=-1, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)  # ligne verticale
    ax_train_boxp.axvline(x=0, ymin=-1, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)  # ymin-1
    ax_test_boxn.axvline(x=0, ymin=-1, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)
    ax_test_boxp.axvline(x=0, ymin=0, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)
    sn.despine(ax=ax_train_boxn, left=True, bottom=True)  # on enleve aussi gauche et bas
    sn.despine(ax=ax_train_boxp, left=True)#, bottom=True)
    sn.despine(ax=ax_test_boxn, left=True, bottom=True)
    sn.despine(ax=ax_test_boxp, left=True)
    ax_train_boxn.tick_params(left=False, bottom=False)
    ax_train_boxp.tick_params(left=False, bottom=True, size=20, width=linewd)
    # si seulement train : labelbottom=True, size=20, width=linewd
    # si train et test : bottom=True, size=20, width=linewd
    ax_train_boxp.spines['bottom'].set_linewidth(linewd)
    ax_test_boxn.tick_params(left=False, bottom=False)
    ax_test_boxp.tick_params(left=False, labelbottom=True, size=20, width=linewd)
    ax_test_boxp.spines['bottom'].set_linewidth(linewd)
    ftsz = 50
    #ax_train_boxn.set_ylabel('naloxone', rotation=0, fontsize=12)
    #ax_train_boxp.set_ylabel('small lipectomy', rotation=0, fontsize=12)
    #ax_test_boxn.set_ylabel('naloxone', rotation=0, fontsize=12)
    #ax_test_boxp.set_ylabel('small lipectomy', rotation=0, fontsize=12)
    ax_train_boxn.set_ylabel('rgn', rotation=0, fontsize=ftsz)
    ax_train_boxp.set_ylabel('scar.', rotation=0, fontsize=ftsz)
    #ax_train_boxp.set_xlabel('SHAP values', fontsize=ftsz)
    ax_test_boxn.set_ylabel('rgn', rotation=0, fontsize=ftsz)
    ax_test_boxp.set_ylabel('scar.', rotation=0, fontsize=ftsz)
    ax_test_boxp.set_xlabel('SHAP values', fontsize=ftsz)
    #ax_test_boxp.set_xlabel('SHAP values for the prediction of the naloxone', fontsize=12)
    #ax_train_boxp.set_title('--- mean', loc='right', size=ftsz-10,
    #                       bbox=dict(facecolor='none', edgecolor='gainsboro', boxstyle='round'))
    #ax_train_boxn.set_title("Training set", fontsize=15)
    #ax_test_boxn.set_title("Test set", fontsize=15)
    plt.xticks(fontsize=ftsz-5)
    plt.tight_layout()
    plt.savefig("./figures_generees/Shap_DepPlot_feat%s/%s_boxtraintest_feat%s.png"
                % (feat_nb, prefixname, feat_nb), dpi=300)
    #plt.savefig("./figures_generees/Shap_DepPlot_article/%s_boxtrain_feat%s.png"
    #            % (prefixname, feat_nb), dpi=300)
    #plt.show()

#boxplots_binary_naloxptlip(featnb, shapvalues, ytrain, ytest, idxtraintest, selectedfeat_names, dataset_name,
#                  prefix_name)