import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import seaborn as sn

# Pour la classification des souris, analyse quantitative des SHAP avec le trace des graphes de dependance partielle.

# Parametres :
dataset_name = '240110_mice'
suffix = '_'

if suffix == '_':
    prefix_name = 'miceclassif_ML_%s' % dataset_name
else:
    prefix_name = 'miceclassif_ML_%s_%s' % (dataset_name, suffix)

npy_path_shapvalues = './npy_files/SHAPvalues_miceclassif_ML_%s_%sTV.npy' % (dataset_name, dataset_name)
npy_path_Xtrainfiltered = './npy_files/Xtrainfiltered_miceclassif_ML_%s_%sTV.npy' % (dataset_name, dataset_name)
npy_path_Xtestfiltered = './npy_files/Xtestfiltered_miceclassif_ML_%s_%sTe.npy' % (dataset_name, dataset_name)
npy_path_featmask = './npy_files/featmask_miceclassif_ML_230330_22miceTV_1141510.npy'

# Utile seulement si on veut colorer en fonction de la classe :
npy_path_ytrain = './npy_files/ytrain_miceclassif_ML_%s_%sTV.npy' % (dataset_name, dataset_name)
npy_path_ytest = './npy_files/ytest_miceclassif_ML_%s_%sTe.npy' % (dataset_name, dataset_name)

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
# liste de dimension  (22, 77836, 1057) ie (nb de classes/souris, nb d'images, nb de variables par image)
print("Xallfiltered", type(Xallfiltered), np.shape(Xallfiltered))

# Shap values plot
for mousenb in range(22):
    shap.summary_plot(shapvalues[mousenb], features=Xallfiltered, feature_names=selectedfeat_names, max_display=20,
                      show=True)
"""

# -- Valeur des variables --
featnb = 93
print("Variable consideree :", featnb, selectedfeat_names[featnb])
listmice = [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21]
listmicefeat68 = [1, 3, 4, 8, 10, 12, 15, 17, 20, 21]
listmicefeat80 = [0, 1, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 21]
listmicefeat85 = [0, 3, 6, 7, 8, 9, 10, 13, 14, 15, 21]
listmicefeat93 = [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 20, 21]


def dp_onemouse(feat_nb, list_mice, shap_values, Xall_filtered, selected_feat_names, datasetname, prefixname):
    '''
    Trace et enregistrement des graphes de dependance partielle de chaque souris de la liste donnee.
    :param feat_nb: numero de la variable consideree
    :param list_mice: liste des numeros des souris dont on veut tracer les graphes de DP
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param Xall_filtered: tableau contenant les images
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer les fichiers crees
    :return: enregistre les graphes de toutes les souris demandees
    '''
    for mousenb in list_mice:
        fig1, ax1 = plt.subplots(1, 1)
        shap.dependence_plot(feat_nb, shap_values[mousenb], features=Xall_filtered, feature_names=selected_feat_names,
                             interaction_index=None, dot_size=2, alpha=1, show=False, ax=ax1)
        # alternative pour le 1er agt : "rank(%s)" % featrank avec featrank=1 par exemple
        plt.gcf().set_size_inches(19, 10)
        ax1.set_xlim(xmin=-0.0005, xmax=0.07)
        ax1.set_ylim(ymin=-1.4, ymax=3.3)
        plt.title("Dependence plot for mouse %s - Mice classification (%s)" % (mousenb, datasetname))
        plt.tight_layout()
        plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_mouse%s.png"
                    % (feat_nb, prefixname, feat_nb, mousenb), dpi=300)
        #plt.show()
        plt.close()

listmice0 = [0]
#dp_onemouse(featnb, listmice0, shapvalues, Xallfiltered, selectedfeat_names, dataset_name, prefix_name)


def dp_manymice(feat_nb, list_mice, shap_values, Xall_filtered, selected_feat_names, datasetname, prefixname):
    '''
    Superposition sur une meme figure des graphes de dependance partielle de plusieurs souris.
    :param feat_nb: numero de la variable consideree
    :param list_mice: liste des numeros des souris dont on veut tracer les graphes de DP
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param Xall_filtered: tableau contenant les images
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer les fichiers crees
    :return: enregistre le graphe demande
    '''
    cmap = plt.get_cmap('nipy_spectral')
    colorlist = [cmap(i) for i in np.linspace(0, 1, 22)]  # selection de 22 couleurs dans la carte cmap
    cpt=0  # pour position de la légende
    fig1, ax1 = plt.subplots(1, 1)
    ftsz=30
    linewd=2
    dotsz=2
    # version classique : 1, version big : 2
    pos93 = 0

    idx = 0
    listnb = [1,4,9]  # nb to plot
    for mousenb in list_mice:
        shap.dependence_plot(feat_nb, shap_values[mousenb], features=Xall_filtered,
                             feature_names=selected_feat_names, interaction_index=None, dot_size=dotsz, alpha=0.6,
                             show=False, ax=ax1, color=colorlist[mousenb])
        #ax1.text(0.03+0.11*(pos93//4), 1.0-(pos93%4)*0.04, "Mouse %s" % mousenb, color=colorlist[mousenb], ha='left', va='top',
        #         transform=ax1.transAxes, size=ftsz-5)  # Legende  # (0.03, 1.0-cpt)
        # Legende pour coherence des numeros avec matrices de confusion :
        ax1.text(0.03+0.11*(pos93//4), 1.0-(pos93%4)*0.04, "Mouse %s" % listnb[idx], color=colorlist[mousenb], ha='left', va='top',
                 transform=ax1.transAxes, size=ftsz-5)  # Legende  # (0.03, 1.0-cpt)
        idx += 1
        cpt += 0.04
        pos93+=1

    ax1.set_xlabel(selected_feat_names[feat_nb], fontsize=ftsz)
    ax1.set_ylabel("SHAP value for \n%s" % selected_feat_names[feat_nb], fontsize=ftsz)

    plt.gcf().set_size_inches(19, 10)
    ax1.set_xlim(xmin=0, xmax=0.26)
    ax1.set_ylim(ymin=-9.5, ymax=7.9)
    ax1.tick_params(size=ftsz-20, width=linewd)
    ax1.spines['bottom'].set_linewidth(linewd)
    ax1.spines['left'].set_linewidth(linewd)
    plt.xticks(fontsize=ftsz-8)
    plt.yticks(fontsize=ftsz-8)
    #plt.title("Dependence plot - Mice classification (%s)" % datasetname)
    plt.tight_layout()
    #plt.savefig("./figures_generees/Shap_DepPlot_%s_feat%s_allmice_view2.png" % (prefixname, feat_nb), dpi=300)
    plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_top20group3_view1_big_250926.png"
                % (feat_nb, prefixname, feat_nb), dpi=300)
    #plt.show()
    plt.close()

# Liste des souris qu'on veut tracer (pour toutes les faire : listmice) :
listofmice = [1,6,12]
#dp_manymice(featnb, listofmice, shapvalues, Xallfiltered, selectedfeat_names, dataset_name, prefix_name)


def dp_manymice_traintest(feat_nb, list_mice, shap_values, Xall_filtered, idx, selected_feat_names,
                          datasetname, prefixname):
    '''
    Superposition sur une meme figure des graphes de dependance partielle de plusieurs souris. Toutes les souris ont la
    meme couleur, mais on distingue les images issues de l'ensemble d'entrainement de celui de test.
    :param feat_nb: numero de la variable consideree
    :param list_mice: liste des numeros des souris dont on veut tracer les graphes de DP
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param Xall_filtered: tableau contenant toutes les images (entrainement et test)
    :param idx: nb d'images de l'ensemble d'entrainement (c'est l'indice ou on coupe shap_values)
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer les fichiers crees
    :return: enregistre le graphe demande
    '''
    fig1, ax1 = plt.subplots(1, 1)
    dotsize = 1
    paramalpha = 0.6
    # v1 : dotsize=2, paramalpha=1
    # v2 : dotsize=1, paramalpha=0.6

    for mousenb in list_mice:
        # Ensemble d'entrainement :
        shap.dependence_plot(feat_nb, shap_values[mousenb][:idx], features=Xall_filtered[:idx],
                             feature_names=selected_feat_names, interaction_index=None, dot_size=dotsize,
                             alpha=paramalpha, show=False, ax=ax1, color="mediumslateblue")
        # Ensemble de test :
        shap.dependence_plot(feat_nb, shap_values[mousenb][idx:], features=Xall_filtered[idx:],
                             feature_names=selected_feat_names, interaction_index=None, dot_size=dotsize,
                             alpha=paramalpha, show=False, ax=ax1, color="Limegreen")

    plt.gcf().set_size_inches(19, 10)
    ax1.set_xlim(xmin=7.2, xmax=7.75)
    ax1.set_ylim(ymin=-8, ymax=8.1)
    ax1.text(0.01, 0.95, 'Training set', ha='left', va='top', transform=ax1.transAxes, size=15, color="mediumslateblue")
    ax1.text(0.01, 0.90, 'Test set', ha='left', va='top', transform=ax1.transAxes, size=15, color="Limegreen")
    plt.title("Dependence plot - Mice classification (%s)" % datasetname)
    plt.tight_layout()
    #plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_top20mice_view1_TnTe.png"
    #            % (feat_nb, prefixname, feat_nb), dpi=300)
    plt.show()
    plt.close()

# Liste des souris qu'on veut tracer (pour toutes les faire : listmice) :
list_ofmice = [0]#, 1, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 21]
#dp_manymice_traintest(featnb, list_ofmice, shapvalues, Xallfiltered, idxtraintest, selectedfeat_names, dataset_name,
#                      prefix_name)


def dp_onemouse_mnotm(feat_nb, list_mice, shap_values, setname, X_filtered, y, idx, selected_feat_names, datasetname,
                      prefixname):
    '''
    Pour chaque souris de la liste list_mice, trace du graphe de dependance partielle pour la prédiction de cette souris
    ou on distingue les images de cette souris (m) de celles des autres souris (notm).
    :param feat_nb: numero de la variable consideree
    :param list_mice: liste des numeros des souris dont on veut tracer les graphes d'analyse du signe de SHAP
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param setname: 'train' ou 'test' pour indiquer a quel ensemble appartiennent les X et y donnes ensuite
    :param X_filtered: tableau contenant les images de l'ensemble considere (entrainement ou test)
    :param y: tableau contenant les labels des images de l'ensemble considere
    :param idx: nb d'images de l'ensemble d'entrainement (c'est l'indice ou on coupe shap_values)
    :param selected_feat_names: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer les fichiers crees
    :return: enregistre les graphes demandes
    '''
    for mousenb in list_mice:
        # Indices des images de la souris consideree :
        idx_m = [i for i in range(len(y)) if y[i] == mousenb]
        # Indices des images des autres souris :
        idx_notm = [i for i in range(len(y)) if y[i] != mousenb]
        # Images de la souris consideree dans l'ensemble considere :
        X_m = np.array([X_filtered[i] for i in idx_m])
        # Images de toutes les souris sauf la souris consideree dans l'ensemble considere :
        X_notm = np.array([X_filtered[i] for i in idx_notm])
        # Valeurs de SHAP pour la prediction de la souris mousenb pour les images de l'ensemble donne
        if setname == 'train':
            shap_predm = shap_values[mousenb][:idx]
            colorm = "mediumslateblue"
            colornotm = "Orange"
            dotsizem = 2
            dotsizenotm = 1
        else:
            shap_predm = shap_values[mousenb][idx:]
            colorm = "Red"
            colornotm = "Limegreen"
            dotsizem = 4  # 3
            dotsizenotm = 3  # 2
        ftsz=30
        linewd=2
        # Extrait du tableau precedent avec seulement les images de la souris consideree :
        shap_predm_imgm = np.array([shap_predm[i] for i in idx_m])
        # Extrait du tableau precedent avec seulement les images de toutes les autres souris :
        shap_predm_notimgm = np.array([shap_predm[i] for i in idx_notm])

        fig1, ax1 = plt.subplots(1, 1)
        # Trace des graphes de la souris et des autres pour l'ensemble prescrit
        # On trace un peu plus gros les points de la souris consideree car il y en a moins que ceux des autres souris.
        shap.dependence_plot(feat_nb, shap_predm_notimgm, features=X_notm, feature_names=selected_feat_names,
                             interaction_index=None, dot_size=dotsizenotm, alpha=0.6, show=False, ax=ax1, color=colornotm)
        shap.dependence_plot(feat_nb, shap_predm_imgm, features=X_m, feature_names=selected_feat_names,
                             interaction_index=None, dot_size=dotsizem, alpha=1, show=False, ax=ax1, color=colorm)
        #ax1.text(0.03, 0.96, '%s set - All mice except mouse %s' % (setname, mousenb), ha='left', va='top',
        #         transform=ax1.transAxes, size=ftsz-5, color=colornotm)
        #ax1.text(0.03, 0.90, '%s set - Mouse %s' % (setname, mousenb), ha='left', va='top',
        #         transform=ax1.transAxes, size=ftsz-5, color=colorm)
        ax1.text(0.03, 0.90, 'All mice except mouse %s' % mousenb, ha='left', va='top',
                 transform=ax1.transAxes, size=ftsz-5, color=colornotm)
        ax1.text(0.03, 0.96, 'Mouse %s' % mousenb, ha='left', va='top',
                 transform=ax1.transAxes, size=ftsz-5, color=colorm)

        ax1.set_xlabel(selected_feat_names[feat_nb], fontsize=ftsz)
        ax1.set_ylabel("SHAP value for \n%s" % selected_feat_names[feat_nb], fontsize=ftsz)

        plt.gcf().set_size_inches(19,10)
        ax1.set_xlim(xmin=-0.0005, xmax=0.07)
        ax1.set_ylim(ymin=-1.4, ymax=3.3)
        ax1.tick_params(size=ftsz-20, width=linewd)
        ax1.spines['bottom'].set_linewidth(linewd)
        ax1.spines['left'].set_linewidth(linewd)
        plt.xticks(fontsize=ftsz-8)
        plt.yticks(fontsize=ftsz-8)
        #plt.title("Dependence plot - Mice classification (%s) - Prediction of mouse %s" % (datasetname, mousenb))
        plt.tight_layout()
        plt.savefig("./figures_generees/Shap_DepPlot_feat%s/Shap_DepPlot_%s_feat%s_mouse%s_%s_mnotm_big.png"
                    % (feat_nb, prefixname, feat_nb, mousenb, setname), dpi=300)
        #plt.show()
        plt.close()

# Liste des souris qu'on veut tracer (pour toutes les faire : listmice) :
list_of_mice = [0]#, 3, 6, 7, 8, 9, 10, 13, 14, 15, 21]
#dp_onemouse_mnotm(featnb, list_of_mice, shapvalues, 'test', Xtestfiltered, ytest, idxtraintest,
#                 selectedfeat_names, dataset_name, prefix_name)



def boxplots_manymice_mnotm(feat_nb, list_mice, shap_values, y_train, y_test, idx, selectedfeatnames, datasetname,
                            prefixname):
    '''
    Analyse des graphes des donnees presentees par la fonction dp_onemouse_mnotm : repartition des images de la souris
    m et des autres souris dans le cadre de la prediction de la souris m. On considere la prediction des souris de
    list_mice. Reprise des boxplots de la fonction histsign_manymice_mnotm avec les ensembles d'entrainement et de test
    compares.
    :param feat_nb: numero de la variable consideree
    :param list_mice: liste des numeros des souris dont on veut tracer les graphes d'analyse du signe de SHAP
    :param shap_values: valeurs de SHAP pour la classification des souris calculees precedemment
    :param y_train: tableau contenant les labels des images de l'ensemble d'entrainement
    :param y_test: tableau contenant les labels des images de l'ensemble de test
    :param idx: nb d'images de l'ensemble d'entrainement (c'est l'indice ou on coupe shap_values)
    :param selectedfeatnames: liste contenant les noms de toutes les variables
    :param datasetname: nom de l'ensemble de donnees
    :param prefixname: prefixe pour nommer le fichier cree
    :return: enregistre le graphe demande
    '''
    plt.rcParams["figure.figsize"] = [25, 13]
    plt.rcParams["figure.autolayout"] = True

    for mousenb in list_mice:
        # -- Ensemble d'entrainement --
        # Indices des images de la souris consideree :
        idx_train_m = [i for i in range(len(y_train)) if y_train[i] == mousenb]
        # Indices des images des autres souris :
        idx_train_notm = [i for i in range(len(y_train)) if y_train[i] != mousenb]
        # Valeurs de SHAP pour la prediction de la souris mousenb pour la variable feat_nb :
        shap_train_predm = shap_values[mousenb][:idx][:, feat_nb]
        # Extrait du tableau precedent avec seulement les images de la souris consideree :
        shap_train_predm_imgm = np.array([shap_train_predm[i] for i in idx_train_m])
        # Extrait du tableau precedent avec seulement les images de toutes les autres souris :
        shap_train_predm_notimgm = np.array([shap_train_predm[i] for i in idx_train_notm])

        # -- Ensemble de test --
        # Indices des images de la souris consideree :
        idx_test_m = [i for i in range(len(y_test)) if y_test[i] == mousenb]
        # Indices des images des autres souris :
        idx_test_notm = [i for i in range(len(y_test)) if y_test[i] != mousenb]
        # Extrait du tableau precedent avec seulement les images de la souris consideree :
        shap_test_predm = shap_values[mousenb][idx:][:, feat_nb]
        # Extrait du tableau precedent avec seulement les images de la souris consideree :
        shap_test_predm_imgm = np.array([shap_test_predm[i] for i in idx_test_m])
        # Extrait du tableau precedent avec seulement les images de toutes les autres souris :
        shap_test_predm_notimgm = np.array([shap_test_predm[i] for i in idx_test_notm])

        sn.set_theme(style='ticks')
        f, (ax_train_boxm, ax_train_boxnotm, ax_test_boxm, ax_test_boxnotm) \
            = plt.subplots(4, sharex='all', gridspec_kw={'height_ratios': (0.25, 0.25, 0.25, 0.25)})
        #f, (ax_test_boxm, ax_test_boxnotm) \
        #    = plt.subplots(2, sharex='all', gridspec_kw={'height_ratios': (0.5, 0.5)})

        colortrainm = "mediumslateblue"
        colortrainnotm = "Orange"
        colortestm = "Red"
        colortestnotm = "Limegreen"

        linewd = 4
        sn.boxplot(x=shap_train_predm_imgm, fill=False, color=colortrainm, ax=ax_train_boxm, linewidth=linewd,
                   showmeans=True, meanline=True, meanprops=dict(color=colortrainm, linewidth=linewd))  # affichage moyenne
        sn.boxplot(x=shap_train_predm_notimgm, fill=False, color=colortrainnotm, ax=ax_train_boxnotm, linewidth=linewd,
                   showmeans=True, meanline=True, meanprops=dict(color=colortrainnotm, linewidth=linewd))
        sn.boxplot(x=shap_test_predm_imgm, fill=False, color=colortestm, ax=ax_test_boxm, linewidth=linewd,
                   showmeans=True, meanline=True, meanprops=dict(color=colortestm, linewidth=linewd))
        sn.boxplot(x=shap_test_predm_notimgm, fill=False, color=colortestnotm, ax=ax_test_boxnotm, linewidth=linewd,
                   showmeans=True, meanline=True, meanprops=dict(color=colortestnotm, linewidth=linewd))

        # Reglage des axes :
        # ligne verticale :
        ax_train_boxm.axvline(x=0, ymin=-1, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)
        ax_train_boxnotm.axvline(x=0, ymin=-1, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)
        ax_test_boxm.axvline(x=0, ymin=-1, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)
        ax_test_boxnotm.axvline(x=0, ymin=0, ymax=1, color='gray', ls='-.', clip_on=False, linewidth=linewd-1)

        sn.despine(ax=ax_train_boxm, left=True, bottom=True)  # on enleve aussi gauche et bas
        sn.despine(ax=ax_train_boxnotm, left=True)  #, bottom=True)
        sn.despine(ax=ax_test_boxm, left=True, bottom=True)
        sn.despine(ax=ax_test_boxnotm, left=True)
        ax_train_boxm.tick_params(left=False, bottom=False)
        ax_train_boxnotm.tick_params(left=False, bottom=True, size=20, width=linewd)
        ax_train_boxnotm.spines['bottom'].set_linewidth(linewd)
        ax_test_boxm.tick_params(left=False, bottom=False)
        ax_test_boxnotm.tick_params(left=False, labelbottom=True, size=20, width=linewd)
        ax_test_boxnotm.spines['bottom'].set_linewidth(linewd)
        ftsz = 50
        # Pour coherence numeros dans article :
        if mousenb<2:
            mousenb_toplot = mousenb
        elif mousenb in [3,4]:
            mousenb_toplot = mousenb-1
        elif mousenb in [6,7,8,9,10]:
            mousenb_toplot = mousenb-2
        elif mousenb in [12,13,14,15,16,17]:
            mousenb_toplot = mousenb-3
        else:
            mousenb_toplot = mousenb-5
        ax_train_boxm.set_ylabel('mouse %s' % mousenb_toplot, rotation=0, fontsize=ftsz)
        ax_train_boxm.yaxis.set_label_coords(-0.035, 0.35)
        # si train et test : -0.035, 0.35
        # si train ou test seul : -0.035, 0.44
        ax_train_boxnotm.set_ylabel('all mice \nexcept \nmouse %s' % mousenb_toplot, rotation=0, fontsize=ftsz)
        ax_train_boxnotm.yaxis.set_label_coords(-0.035, 0.08)
        # si train et test : -0.035, 0.08
        # si train ou test seul : -0.035, 0.3
        ax_train_boxnotm.set_xlabel('SHAP values', fontsize=ftsz)
        ax_test_boxm.set_ylabel('mouse %s' % mousenb_toplot, rotation=0, fontsize=ftsz)
        ax_test_boxm.yaxis.set_label_coords(-0.035, 0.35)
        ax_test_boxnotm.set_ylabel('all mice \nexcept \nmouse %s' % mousenb_toplot, rotation=0, fontsize=ftsz)
        ax_test_boxnotm.yaxis.set_label_coords(-0.035, 0.08)
        ax_test_boxnotm.set_xlabel('SHAP values', fontsize=ftsz)
        # ax_test_boxnotm.set_xlabel('SHAP values for the prediction of mouse %s' % mousenb, fontsize=20)
        #ax_test_boxm.set_title('--- mean', loc='right', size=15,
        #                       bbox=dict(facecolor='none', edgecolor='gainsboro', boxstyle='round'))
        #ax_train_boxm.set_title("Training set", fontsize=20)
        #ax_test_boxm.set_title("Test set", fontsize=20)
        plt.xticks(fontsize=ftsz-5)
        plt.tight_layout()
        plt.savefig("./figures_generees/Shap_DepPlot_article/%s_boxTnTe_feat%s_predm%s.png"
                    % (prefixname, feat_nb, mousenb), dpi=300)
        #plt.show()
        plt.close()


listofthemicee = [3,4,6,7,8,9,10,12,13,15,16,17,20,21]
boxplots_manymice_mnotm(featnb, listofthemicee, shapvalues, ytrain, ytest, idxtraintest, selectedfeat_names,
                  dataset_name, prefix_name)


# ----- Valeurs des echelles des axes -----
"""
--- feat 68 ---
# view 1 :
ax1.set_xlim(xmin=0, xmax=47000)
ax1.set_ylim(ymin=-2.6, ymax=1.4)
# view 2 :
ax1.set_xlim(xmin=0, xmax=47000)
ax1.set_ylim(ymin=-6.1, ymax=3.4)

--- feat 80 ---
# view 1 :
ax1.set_xlim(xmin=0, xmax=0.26)
ax1.set_ylim(ymin=-9.5, ymax=7.9)
# view 2 :
ax1.set_xlim(xmin=-0.001, xmax=0.71)
ax1.set_ylim(ymin=-9, ymax=8)

--- feat 85 ---
# view 1 :
ax1.set_xlim(xmin=-0.0005, xmax=0.07)
ax1.set_ylim(ymin=-6.1, ymax=10.1)
# y pour souris 0 figure 5 :
ax1.set_ylim(ymin=-1.4, ymax=3.3)
# view 2 :
ax1.set_xlim(xmin=-0.001, xmax=0.21)
ax1.set_ylim(ymin=-6.1, ymax=10.1)

--- feat 93 ---
# view 1 :
ax1.set_xlim(xmin=7.2, xmax=7.75)
ax1.set_ylim(ymin=-8, ymax=8.1)
# view 2 :
ax1.set_xlim(xmin=6.5, xmax=7.75)
ax1.set_ylim(ymin=-8, ymax=8.1)
"""