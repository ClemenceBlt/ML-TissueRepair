import os
import operator
import numpy as np

echeance_list = ["j03", "j10"]
kfolds = 5  # nb de folds

dataset_name = 'tiff_230323_J3PLJ10PL_trainmosTe'
path = "/home/begonias/Documents/Images/Jeux_donnees/"  # chemin pour arriver au dossier d'images

method = 'divisible_mosaic'
# divisible_mosaic : tiles issues d'une meme mosaique peuvent se retrouver dans l'ensemble train et dans le test. Toutes
# les tiles sont considerees comme independantes, la repartition ne tient pas du tout compte de leur appartenance a une
# meme mosaique


if len(echeance_list) > 1:
    f = open('./text_files/folds%s_%s%s_%s_%s.txt' % (kfolds, echeance_list[0], echeance_list[1], dataset_name[5:], method), 'w')
else:
    f = open('./text_files/folds%s_%s_%s_%s.txt' % (kfolds, echeance_list[0], dataset_name[5:], method), 'w')

allfoldsnalox = []
# liste de taille kfolds dont chaque element est une liste des couples des images naloxone de ce fold :
# (nom de l'img czi/lsm, nombre d'images tif a deux canaux correspondantes)
# eg allfoldsnalox[0] = [('nalox_j03_ztt_manip19_OD_afm1.lsm', 278), ('nalox_j10_ztt_manip27_ODD_mos140_tiles', 19)]
# est la liste des images naloxone pour le fold 0
allfoldsnalox_len = []
# liste de taille kfolds avec le nb d'img nalox tiff de chaque fold
# eg [297, 302, 299, 302, 439]
allfoldspl = []  # pareil pour la petite lipectomie
allfoldspl_len = []

all_len_width = {'nalox_j03_ztt_manip17_OD_mos160': (16, 10), 'nalox_j03_ztt_manip19_OD_mos130': (26, 5),
                 'nalox_j03_ztt_manip19_OG_mos135': (27, 5), 'nalox_j03_ztt_manip20_OG_mos153': (17, 9),
                 'nalox_j03_ztt_manip21_ODD_mos225': (25, 9), 'ptlip_j03_ztt_manip20_mos195': (13, 15),
                 'ptlip_j03_ztt_manip21_acqu1_mos128': (16, 8), 'ptlip_j03_ztt_manip21_acqu2_mos91': (13, 7),
                 'ptlip_j03_ztt_manip26_mos146': (16, 10), 'ptlip_j03_ztt_manip28_mos240': (30, 8),
                 'nalox_j10_ztt_manip27_ODD_mos140': (14, 10), 'nalox_j10_ztt_manip27_OG_mos120': (15, 8),
                 'nalox_j10_ztt_manip27_OGG_mos100': (20, 5), 'nalox_j10_ztt_manip30_ODD_mos144': (16, 9),
                 'nalox_j10_ztt_manip22_OG_mos168': (14, 12), 'nalox_j10_ztt_manip19_OG_mos252': (28, 9),
                 'ptlip_j10_ztt_manip20_OG_mos200': (20, 10), 'ptlip_j10_ztt_manip25_OD_mos270': (18, 15),
                 'ptlip_j10_ztt_manip25_OG_mos52': (13, 4), 'ptlip_j10_ztt_manip27_mos45': (5, 9),
                 'ptlip_j10_ztt_manip27_OD_mos85': (17, 5), 'ptlip_j10_ztt_manip27_OGG_mos203': (29, 7)
                 }
# pour chaque image en mosaique, couple (longueur, hauteur)

img_s_numbering = ["nalox_j10_ztt_manip27_OGG_mos100", "nalox_j10_ztt_manip20_tile2x10",
                   "nalox_j03_ztt_manip30_ODD_tile2x7", "nalox_j03_ztt_manip30_ODD_tile2x2"]
# liste des prefixes de mosaiques ou la numerotation est en S et pas en L


for classe in ["nalox", "ptlip"]:
    img_features_j3 = {}
    img_features_j10 = {}
    # dictionnaire : cles = noms lsm-czi des images, valeurs = nombre d'images tiff a 2 canaux correspondant
    for echeance in echeance_list:
        # on construit independemment les tableaux pour naloxone et petite lipectomie et on associe les nalox du fold 0
        # avec les petite lipectomie du fold 0

        used_img_dir = os.path.join(path, dataset_name, echeance, classe)

        # - Creation de la liste des images (acquisitions lsm-czi) avec le nombre d'images tiff a deux canaux associe -
        list_images = list(os.listdir(os.path.join(used_img_dir)))

        prefix_regrouper = []
        # Nombre d'images correspondant a chacun de ces prefixes
        len_prefix = []
        img_to_del = []

        # Construction du dictionnaire : ajout des acquisitions mosaiques
        if echeance == 'j03':
            img_feature_echeance = img_features_j3
        else:  # j10
            img_feature_echeance = img_features_j10

        for idx, img_name in enumerate(prefix_regrouper):
            img_feature_echeance[img_name + "_tiles"] = len_prefix[idx]
            # ajout du suffixe tiles pr rappeler que ces acquisitions sont issues de la reunion de plsrs mosaiques

        # Construction du dictionnaire : ajout des autres acquisitions
        for image in list_images:  # parcours des dossiers lsm/czi
            if image not in img_to_del:
                img_feature_echeance[image] = len(list(os.listdir(os.path.join(used_img_dir, image, "ch1"))))

    img_features = {**img_features_j3, **img_features_j10}  # concatenation des deux dictionnaires
    # Tri du dictionnaire par ordre croissant de ses valeurs (le nombre d'images tiff de chaque acquisition)
    img_features = dict(sorted(img_features.items(), key=operator.itemgetter(1)))

    # Transformation en liste car plus simple a manipuler dans la suite
    img_features_sorted = list(img_features.items())
    print("Liste initiale des couples J3 et J10 " + str(classe) + " (nom image, nb de tiff) :\n", img_features_sorted)

    resu = 0
    for _, nb in img_features_sorted:
        resu += nb
    print("Nb imgs lsm czi " + str(classe), len(img_features_sorted))
    print("Nb imgs tiff a un canal " + str(classe), 2*resu)

    # -- Repartition des images dans les folds -- #

    # Etape 1 : on met les kfolds plus grosses acquisitions dans des folds distincts

    for i in range(kfolds):
        if len(img_features_sorted)>0:  # pour cas ou la liste serait vide
            img_name, img_len = img_features_sorted.pop()  # on enleve les kfolds plus grosses acquisitions
            if classe == "nalox":
                allfoldsnalox.append([(img_name, img_len)])
                allfoldsnalox_len.append(img_len)
            else:
                allfoldspl.append([(img_name, img_len)])
                allfoldspl_len.append(img_len)

    # Etape 2 : on ajoute la plus grande acquisition restante au fold avec le moins d'images tiff

    while len(img_features_sorted) > 0:
        if classe == "nalox":
            valeur_min = min(allfoldsnalox_len)  # nb d'images tiff minimale parmi les folds
            fold_min = allfoldsnalox_len.index(valeur_min)  # numero du fold avec le moins d'images
            new_img_name, new_img_len = img_features_sorted.pop()
            # on recupere l'acquisition avec le plus d'images parmi celles qui restent
            allfoldsnalox[fold_min].append((new_img_name, new_img_len))
            allfoldsnalox_len[fold_min] += new_img_len
        else:
            valeur_min = min(allfoldspl_len)  # nb d'images tiff minimale parmi les folds
            fold_min = allfoldspl_len.index(valeur_min)  # numero du fold avec le moins d'images
            new_img_name, new_img_len = img_features_sorted.pop()
            # on recupere l'acquisition avec le plus d'images parmi celles qui restent
            allfoldspl[fold_min].append((new_img_name, new_img_len))
            allfoldspl_len[fold_min] += new_img_len


print("allfoldsnalox:", allfoldsnalox)
print("allfoldsnaloxlen", allfoldsnalox_len)
print("allfoldspl:", allfoldspl)
print("allfoldspllen", allfoldspl_len)

# Gestion des cas ou il n'y a pas de nalox ou pas de ptlip
if len(allfoldsnalox) == 0:
    allfoldsnalox = [[], [], [], [], []]
    allfoldsnalox_len = [0, 0, 0, 0, 0]
elif len(allfoldspl) == 0:
    allfoldspl = [[], [], [], [], []]
    allfoldspl_len = [0, 0, 0, 0, 0]


# Construction liste finale et ecriture dans le fichier texte

allfolds = [[] for _ in range(kfolds)]

for i in range(kfolds):
    for elmt in allfoldsnalox[i]:
        name, _ = elmt
        allfolds[i].append(name)
    for elmt in allfoldspl[i]:
        name, _ = elmt
        allfolds[i].append(name)

    print("Images du fold " + str(i) + " :", allfolds[i])
    print("Nb d'images - nalox : {lenn}, petite lip : {lenpl}, total : {tot}"
          .format(lenn=allfoldsnalox_len[i], lenpl=allfoldspl_len[i], tot=allfoldsnalox_len[i] + allfoldspl_len[i]))

print("Tous les folds (allfolds):\n", allfolds)
# f.write("allfolds : \n")
f.write("{val}\n".format(val=allfolds))
# f.write("Nombres d'images tiff de naloxone dans les folds : \n")
# f.write("{val}\n".format(val=allfoldsnalox_len))
# f.write("Nombres d'images tiff de petite lipectomie dans les folds : \n")
# f.write("{val}\n".format(val=allfoldspl_len))

f.close()
