import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import argparse
import os
from PIL import Image
import pickle

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import pyfeats as pf


# Extraction de features des images avec PyFeats en lancant les calculs en parallele sur calmip avec chdb.
# Genere : - 1 fichier texte de resume de l'entrainement
#          - 1 fichier pkl avec les features de l'ensemble d'images selectionne.

parser = argparse.ArgumentParser()
parser.add_argument('-name', '--imagename_txt', default='_', type=str)
parser.add_argument('-basenm', '--basename_txt', default='_', type=str)
parser.add_argument('-indir', '--indir_txt', default='_', type=str)
parser.add_argument('-path', '--path_txt', default='_', type=str)
parser.add_argument('-dirnm', '--dirname_txt', default='_', type=str)
parser.add_argument('-outdir', '--outdir_txt', default='_', type=str)
args = parser.parse_args()

# Parameters:
img_name = args.imagename_txt
basename = args.basename_txt
input_dir = args.indir_txt
path = args.path_txt
dirname = args.dirname_txt
output_dir = args.outdir_txt

big_start = time.time()

torch.manual_seed(0)

print('image courante :', img_name)
print('image sans son extension :', basename)
print('dossier courant contenant les images :', input_dir)
print('chemin de l image au sein du dossier courant :', path)
print('chemin des dossiers ou se trouve l image :', dirname)
print('dossier courant dans lequel sont generes les nouveaux fichiers :', output_dir)

# print('environment variables:', os.environ)

chdb_rank = os.environ['CHDB_RANK']
name = os.environ['SLURM_SUBMIT_HOST']
arg = img_name
ntaskspernode = os.environ['SLURM_NTASKS_PER_NODE']
ntaskspercore = os.environ['SLURM_NTASKS_PER_CORE']

jobid = os.environ['SLURM_JOBID']

imgpath = os.path.join(input_dir, path)

pklfile_path = os.path.join(output_dir, dirname, 'features_%s.pkl' % basename)
# pklfile_path = './json_files/features_essai.pkl'
# un fichier pkl par image
# pklfile_path = './pyfeats_ACDEfeatures_names.pkl'
textfile_path = os.path.join(output_dir, 'features_chdb_%s_%s.txt' % (jobid, chdb_rank))
# textfile_path = './text_files/features_essai.txt'
# un fichier txt par travailleur : il faut creer autant de fichiers txt differents qu'il y a de travailleurs sinon
# conflits en lecture et ecriture.

f = open(textfile_path, 'a')

print('\n feature_extraction_chdb.py')
f.write('\n feature_extraction_chdb.py\n')

print('Image path:', imgpath)
f.write('Image path: {val} \n'.format(val=imgpath))

print('pkl file path:', pklfile_path)
f.write('pkl file path: {val}\n'.format(val=pklfile_path))
print('txt file path:', textfile_path)
f.write('txt file path: {val}\n'.format(val=textfile_path))

print("Travailleur numero %s. Tourne sur le serveur %s, fichier %s. %s tache(s) par noeud, %s tache(s) par coeur."
      % (chdb_rank, name, arg, ntaskspernode, ntaskspercore))
f.write("Travailleur numero %s. Tourne sur le serveur %s, fichier %s. %s tache(s) par noeud, %s tache(s) par coeur. \n"
        % (chdb_rank, name, arg, ntaskspernode, ntaskspercore))

print('used environment:', sys.prefix)
f.write('used environment: {val} \n'.format(val=sys.prefix))

print('parameters', args)
f.write('parameters: {val} \n'.format(val=args))

dataset_name = input_dir[4:]  # nom du dataset
print('Dataset name:', dataset_name)
f.write('Dataset name: {val} \n'.format(val=dataset_name))


# --- Debut des calculs pour l'extraction des features ---
# Creation du fichier pkl contenant les informations de l'image courante


# Transformations pour l'augmentation de la donnee : les memes pour les images qui feront l'entrainement et le test
class MyTransformCrop:
    '''Crop at the given position.'''

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, x):
        return F.crop(x, self.top, self.left, self.height, self.width)


transform_crop = MyTransformCrop(top=56, left=0, height=400, width=400)
# on ne recupere qu'un carre de cote 400 colle a gauche de l'image de base et centre en hauteur

deterministic_transform_ch12 = transforms.Compose([transform_crop, transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0], std=[257])])
# Pyfeats veut du 8 bits


# Extract features from image with PyFeats. Print them and save them in pickle file.
def extract_features(img_path):
    image = Image.open(img_path).convert('F')
    image = deterministic_transform_ch12(image)
    image = image[0].numpy()
    # On prend le premier element car sinon on a une shape (1, 400, 400) (on veut (400, 400)).
    # Et tranformation en tableau numpy pour que ca puisse etre utilise par PyFeats.
    # print('Image shape', np.shape(image))
    # f.write('Image shape: {val} \n'.format(val=np.shape(image)))
    # plt.imshow(image, cmap='gray')
    # plt.title('Image')
    # plt.colorbar()
    # plt.show()

    # Masque (pour nous toute l'image)
    mask = np.ones(np.shape(image))
    # print('Mask shape', np.shape(mask))

    # Perimetre du masque
    perimeter = np.zeros(np.shape(image))
    for j in range(400):
        perimeter[0][j] = 1  # premiere ligne avec des 1
        perimeter[399][j] = 1  # derniere ligne avec des 1
        perimeter[j][0] = 1  # premiere colonne avec des 1
        perimeter[j][399] = 1  # derniere colonne avec des 1

    # Number of features working: 414
    features_values = []  # tableau avec les valeurs correspondantes
    features_labels = []  # tableau avec les noms des features calcules
    '''
    # test comptage
    test_values = [np.shape(image)[0]]
    test_labels = ['image shape']
    features_values.extend(test_values)
    features_labels.extend(test_labels)
    '''
    #'''
    # %% A1. Texture features
    # 108 features en tout
    print('A. Texture features')
    f.write('A. Texture features \n')

    # First order statistics
    # 16
    # Caracteristiques calculees a partir de l'histogramme de l'image : moyenne des valeurs des pixels, ecart-type,
    # mediane, mode (valeur de pixel la plus representee), coefficient d'asymetrie, kurtosis (aplatissement), energie
    # (somme des carres des hauteurs pour chaque pixel), entropie (avec du ln), niveau de gris minimal et maximal,
    # coef de variation (ecart-type / moyenne), percentiles et largeur de l'histogramme.
    print('fos')
    f.write('fos \n')
    afos_values, afos_labels = pf.fos(image, mask)
    features_values.extend(afos_values)
    features_labels.extend(afos_labels)

    # Gray level co-occurence matrix / Spatial Gray Level Dependence Matrix
    # 28
    # Liens du second ordre entre les valeurs des pixels selon leurs positions relatives. Pour chaque mesure, calcul de
    # la valeur moyenne et de l'amplitude des valeurs. Mesures : moment angulaire de second ordre (carre de l'energie),
    # contraste, correlation, variance, homogeneite, somme ponderee, somme des variances, somme des entropies, entropie,
    # difference des variances, difference des entropies, mesures de correlation.
    print('glcm')
    f.write('glcm \n')
    A_GLCM_values_mean, A_GLCM_values_range, A_GLCM_labels_mean, A_GLCM_labels_range\
        = pf.glcm_features(image, ignore_zeros=False)
    features_values.extend(A_GLCM_values_mean)
    features_labels.extend(A_GLCM_labels_mean)
    features_values.extend(A_GLCM_values_range)
    features_labels.extend(A_GLCM_labels_range)

    # Gray Level Difference Statistics
    # 5
    # Mesures de textures de l'image obtenues par des statistiques d'ordre 1 sur des proprietes locales de l'image
    # elles-memes obtenues par des differences entre paires de valeurs de pixels en nuances de gris : homogeneite,
    # contraste, energie, entropie, moyenne.
    print('glds')
    f.write('glds \n')
    aglds_values, aglds_labels = pf.glds_features(image, mask, Dx=[0, 1, 1, 1], Dy=[1, 1, 0, -1])
    features_values.extend(aglds_values)
    features_labels.extend(aglds_labels)

    # Neighborhood Gray Tone Difference Matrix (visual properties of texture)
    # 5
    # Mesures de proprietes visuelles de textures de l'image (par rapport aux tons de nuances de gris) : rugosite,
    # contrast, degre d'occupation, complexite, force.
    print('ngtdm')
    f.write('ngtdm \n')
    angtdm_values, angtdm_labels = pf.ngtdm_features(image, mask, d=1)
    features_values.extend(angtdm_values)
    features_labels.extend(angtdm_labels)

    # Statistical Feature Matrix
    # 4
    # Mesures a partir d'une matrice dont les valeurs sont des proprietes statistiques de la paire de pixels
    # correspondant (i, j) : rugosite, contraste, periodicite, irregularites de surface.
    print('sfm')
    f.write('sfm \n')
    asfm_values, asfm_labels = pf.sfm_features(image, mask, Lr=4, Lc=4)
    features_values.extend(asfm_values)
    features_labels.extend(asfm_labels)

    # Law's Texture Energy Measures
    # 6
    # Mesures de textures obtenues de convolutions entre l'image et des vecteurs de transformation simples (moyenne
    # locale centree, difference entre pixels pour la detection d'aretes et difference entre pixels pour la detection de
    # taches/points) : mesures d'energie de texture avec differentes combinaisons de ces transformations, moyennees et
    # non-moyennees.
    print('lte')
    f.write('lte \n')
    alte_values, alte_labels = pf.lte_measures(image, mask, l=7)
    features_values.extend(alte_values)
    features_labels.extend(alte_labels)

    # Fractal Dimension Texture Analysis
    # 4
    # Mesure de la rugosite de la surface de l'image en se basant sur le modele de mouvement brownien fractionmaire :
    # coefficient de Hurst H, qui est inversement proportionnel a la dimension fractale D (surface lisse => D faible et
    # H eleve).
    print('fdta')
    f.write('fdta \n')
    afdta_values, afdta_labels = pf.fdta(image, mask, s=3)
    features_values.extend(afdta_values)
    features_labels.extend(afdta_labels)

    # Gray Level Run Length Matrix
    # 11
    # Gray level run = un chemin de pixels consecutifs ayant la meme valeur (en niveau de gris). On met dans la matrice
    # la longueur de ces chemins en fonction de leur position dans l'image. Mesures sur cette matrice : mesure des
    # chemins courts, des chemins longs, distribution des niveaux de gris, distribution des longueurs de chemins,
    # proportion de chemins, mesure des chemins avec un faible niveau de gris, de ceux avec un niveau de gris eleve, de
    # ceux courts et de ceux longs avec un faible niveau de gris, de ceux courts et de ceux longs avec un niveau de gris
    # eleve.
    print('glrlm')
    f.write('glrlm \n')
    aglrlm_values, aglrlm_labels = pf.glrlm_features(image, mask, Ng=256)
    features_values.extend(aglrlm_values)
    features_labels.extend(aglrlm_labels)

    # Fourier Power Spectrum / Densite spectrale de puissance de Fourier
    # 2
    # La distribution radiale de ce spectre est sensible a la rugosite de l'image, sa distribution angulaire est
    # sensible a la directionalite de tecture de l'image (on considere la transformee de Fourier discrete). Mesures :
    # sommes radiale et angulaire.
    print('fps')
    f.write('fps \n')
    afps_values, afps_labels = pf.fps(image, mask)
    features_values.extend(afps_values)
    features_labels.extend(afps_labels)

    # Shape parameters
    # 5
    # Mesures : longueurs maximales selon x et y, aire, perimetre, carre du perimetre / aire.
    print('shape')
    f.write('shape \n')
    ashape_values, ashape_labels = pf.shape_parameters(image, mask, perimeter, pixels_per_mm2=2409.06)
    # pixels_per_mm2 = nombre de pixels qu'on peut aligner (une seule ligne) sur une longueur de 1mm.
    # Par def de nos images, 1 pixel a un cote de longueur 0.4151 microns et comme il n'y a que du rognage et pas de
    # resize, c'est encore le cas pour 'image'.
    # Donc pixels_per_mm2 = 1 / (0.4151*10^(-3)) = 2409.06.
    features_values.extend(ashape_values)
    features_labels.extend(ashape_labels)

    # Gray Level Size Zone Matrix
    # 14
    # Etude des zones de niveaux de gris, ie de voxels connectes avec la meme valeur de niveau de gris. Matrice
    # invariante par rotation. Mesures : mesure des zones de faible et grande surface, non-uniformite des niveaux de
    # gris, non-uniformite des tailles des zones, proportion de zones, mesure des zones de faible niveau de gris et de
    # niveau de gris eleve, mesure des petites et grandes zones de faible niveau de gris, mesure des petites et grandes
    # zones de niveau de gris eleve, variance des niveaux de gris, variance de la taille des zones, entropie de la
    # taille des zones.
    print('glszm')
    f.write('glszm \n')
    aglszm_values, aglszm_labels = pf.glszm_features(image, mask)
    features_values.extend(aglszm_values)
    features_labels.extend(aglszm_labels)

    # Higher Order Spectra
    # 2
    # Base sur la transformee de Radon qui envoie une image avec des lignes dans un espace ou chaque ligne est
    # representee par un point. On considere le bispectre de la transformee de Radon de l'image, ou le bispectre est la
    # transformee de Fourier de la correlation de 3e ordre de l'image. L'HOS regarde les moments les plus eleves des
    # composants de ce bispectre. Mesure : entropie du bispectre.
    print('hos')
    f.write('hos \n')
    ahos_values, ahos_labels = pf.hos_features(image, th=[135, 140])
    features_values.extend(ahos_values)
    features_labels.extend(ahos_labels)

    # Local Binary Pattern
    # 6
    # Description "robuste et efficace" de la texture de l'image. Mesure de l'uniformite de l'image en considerant les
    # pixels equidistants d'un pixel commmun et en regardant s'ils ont des valeurs plus grandes ou plus petites que
    # celle du pixel central. Mesures : energie et entropie de l'image obtenue par la transformation LBP.
    print('lbp')
    f.write('lbp \n')
    albp_values, albp_labels = pf.lbp_features(image, image, P=[8, 16, 24], R=[1, 2, 3])
    features_values.extend(albp_values)
    features_labels.extend(albp_labels)

    # image_name = 'name of the image used'
    # pf.plot_sinogram(image, image_name)
    # plt.savefig('./figures_generees/SVM_sinogram.png')

    # %% B. Morphological features
    # 60
    # le multilevel binary fait une erreur de division par zero.
    print('B. Morphological features')
    f.write('B. Morphological features \n')

    # Grayscale Morphological Analysis
    # 60
    # Pareil que multilevel binary morphological analysis mais pour images en nuances de gris.
    # Mesures : fonctions moyennes de distribution cumulative et fonctions moyennes de densite de probabilite.
    print('grayscale')
    f.write('grayscale \n')
    bgraypdf_values, bgraycdf_values = pf.grayscale_morphology_features(image, N=30)
    labels_pdf = ['Morpho_Gray_pdf_%s' % str(i) for i in range(1, 31)]
    labels_cdf = ['Morpho_Gray_cdf_%s' % str(i) for i in range(1, 31)]
    features_values.extend(bgraypdf_values)
    features_labels.extend(labels_pdf)
    features_values.extend(bgraycdf_values)
    features_labels.extend(labels_cdf)

    # Multilevel Binary Morphological Analysis
    # 90
    # Proprietes geometriques de l'image. On extrait de l'image originelle 3 sous-images composees resp. des pixels de
    # faible intensite, de ceux d'intensite moyenne et de ceux de forte intensite puis on calcule leur spectre.
    # On ne l'utilise pas car pour images couleur.
    '''
    print('multilevel binary')
    f.write('multilevel binary \n')
    blpdf_values, bmpdf_values, bhpdf_values, blcdf_labels, bmcdf_labels, bhcdf_labels\
        = pf.multilevel_binary_morphology_features(image, mask, N=30, thresholds=[25, 50])
    features_values.extend(blpdf_values)
    features_labels.extend(blcdf_labels)
    features_values.extend(bmpdf_values)
    features_labels.extend(bmcdf_labels)
    features_values.extend(bhpdf_values)
    features_labels.extend(bhcdf_labels)
    
    # pf.plot_pdf_cdf(bgraypdf_values, bgraycdf_values, img_path)
    # plt.savefig('./figures_generees/SVM_pdfcdf.png')
    # pf.plot_pdfs_cdfs(features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'],
    #                  features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'],
    #                  features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'])
    # plt.savefig('./figures_generees/SVM_pdfscdfs.png')
    '''

    # %% C. Histogram Based features
    # 2176
    print('C. Histogram Based features')
    f.write('C. Histogram Based features \n')

    # Histogram
    # 32
    # Histogramme en niveaux de gris de l'image avec 32 barres de largeur egale.
    print('histogram')
    f.write('histogram \n')
    chist_values, chist_labels = pf.histogram(image, mask, bins=32)
    features_values.extend(chist_values)
    features_labels.extend(chist_labels)

    # Multi-region histogram
    # 96
    # Histogrammes de trois zones de l'image.
    print('multiregion_histogram')
    f.write('multiregion_histogram \n')
    cmhist_values, cmhist_labels = pf.multiregion_histogram(image, mask, bins=32, num_eros=3, square_size=3)
    features_values.extend(cmhist_values)
    features_labels.extend(cmhist_labels)

    # Correlogram
    # 2048
    # Correlogramme = histogramme qui mesure non seulement des caracteristiques de l'image mais prend aussi en compte la
    # distribution spatiale de ces caracteristiques. On considere ici deux correlogrammes : un base sur la distance au
    # centre de l'image et un sur la distribution angulaire, chacun etant in fine une matrice.
    print('correlogram')
    f.write('correlogram \n')
    ccorhd_values, ccorht_values, ccor_labels = pf.correlogram(image, mask, bins_digitize=32, bins_hist=32, flatten=True)
    features_values.extend(ccorhd_values)
    features_labels.extend(ccor_labels)
    features_values.extend(ccorht_values)
    features_labels.extend(ccor_labels)

    # pf.plot_histogram(image, mask, bins=1000, Ng=400, name=img_name)
    # plt.title('Histogram')
    # plt.savefig('./figures_generees/SVM_histogram.png')
    # pf.plot_correlogram(image, mask, bins_digitize=32, bins_hist=32, name=image_name)
    # plt.savefig('./figures_generees/SVM_correlogram.png')
    
    # %% D. Multi-Scale features
    # 306
    print('D. Multi-Scale features')
    f.write('D. Multi-Scale features \n')

    # Discrete Wavelet Transform
    # 18
    # Transformations de l'image en la multipliant avec certaines fonctions, sur ses lignes et colonnes, et en faisant
    # des convolutions avec des filtres passe-bas qui permet de decomposer l'image en sous-images : image
    # d'approximation, image detaillee sur l'horizontale, sur la verticale et sur la diagonale. Mesures : moyenne et
    # ecart-type des sous-images.
    print('dwt')
    f.write('dwt \n')
    ddwt_values, ddwt_labels = pf.dwt_features(image, mask, wavelet='bior3.3', levels=3)
    features_values.extend(ddwt_values)
    features_labels.extend(ddwt_labels)

    # Stationary Wavelet Transform
    # 18
    # Proche de discrete wavelet transform mais sans baisse de precision (pas d'image d'approximation), ce qui permet
    # d'etre invariant par translation, mais plus cher en ressources.
    print('swt')
    f.write('swt \n')
    dswt_values, dswt_labels = pf.swt_features(image, mask, wavelet='bior3.3', levels=3)
    features_values.extend(dswt_values)
    features_labels.extend(dswt_labels)

    # Wavelet Packets
    # 126
    # Proche de discrete wavelet transform mais avec une meilleure representation 'espace-frequence'.
    print('wp')
    f.write('wp \n')
    dwp_values, dwp_labels = pf.wp_features(image, mask, wavelet='coif1', maxlevel=3)
    features_values.extend(dwp_values)
    features_labels.extend(dwp_labels)

    # Gabor Transform
    # 16
    # Convolution entre l'image et la fonction de Gabor. Sort des features interessants pour segmentation de textures
    # et classification.
    print('gt')
    f.write('gt \n')
    dgt_values, dgt_labels = pf.gt_features(image, mask, deg=4, freq=[0.05, 0.4])
    features_values.extend(dgt_values)
    features_labels.extend(dgt_labels)

    # Amplitude Modulation – Frequency Modulation
    # 128
    # Application de la transformee d'Hilbert a l'image puis de filtres basse-bande qui fournit l'amplitude, la phase
    # et la frequence instantanees. Mesures : histogramme des images reconstruites basse, moyemme, haute et dc (?).
    print('pf')
    f.write('pf \n')
    damfm_values, damfm_labels = pf.amfm_features(image, bins=32)
    features_values.extend(damfm_values)
    features_labels.extend(damfm_labels)

    # %% E. Other
    # 32
    print('E. Other features')
    f.write('E. Other features \n')

    # Histogram of Oriented Gradients
    # 186624
    # Aucune doc la-dessus ! On ne l'utilise pas car on ne sait pas ce que c'est et qu'il y a enormement de features.
    '''
    print('hog')
    f.write('hog \n')
    ehog_values, ehog_labels = pf.hog_features(image, ppc=8, cpb=3)
    features_values.extend(ehog_values)
    features_labels.extend(ehog_labels)
    '''

    # Hu’s Moments
    # 7
    # Des moments invariants par translation, echelle et rotation.
    print('hu')
    f.write('hu \n')
    ehu_values, ehu_labels = pf.hu_moments(image)
    features_values.extend(ehu_values)
    features_labels.extend(ehu_labels)

    # Threshold Adjacency Matrix
    # Pas de doc la-dessus donc on ne l'utilise pas, et aussi car donne l'erreur 'This function only accepts integer
    # types (passed array of type float32)'
    # features['E_TAS'] = pf.tas_features(image)

    # Zernikes’ Moments
    # 25
    # Des moments orthogonaux invariants par translation.
    print('zernikes')
    f.write('zernikes \n')
    ezernikes_values, ezernikes_labels = pf.zernikes_moments(image, radius=9)
    features_values.extend(ezernikes_values)
    features_labels.extend(ezernikes_labels)

    # %% Print
    # print('%s Values:' % len(features_values), features_values)
    # print('%s Labels:' % len(features_labels), features_labels)

    return features_values, features_labels


features_img = {}
feat_values, _ = extract_features(imgpath)
# _, feat_labels = extract_features(imgpath)
# print(basename, feat_values, feat_labels)
# f.write('basename, values, labels: {val1}, {val2}, {val3} \n'.format(val1=basename, val2=feat_values,
# val3=feat_labels))
features_img[basename] = feat_values

print('Nombre de features extraits :', len(feat_values))
f.write('Nombre de features extraits : {val} \n'.format(val=len(feat_values)))

# Saving the features in pickle file

with open(pklfile_path, 'wb') as g:
    pickle.dump(features_img, g)
    # pickle.dump(feat_labels, g)


big_end = time.time()
f.write('Total run time in s: {val:0.3f} \n'.format(val=(big_end - big_start)))

f.close()
