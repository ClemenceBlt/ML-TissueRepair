import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, ConcatDataset
from torch import nn

from ast import literal_eval
import os
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import time
import sys
import json
import ast
import shutil

import h5py

import argparse

# Fichier pour lancer entrainement sur ResNet, SqueezeNet ou vgg depuis images qu'on enregistre dans fichier
# h5 decoupees en folds. Entrainement pour un fold de validation specifie. Validation croisee ; enregistrement du modele
# a la meilleure performance et en fin d'entrainement ; possibilite de reprendre un entrainement apres interruption (en
# reglant start_epoch et last_epoch).
# Genere : - 1 fichier texte de resume de l'entrainement
#          - 1 sous-dossier dans "tar_files" au nom de l'entrainement, lui-meme contenant un dossier best_models et un
# dossier checkpoints. best_models contient pour le fold le checkpoint du meilleur modele. checkpoints contient pour
# le fold le checkpoint a la derniere epoque.
#          - 1 fichier json avec le dictionnaire des valeurs des loss et accuracy pour le fold.
#          - 2 fichiers png avec loss et acc tracees (se trouvent dans le dossier ecm/figures/).

parser = argparse.ArgumentParser()
parser.add_argument('-sep', '--startepoch', default=0, type=int)
parser.add_argument('-lep', '--lastepoch', default=1, type=int)
parser.add_argument('-validfld', '--validfold_nb', default=0, type=int)
parser.add_argument('-suf', '--suffix_txt', default='_', type=str)
parser.add_argument('-lr', '--learningrate', default=1e-4, type=float)
parser.add_argument('-bs', '--batchsize', default=16, type=int)
parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float)
args = parser.parse_args()

big_start = time.time()

# parametres :
start_epoch = args.startepoch
last_epoch = args.lastepoch
valid_fold_nb = args.validfold_nb
suffix = args.suffix_txt
learning_rate = args.learningrate
batch_size = args.batchsize
w_decay = args.weight_decay

torch.manual_seed(0)

# dossier d'images utilise :
echeance_list = ["j03", "j10"]

echeance_name = ""
for day in echeance_list:
    echeance_name += day

method = 'divisible_mosaic'
shortmethod = 'divmos'

used_img_dir = "./tiff_230323_J3PLJ10N_trainmosTV"
# dataset_name = used_img_dir[10:]
dataset_name = '230323_J3PLJ10N_trainmosTV'

# neuralnet_used = 'ResNet'
neuralnet_used = 'SqueezeNet'
# neuralnet_used = 'vgg11'

if suffix == '_':
    prefix_name = 'train_ch12_%s_%s_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s_%s' \
                  % (echeance_name, neuralnet_used, dataset_name, start_epoch, last_epoch, batch_size, learning_rate,
                     w_decay, valid_fold_nb, shortmethod)
    prefix_name_toload = 'train_ch12_%s_%s_%s_ep0_%s_bs%s_lr%s_wd%s_validfld%s_%s' \
                         % (echeance_name, neuralnet_used, dataset_name, start_epoch - 1, batch_size, learning_rate,
                            w_decay, valid_fold_nb, shortmethod)
    # prefix_name_toload = 'train_ch12_%s_%s_%s_ep100_%s_bs%s_lr%s_wd%s_validfld%s_%s' \
    #                     % (echeance_name, neuralnet_used, dataset_name, start_epoch - 1, batch_size, learning_rate,
    #                        w_decay, valid_fold_nb, shortmethod)
else:
    prefix_name = 'train_ch12_%s_%s_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s_%s_%s' \
                  % (echeance_name, neuralnet_used, dataset_name, start_epoch, last_epoch, batch_size, learning_rate,
                     w_decay, valid_fold_nb, shortmethod, suffix)
    prefix_name_toload = 'train_ch12_%s_%s_%s_ep0_%s_bs%s_lr%s_wd%s_validfld%s_%s_%s' \
                         % (echeance_name, neuralnet_used, dataset_name, start_epoch - 1, batch_size, learning_rate,
                            w_decay, valid_fold_nb, shortmethod, suffix)
    # prefix_name_toload = 'train_ch12_%s_%s_%s_ep100_%s_bs%s_lr%s_wd%s_validfld%s_%s_%s' \
    #                     % (echeance_name, neuralnet_used, dataset_name, start_epoch - 1, batch_size, learning_rate,
    #                        w_decay, valid_fold_nb, shortmethod, suffix)

f = open('./text_files/%s.txt' % prefix_name, 'w')

f.write('train_ch12_allNets_fldbyfld.py\n')
print('train_ch12_allNets_fldbyfld.py')
print('used environment:', sys.prefix)
f.write('used environment: {val} \n'.format(val=sys.prefix))

print("Method:", method)
print("Used image directory:", used_img_dir)
f.write('Method: {val} \n'.format(val=method))
f.write('Used image directory: {val} \n'.format(val=used_img_dir))

# creation dossiers qui contiendront les sauvegardes du modele (uniquement s'ils n'existent pas deja) :
os.makedirs('./tar_files/%s/best_models' % prefix_name, exist_ok=True)
os.makedirs('./tar_files/%s/checkpoints' % prefix_name, exist_ok=True)

print('parameters', args)
f.write('parameters: {val} \n'.format(val=args))

print('Used neural network:', neuralnet_used)
f.write('Used neural network: {val} \n'.format(val=neuralnet_used))

# Mise en place de l'implementation sur GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Used device (CPU or GPU with cuda):', device)
f.write('Used device (CPU or GPU with cuda): {val} \n'.format(val=device))

kfolds = 5

# ---- Creation des fichiers HDF5 ---- #

path_h5 = './h5_files/dataset%s_%sfolds_%s_ch12_%s.h5' % (dataset_name, str(kfolds), echeance_name, shortmethod)
# path_h5 = './h5_files/dataset230323_trainmosTe_5folds_j03j10_ch12_divmos.h5'
# path_h5 = './h5_files/dataset_revgrad_230330_16mice_5folds_j03j10_ch12.h5'
print("Used h5 file:", path_h5)
f.write("Used h5 file: {val} \n".format(val=path_h5))

"""
print('Making HDF5 files')
f.write('Making HDF5 files \n')
start_file = time.time()

# -- Listes des noms des images composant chaque fold --
txt_file = open('./text_files/folds%s_%s_%s_%s.txt' % (str(kfolds), echeance_name, dataset_name, shortmethod))
txt_data = txt_file.readlines()
txt_file.close()

allfolds_tiles = ast.literal_eval(txt_data[0])

for i in range(kfolds):
    f.write("Fold {nb} : {val}\n".format(nb=i, val=allfolds_tiles[i]))

# Nombre de folds
# kfolds = len(allfolds_tiles)

# Construction d'un tableau allfolds ou numeros des tiles sont detailles (elements du tableau sont des noms d'images
# lsm-czi) et pas regroupes ensemble avec le suffixe _tiles
allfolds = [[] for _ in range(kfolds)]

for i in range(kfolds):
    for name in allfolds_tiles[i]:  # parcours des images du fold i
        prefix = name[:-6]
        classe_npl = name[:5]
        echeance_j310 = name[6:9]
        # on recupere le prefixe de ces images (c'est tout le nom sauf le "tiles" final pour img avec tiles)
        allfolds[i].append(name)

# print("allfolds :", allfolds)
# print("Fold 0 :", allfolds[0])
# print("Fold 1 :", allfolds[1])
# print("Fold 2 :", allfolds[2])
# print("Fold 3 :", allfolds[3])
# print("Fold 4 :", allfolds[4])
"""
"""
# -- Valeurs des moyennes et ecart-types --

# Valeurs pour 220902_nomos
if len(echeance_list) == 1:
    if echeance_list[0] == "j03":
        # J3
        mean_dataset_ch1 = 1937.37
        std_dataset_ch1 = 5610.18
        mean_dataset_ch2 = 933.626
        std_dataset_ch2 = 2404.23
    else:
        # J10
        mean_dataset_ch1 = 3676.36
        std_dataset_ch1 = 6993.60
        mean_dataset_ch2 = 771.406
        std_dataset_ch2 = 2057.07
else:
    # J3 + J10
    mean_dataset_ch1 = 2720.86
    std_dataset_ch1 = 6330.76
    mean_dataset_ch2 = 860.540
    std_dataset_ch2 = 2255.89


# Valeurs pour 220902_mos
if len(echeance_list) == 1:
    if echeance_list[0] == "j03":
        # J3
        mean_dataset_ch1 = 1187.36
        std_dataset_ch1 = 4473.55
        mean_dataset_ch2 = 938.587
        std_dataset_ch2 = 2245.77
    else:
        # J10
        mean_dataset_ch1 = 1985.94
        std_dataset_ch1 = 5653.84
        mean_dataset_ch2 = 751.445
        std_dataset_ch2 = 2288.51
else:
    # J3 + J10
    mean_dataset_ch1 = 1598.21
    std_dataset_ch1 = 5130.46
    mean_dataset_ch2 = 842.306
    std_dataset_ch2 = 2269.79
"""


"""
# -- Creation de la classe personnalisee pour le rognage --
# rogne sur la gauche, pour ne pas prendre en compte les bandes noires à droite des images

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


# -- Creation du fichier contenant les informations de l'ensemble d'entrainement (images et labels) -- #

# Transformations pour l'augmentation de la donnee : les memes pour les images qui feront l'entrainement et la validation
# deterministic_transform_ch1 = transforms.Compose([transforms.Resize(400), transforms.ToTensor(),
#                                                  transforms.Normalize(mean=[mean_dataset_ch1],
#                                                                       std=[std_dataset_ch1])])

# deterministic_transform_ch2 = transforms.Compose([transforms.Resize(400), transforms.ToTensor(),
#                                                  transforms.Normalize(mean=[mean_dataset_ch2],
#                                                                       std=[std_dataset_ch2])])

deterministic_transform_ch12 = transforms.Compose([transform_crop, transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0], std=[2**16-1])])

# Liste composee des noms des images du dossier images/echeance :
# list_img = list(os.listdir(os.path.join(used_img_dir, 'nalox'))) \
#            + list(os.listdir(os.path.join(used_img_dir, 'ptlip')))
list_img = []
for classe in ['nalox', 'ptlip']:
    for echeance in echeance_list:
        list_img += list(os.listdir(os.path.join(used_img_dir, echeance, classe)))


# Calcul du nombre total d'images a deux canaux pour chaque fold :
len_dataset = [0 for i in range(kfolds)]  # tableau dont l'indice i est le nombre d'images du fold i
for i in range(kfolds):
    for lsmczi_img_name in allfolds[i]:
        len_dataset[i] += len(os.listdir(os.path.join(used_img_dir, lsmczi_img_name[6:9], lsmczi_img_name[:5],
                                                      lsmczi_img_name, 'ch1')))
        # suppose qu'il y a toujours autant d'images ch1 que ch2

print("Taille des fold :", len_dataset)
f.write("Taille des fold : {val} \n".format(val=len_dataset))


# --- Creation des listes d'images et de labels qui vont constituer les fichiers HDF5 --- #

def build_all_img_labels(folds_number, all_folds, img_dir, lendataset, transform_ch12):

    # Initialisation des listes
    allimages = []
    alllabels = []
    for kk in range(folds_number):
        allimages.append(np.empty((lendataset[kk], 2, 400, 400), dtype='f'))
        alllabels.append(np.empty(lendataset[kk], dtype=bool))

    # Remplissage des listes
    for k in range(folds_number):  # parcours des folds successifs
        print("Fold en cours :", k)
        cpt = 0  # compteur du nombre d'images (a deux canaux) dans chaque fold
        for lsmczi_imgname in all_folds[k]:  # parcours des images du fold k
            print("Image en cours :", lsmczi_imgname)
            tif_ch1_names = os.listdir(os.path.join(img_dir, lsmczi_imgname[6:9], lsmczi_imgname[:5], lsmczi_imgname,
                                                    'ch1'))
            # noms des images (ch1) constituant les couches z de l'image consideree

            for tif_ch1img_name in tif_ch1_names:
                # print("Fold :", k, "- Indice dans all_images[fold] :", cpt, "- Nom image :", tif_ch1img_name)

                img_path_ch1 = os.path.join(img_dir, lsmczi_imgname[6:9], lsmczi_imgname[:5], lsmczi_imgname, 'ch1',
                                            tif_ch1img_name)
                image_ch1 = Image.open(img_path_ch1).convert('F')
                image_ch1 = transform_ch12(image_ch1)

                allimages[k][cpt][0] = image_ch1.numpy()
                # remplissage du tensor qui contient toutes les images avec le canal 1,
                # qui se trouve à l'indice 0 pour chaque image

                if tif_ch1img_name[-6] == 'T':  # cas des images d'avant 220404
                    pos_z = tif_ch1img_name.find('Z')  # indice ou se trouve le Z
                    pos_t = tif_ch1img_name.find('T')  # indice ou se trouve le T
                    num_z = tif_ch1img_name[pos_z+1:pos_t-1]  # profondeur z
                    tif_ch2img_name = lsmczi_imgname + " - Ch2 - C2 Z" + str(num_z) + " T1.tif"
        
                else:  # cas des images de 220404
                    tif_ch2img_name = tif_ch1img_name[:-10] + "2" + tif_ch1img_name[-9:]

                img_path_ch2 = os.path.join(img_dir, lsmczi_imgname[6:9], lsmczi_imgname[:5], lsmczi_imgname, 'ch2', tif_ch2img_name)
                image_ch2 = Image.open(img_path_ch2).convert('F')
                image_ch2 = transform_ch12(image_ch2)

                allimages[k][cpt][1] = image_ch2.numpy()
                # remplissage du tensor qui contient toutes les images avec le canal 2 (indice 1)

                if lsmczi_imgname[:5] == 'nalox':
                    label_bool = False
                else:
                    label_bool = True
                alllabels[k][cpt] = label_bool
                # remplissage du tensor qui contient tous les labels
                # On ne le fait qu'une fois (ici, avec ch1) car ch1 et ch2 ont le meme label

                cpt += 1

    return allimages, alllabels


all_images, all_labels = build_all_img_labels(kfolds, allfolds, used_img_dir, len_dataset, deterministic_transform_ch12)
"""

"""
# Verification qu'on a bien enregistre les bons canaux au bon endroit
num_fold = 4
num_img = 32
_, axis = plt.subplots(2, 1)
axis[0].imshow(all_images[num_fold][num_img][0])  # affichage canal 1
axis[1].imshow(all_images[num_fold][num_img][1])  # affichage canal 2
plt.show()
"""

"""
# -- Ecriture dans le fichier h5 --

file_h5 = h5py.File(path_h5, 'w', swmr=True)  # swmr = Single Write Multiple Reads

for i in range(kfolds):
    file_h5.create_dataset(str(i) + 'images', shape=all_images[i].shape, dtype='f', chunks=(10, 2, 400, 400),
                           data=all_images[i])
    file_h5.create_dataset(str(i) + 'labels', shape=all_labels[i].shape, dtype=bool, data=all_labels[i])

file_h5.close()

stop_file = time.time()
runtime_file = stop_file - start_file
f.write('HDF5 files created. Runtime: {val:0.4f} sec. \n'.format(val=runtime_file))
f.write('HDF5 files created. Runtime: {val:0.4f} min \n'.format(val=runtime_file / 60))
f.write('HDF5 files created. Runtime: {val:0.4f} h \n'.format(val=runtime_file / 3600))

"""

#"""
# ---- Chargement des donnees ---- #


class ECMDataset(Dataset):
    # ECM dataset.
    # Loads images and labels.

    def __init__(self, h5path, fold_number, transform_color=None, transform=None):
        # Args:
        #    root_dir (string): Directory with all the images.
        #    set_name (string): Number of the fold.
        #    transform (callable, optional): Optional transform to be applied on a sample.

        self.h5path = h5path
        self.fold_number = fold_number
        self.transform_color = transform_color
        self.transform = transform

    def __len__(self):
        return len(h5py.File(self.h5path, 'r')[str(self.fold_number) + 'labels'])

    def __getitem__(self, idx):
        file = h5py.File(self.h5path, 'r')

        image = file[str(self.fold_number) + 'images'][idx]
        label = file[str(self.fold_number) + 'labels'][idx].astype(int)

        if self.transform_color is None:
            self.transform_color = transforms.Lambda(lambda img: img)
        if self.transform is None:
            self.transform = transforms.Lambda(lambda img: img)

        # image = self.transform(torch.Tensor(image).view(2, 400, 400))  # ancienne version

        image_ch12 = torch.Tensor(image).view(2, 400, 400)
        
        # Application transformations aleatoires de flip et rognage 
        img_cropped = self.transform(image_ch12)
        #'''
        # Application transformations de luminosite et contraste
        # Creation d'une image a trois canaux pour pouvoir lui appliquer ColorJitter
        # (on copie le canal ch1 dans chacun des trois canaux, et pareil pour ch2)
        image_ch1_3channels = torch.Tensor(3, 250, 250)
        image_ch1_3channels[0] = img_cropped[0]
        image_ch1_3channels[1] = img_cropped[0]
        image_ch1_3channels[2] = img_cropped[0]

        image_ch2_3channels = torch.Tensor(3, 250, 250)
        image_ch2_3channels[0] = img_cropped[1]
        image_ch2_3channels[1] = img_cropped[1]
        image_ch2_3channels[2] = img_cropped[1]

        img_cropped[0] = F.rgb_to_grayscale(self.transform_color(image_ch1_3channels))
        img_cropped[1] = F.rgb_to_grayscale(self.transform_color(image_ch2_3channels))
        #'''
        return img_cropped, label


# ---- Application des transformations ---- #

transform_alea_color = transforms.ColorJitter(brightness=[0, 2], contrast=[0, 2])
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.RandomCrop(250)])
transform_valid = transforms.Resize(250)

#"""
"""
dataset0 = ECMDataset(h5path=path_h5, fold_number=0, transform_color=transform_alea_color, transform=transform_train)
len_dataset = len(dataset0)
print("Taille fold i :", len_dataset)
# f.write('Taille ensemble de toutes les donnees : {val} \n'.format(val=len_dataset))


# Calcul de la longueur d'un cote des images :
i=100
image0, label0 = dataset0[i]
image1, label1 = dataset0[i+2]
image2, label2 = dataset0[i+4]
image_size = image0.shape[1]
# f.write("Shape of an image: {val} \n".format(val=image0.shape))
print("Shape of an image:", image0.shape)
# print(image0, label0)

# Affichage de trois images
fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)
canal1 = ax1.imshow(image0[0], cmap="gray")  # affichage canal 1
plt.colorbar(canal1)
ax2 = fig.add_subplot(2, 3, 4)
canal2 = ax2.imshow(image0[1], cmap="gray")  # affichage canal 2
plt.colorbar(canal2)
ax3 = fig.add_subplot(2, 3, 2)
canal3 = ax3.imshow(image1[0], cmap="gray")
plt.colorbar(canal3)
ax4 = fig.add_subplot(2, 3, 5)
canal4 = ax4.imshow(image1[1], cmap="gray")
plt.colorbar(canal4)
ax5 = fig.add_subplot(2, 3, 3)
canal5 = ax5.imshow(image2[0], cmap="gray")
plt.colorbar(canal5)
ax6 = fig.add_subplot(2, 3, 6)
canal6 = ax6.imshow(image2[1], cmap="gray")
plt.colorbar(canal6)
plt.show()
"""
#"""
# image_size = 250
# print("Shape of an image:", image_size)


# ---- Entrainement ---- #

# Fonction d'entrainement pour une epoque (donne loss et accuracy)
def train_epoch(neuralnet, gpudevice, epoch, train_loader, losscriterion, name_optimizer):
    epoch_without_data = 0
    epoch_start = time.time()

    sum_train_loss = 0.0
    train_real_labels = np.array([])  # permet calcul accuracy
    train_predicted_labels = np.array([])  # permet calcul accuracy

    neuralnet.train()

    for data in train_loader:  # passage successif dans les batchs
        inputs, labels = data[0].to(gpudevice), data[1].to(gpudevice)

        train_without_data_start = time.time()
        name_optimizer.zero_grad()

        outputs = neuralnet(inputs)

        train_real_labels = np.append(train_real_labels, labels.cpu().numpy())  # liste de tous les labels du batch

        # Calcul des labels predits par le reseau
        if 'BCE' in str(losscriterion):  # pour BCEWithLogitsLoss
            outputs_sig = torch.sigmoid(outputs)
            for x in outputs_sig:
                train_predicted_labels = np.append(train_predicted_labels, int(x + 0.5))
                # donne 0 si x<0.5 et 1 si x >= 0.5

            labels = labels.float()
            labels = labels.unsqueeze(1)  # necessaire quand output de taille 1

        else:  # classification multi-classes (ou binaire avec CrossEntropyLoss)
            _, predicted = torch.max(outputs.data, 1)
            train_predicted_labels = np.append(train_predicted_labels, predicted.cpu().numpy())

        train_batch_loss = losscriterion(outputs, labels)  # la loss des images du batch en cours

        train_batch_loss.backward()
        name_optimizer.step()
        sum_train_loss += train_batch_loss.item()  # pour le calcul de la loss totale

        train_without_data_end = time.time()
        epoch_without_data += train_without_data_end - train_without_data_start

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    if epoch < 10:
        f.write('train 1 epoch without loading data in s.: {val:0.4f} \n'.format(val=epoch_without_data))
        f.write('train 1 epoch total time in s.: {val:0.4f} \n'.format(val=epoch_time))

    train_loss = sum_train_loss / len(train_loader)
    # on divise par le nombre de batchs pour pouvoir comparer les loss entre differents ensembles
    f.write('Epoch {ep:02d}: train loss {tl:0.4f}\n'.format(ep=epoch, tl=train_loss))
    print('Epoch {ep:02d}: train loss {tl:0.4f}'.format(ep=epoch, tl=train_loss))

    train_acc = accuracy_score(train_predicted_labels, train_real_labels)
    f.write('Epoch {ep:02d}: train accuracy {tl:0.4f}\n'.format(ep=epoch, tl=train_acc))
    print('Epoch {ep:02d}: train accuracy {tl:0.4f}'.format(ep=epoch, tl=train_acc))

    return train_loss, train_acc


# Fonction de validation pour une epoque (donne loss et accuracy)
def validation_epoch(neuralnet, gpudevice, epoch, valid_loader, losscriterion):
    sum_valid_loss = 0.0
    valid_real_labels = np.array([])  # permet calcul accuracy
    valid_predicted_labels = np.array([])  # permet calcul accuracy

    neuralnet.eval()

    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data[0].to(gpudevice), data[1].to(gpudevice)

            outputs = neuralnet(inputs)

            valid_real_labels = np.append(valid_real_labels, labels.cpu().numpy())  # liste de tous les labels du batch

            # Calcul des labels predits par le reseau
            if 'BCE' in str(losscriterion):  # valable pour criterion=BCEWithLogitsLoss()
                outputs_sig = torch.sigmoid(outputs)
                for x in outputs_sig:
                    valid_predicted_labels = np.append(valid_predicted_labels, int(x + 0.5))
                    # donne 0 si x<0.5 et 1 si x >= 0.5

                labels = labels.float()
                labels = labels.unsqueeze(1)  # necessaire quand output de taille 1

            else:  # classification multi-classes
                _, predicted = torch.max(outputs.data, 1)
                valid_predicted_labels = np.append(valid_predicted_labels, predicted.cpu().numpy())

            valid_batch_loss = losscriterion(outputs, labels)
            sum_valid_loss += valid_batch_loss.item()

    valid_loss = sum_valid_loss / len(valid_loader)
    f.write('Epoch {ep:02d}: validation loss {vl:0.4f} \n'.format(ep=epoch, vl=valid_loss))
    print('Epoch {ep:02d}: validation loss {vl:0.4f}'.format(ep=epoch, vl=valid_loss))

    valid_acc = accuracy_score(valid_predicted_labels, valid_real_labels)
    f.write('Epoch {ep:02d}: validation accuracy {vl:0.4f} \n'.format(ep=epoch, vl=valid_acc))
    print('Epoch {ep:02d}: validation accuracy {vl:0.4f}'.format(ep=epoch, vl=valid_acc))

    return valid_loss, valid_acc


# ---- Entrainement sur les folds ---- #

# pour charger checkpoints enregistres pendant entrainement
def load_checkpt(checkpt_path, neuralnet, name_optimizer):
    '''
    :param checkpt_path: path where checkpoint is stored
    :param neuralnet: model (neural network) that we want to load checkpoint parameters into
    :param name_optimizer: optimizer we defined in previous training
    '''
    # load check point
    if torch.cuda.is_available():
        checkpt = torch.load(checkpt_path)
    else:
        checkpt = torch.load(checkpt_path, map_location=torch.device('cpu'))

    # initialise epoch number
    epoch = checkpt['epoch']
    # initialise state_dict from checkpoint to model
    neuralnet.load_state_dict(checkpt['model_state_dict'])
    # initialise optimizer from checkpoint to optimizer
    name_optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    # initialise saved loss and accuracy
    train_loss = checkpt['train_loss']
    valid_loss = checkpt['valid_loss']
    train_acc = checkpt['train_acc']
    valid_acc = checkpt['valid_acc']
    # initialise RNG states (Random Number Generator)
    cpu_rng = checkpt['cpu_rng']
    if torch.cuda.is_available():
        gpu_rng = checkpt['gpu_rng']
    else:
        print("Cuda indisponible, chargement etat GPU RNG impossible.")
        f.write("Cuda indisponible, chargement etat GPU RNG impossible. \n")
        gpu_rng = None
    numpy_rng = checkpt['numpy_rng']
    return epoch, neuralnet, name_optimizer, train_loss, valid_loss, train_acc, valid_acc, cpu_rng, gpu_rng, numpy_rng


def train(valid_fold, folds_number, pathh5, used_nn, transformaleacolor, transformvalid, transformtrain, bs, wdecay,
          start_ep, last_ep, losscriterion, prefixname, prefixname_toload):
    '''
    :param valid_fold : numero de fold pour ensemble de validation
    :param folds_number : nombre de folds
    :param pathh5 : chemin du fichier h5 ou sont enregistres images et labels
    :param used_nn : nom du reseau de neurones utilise
    :param transformaleacolor : fonction de transformation sur les couleurs (luminosite et contraste)
    :param transformvalid : fonction de transformation pour l'ensemble de validation
    :param transformtrain : fonction de transformation pour l'ensemble d'entrainement
    :param bs : taille du batch
    :param wdecay : weight decay pour la regularisation L2
    :param start_ep : numero d'epoque a laquelle on commence l'entrainement (les epoques sont numerotees a partir de 0)
    :param last_ep : numero derniere epoque
    :param losscriterion : fonction de perte
    :param prefixname : prefixe_name
    :param prefixname_toload : nom du dossier dans lequel aller recuperer les modeles a charger (dans les cas ou on
    continue un entrainement prealablement commence)
    :return : dictionnaire avec pour chaque fold l'historique des loss et accuracy pour train et valid
    '''

    gpudevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f.write('Used device (CPU or GPU with cuda) for validation fold {valf}: {val} \n'
            .format(valf=valid_fold, val=gpudevice))

    foldresults = {}  # dictionnaire pour stocker les valeurs de loss et acc pour chaque fold
    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
    # stockage des loss et acc pour le fold de validation en cours (valid_fold)

    print("Numero fold de validation :", valid_fold)
    f.write("Numero fold de validation : {val} \n".format(val=valid_fold))
    train_idx = [i for i in range(folds_number) if i != valid_fold]
    # train_idx : liste des numeros de folds pour ensemble d'entrainement

    # Creation des ensembles de validation et d'entrainement
    validset = ECMDataset(h5path=pathh5, fold_number=valid_fold, transform_color=transformaleacolor,
                          transform=transformvalid)
    trainset = ConcatDataset([ECMDataset(h5path=pathh5, fold_number=train_fold, transform_color=transformaleacolor,
                                         transform=transformtrain) for train_fold in train_idx])
    print("Len validset:", len(validset))
    print("Len trainset:", len(trainset))
    f.write("Len validset: {val} \n".format(val=len(validset)))
    f.write("Len trainset: {val} \n".format(val=len(trainset)))

    # Creation des loaders correspondants
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=bs,
                                              num_workers=5,  # 8 possible sur calmip, 5 sur osirim
                                              pin_memory=True,
                                              shuffle=True)

    validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=bs,
                                              num_workers=5,
                                              pin_memory=True,
                                              shuffle=False)

    # -- Initialisations (chargement du checkpoint le cas echeant) --

    output_size = 2  # nombre de neurones en sortie

    if start_ep == 0:  # entrainement depuis zero
        valid_loss_min = np.Inf  # initialisation valeur minimale ensemble validation
        best_epoch = -1  # epoque correspondante

        # Definition du reseau de neurones utilise
        if used_nn == 'ResNet':
            neuralnet = models.resnet18(pretrained=False)
            neuralnet.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # couche convolutionnelle d'entree, avec 2 canaux (et pas 3 comme c'est par defaut)
            neuralnet.fc = torch.nn.Linear(512, output_size)
            # couche de sortie totalement connectee, avec une seule sortie (et pas 1000 comme c'est par defaut)
            print("Using ResNet with output of size %s." % output_size)
            f.write("Using ResNet with output of size {val}. \n".format(val=output_size))
        elif used_nn == 'SqueezeNet':
            neuralnet = models.squeezenet1_1(pretrained=False)
            neuralnet.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(2, 2))
            # couche convolutionnelle d'entree, avec 2 canaux (et pas 3 comme c'est par defaut)
            neuralnet.classifier[1] = torch.nn.Conv2d(512, output_size, kernel_size=(1, 1), stride=(1, 1))
            # couche de sortie avec 22 sorties (et pas 1000 comme c'est par defaut)
            print("Using SqueezeNet with output of size %s." % output_size)
            f.write("Using SqueezeNet with output of size {val}. \n".format(val=output_size))
        elif used_nn == 'vgg11':
            neuralnet = models.vgg11(pretrained=False)
            neuralnet.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            neuralnet.classifier[6] = torch.nn.Linear(in_features=4096, out_features=output_size, bias=True)
            print("Using vgg with output of size %s." % output_size)
            f.write("Using vgg with output of size {val}. \n".format(val=output_size))

        neuralnet.to(gpudevice)

        name_optimizer = torch.optim.SGD(neuralnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=wdecay)
        # name_optimizer = torch.optim.Adam(neuralnet.parameters(), lr=learning_rate)
        print("Using optimizer", name_optimizer)
        f.write("Using optimizer {val} \n".format(val=name_optimizer))
        # attention, on reinitialise bien pour chaques nvx ensembles le reseau de neurones et l'optmizer (qui depend
        # du reseau de neurones)
    else:  # entrainement depuis situation prealablement enregistree
        if used_nn == 'ResNet':
            neuralnet = models.resnet18(pretrained=False)
            neuralnet.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            neuralnet.fc = torch.nn.Linear(512, output_size)
        elif used_nn == 'SqueezeNet':
            neuralnet = models.squeezenet1_1(pretrained=False)
            neuralnet.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(2, 2))
            neuralnet.classifier[1] = torch.nn.Conv2d(512, output_size, kernel_size=(1, 1), stride=(1, 1))
        elif used_nn == 'vgg11':
            neuralnet = models.vgg11(pretrained=False)
            neuralnet.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            neuralnet.classifier[6] = torch.nn.Linear(in_features=4096, out_features=output_size, bias=True)

        name_optimizer = torch.optim.SGD(neuralnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=wdecay)
        # name_optimizer = torch.optim.Adam(neuralnet.parameters(), lr=learning_rate)
        print("Using optimizer", name_optimizer)
        f.write("Using optimizer {val} \n".format(val=name_optimizer))
        _, neuralnet, name_optimizer, _, _, _, _, cpurng, gpurng, numpyrng \
            = load_checkpt('./tar_files/%s/checkpoints/fold%s_ep%s.tar'
                           % (prefixname_toload, valid_fold, start_ep - 1), neuralnet, name_optimizer)
        torch.set_rng_state(cpurng)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpurng)
        np.random.set_state(numpyrng)
        f.write("Load checkpoint from ./tar_files/{pref}/checkpoints/fold{fd}_ep{ep}.tar \n"
                .format(pref=prefixname_toload, fd=valid_fold, ep=start_ep - 1))
        print("Load checkpoint from",
              './tar_files/%s/checkpoints/fold%s_ep%s.tar' % (prefixname_toload, valid_fold, start_ep - 1))
        neuralnet.to(gpudevice)

        for bestmodel in list(os.listdir('./tar_files/%s/best_models' % prefixname_toload)):
            # recuperation meilleurs epoque et validation loss du fold en cours
            if bestmodel[4] == str(valid_fold):
                best_epoch, _, _, _, valid_loss_min, _, _, _, _, _ \
                    = load_checkpt(os.path.join('./tar_files/%s/best_models' % prefixname_toload, bestmodel),
                                   neuralnet, name_optimizer)
                f.write("Load previous best model from ./tar_files/{pref}/best_models/{md} \n"
                        .format(pref=prefixname_toload, md=bestmodel))
                f.write("Meilleur modele prealablement enregistre : {md}, a l'epoque {ep} avec loss de validation de "
                        "{ls} \n".format(md=bestmodel, ep=best_epoch, ls=valid_loss_min))
                print("Load previous best model from",
                      os.path.join('./tar_files/%s/best_models' % prefixname_toload, bestmodel))
                print("Meilleur modele prealablement enregistre : %s, a l'epoque %s avec loss de validation de %s"
                      % (bestmodel, best_epoch, valid_loss_min))

    for epoch in range(start_ep, last_ep + 1):
        torch.manual_seed(epoch)  # pour reinitialiser seed a chaque fois

        train_loss, train_acc = train_epoch(neuralnet, gpudevice, epoch, trainloader, losscriterion, name_optimizer)
        valid_loss, valid_acc = validation_epoch(neuralnet, gpudevice, epoch, validloader, losscriterion)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        # Enregistrement checkpoint (a la derniere epoque et a la meilleure performance)
        if torch.cuda.is_available():
            checkpt = {'epoch': epoch,
                       'model_state_dict': neuralnet.state_dict(),
                       'optimizer_state_dict': name_optimizer.state_dict(),
                       'train_loss': train_loss,
                       'valid_loss': valid_loss,
                       'train_acc': train_acc,
                       'valid_acc': valid_acc,
                       'cpu_rng': torch.get_rng_state(),
                       'gpu_rng': torch.cuda.get_rng_state(),
                       'numpy_rng': np.random.get_state()}
        else:
            print("Cuda indisponible, enregistrement etat GPU RNG impossible.")
            f.write("Cuda indisponible, enregistrement etat GPU RNG impossible. \n")
            checkpt = {'epoch': epoch,
                       'model_state_dict': neuralnet.state_dict(),
                       'optimizer_state_dict': name_optimizer.state_dict(),
                       'train_loss': train_loss,
                       'valid_loss': valid_loss,
                       'train_acc': train_acc,
                       'valid_acc': valid_acc,
                       'cpu_rng': torch.get_rng_state(),
                       'numpy_rng': np.random.get_state()}

        if epoch == last_ep:  # derniere epoque
            torch.save(checkpt, './tar_files/%s/checkpoints/fold%s_ep%s.tar' % (prefixname, valid_fold, epoch))

        if valid_loss <= valid_loss_min:  # cas de la meilleure performance
            if best_epoch >= start_ep:
                # on ne peut supprimer un precedent meilleur resultat que s'il a effectivement ete enregistre dans
                # le dossier courant (le precedent meilleur resultat ne doit donc pas etre np.Inf ni se trouver dans
                # les dossiers du precedent entrainement)
                os.remove('./tar_files/%s/best_models/fold%s_ep%s.tar' % (prefixname, valid_fold, best_epoch))

            torch.save(checkpt, './tar_files/%s/best_models/fold%s_ep%s.tar' % (prefixname, valid_fold, epoch))
            valid_loss_min = valid_loss
            best_epoch = epoch

    foldresults['fold{}'.format(valid_fold)] = history

    return foldresults


# -- Entrainement --

# Valeurs de la loss
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
# transform_alea_color = None

print('Loss function used:', criterion)
f.write("Loss function used: {val} \n".format(val=criterion))

start = time.time()

fold_results = train(valid_fold_nb, kfolds, path_h5, neuralnet_used, transform_alea_color, transform_valid,
                     transform_train, batch_size, w_decay, start_epoch, last_epoch, criterion, prefix_name,
                     prefix_name_toload)

end = time.time()

# print(fold_results)
f.write("Fold results: {val} \n".format(val=fold_results))

f.write('Training time in s.: {val:0.3f} \n'.format(val=end - start))
f.write('Training time in min: {val:0.3f} \n'.format(val=(end - start) / 60))
f.write('Training time in hours: {val:0.3f} \n'.format(val=(end - start) / 3600))

# Enregistrement valeurs loss et accuracy

g = open('./json_files/%s.json' % prefix_name, "w")
foldresults_json = json.dumps(fold_results)  # transformation de dictionnaire a format json
g.write(foldresults_json)  # ecriture dans le fichier .json
g.close()
#"""

#"""
# ---- Traces des fonctions objectif et d'exactitude ----


# Fonctions objectifs
def plot_loss(resultats, valid_fold, used_nn, start_ep, last_ep, losscriterion, lr, bs, wdecay, datasetname, prefixname,
              sfx):
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss function \n (%s)' % losscriterion)

    list_linestyles = ['-', '--', '-.', ':', (0, (1, 10))]

    ax.plot([i for i in range(start_ep, last_ep + 1)], resultats['fold' + str(valid_fold)]['valid_loss'], color='b',
            linestyle=list_linestyles[valid_fold])
    ax.plot([i for i in range(start_ep, last_ep + 1)], resultats['fold' + str(valid_fold)]['train_loss'], color='r',
            linestyle=list_linestyles[valid_fold], label='Fold ' + str(valid_fold))

    ax.legend()
    fig.text(0.5, 0.99, 'Binary Classification', ha='center', va='top', size='large')
    fig.text(0.02, 0.95, '%s, %s, %s \nlearning rate=%s, batch size=%s, w_decay=%s'
             % (echeance_name, used_nn, losscriterion, lr, bs, wdecay), ha='left', va='top')
    fig.text(0.01, 0.04, datasetname, ha='left', size="small")
    fig.text(0.01, 0.01, sfx, ha='left', size="small")
    fig.text(0.9, 0.92, "training loss", ha="right", va="bottom", size="medium", color='r')
    fig.text(0.9, 0.885, "validation loss", ha="right", va="bottom", size="medium", color='b')
    # ha : 'center', 'right', 'left'
    # va : 'top', 'bottom', 'center', 'baseline', 'center_baseline'

    fig.savefig('./figures/%s_loss.png' % prefixname)


plot_loss(fold_results, valid_fold_nb, neuralnet_used, start_epoch, last_epoch, criterion, learning_rate, batch_size,
          w_decay, dataset_name, prefix_name, suffix)


# Trace de l'accuracy
def plot_accuracy(resultats, valid_fold, used_nn, start_ep, last_ep, losscriterion, lr, bs, wdecay, datasetname,
                  prefixname, sfx):
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Accuracy')

    list_linestyles = ['-', '--', '-.', ':', (0, (1, 10))]

    ax.plot([i for i in range(start_ep, last_ep + 1)], resultats['fold' + str(valid_fold)]['valid_acc'], color='b',
            linestyle=list_linestyles[valid_fold])
    ax.plot([i for i in range(start_ep, last_ep + 1)], resultats['fold' + str(valid_fold)]['train_acc'], color='r',
            linestyle=list_linestyles[valid_fold], label='Fold ' + str(valid_fold))

    ax.legend()
    fig.text(0.5, 0.99, 'Binary Classification', ha='center', va='top', size='large')
    fig.text(0.02, 0.95, '%s, %s, %s \nlearning rate=%s, batch size=%s, w_decay=%s'
             % (echeance_name, used_nn, losscriterion, lr, bs, wdecay), ha='left', va='top')
    fig.text(0.01, 0.04, datasetname, ha='left', size="small")
    fig.text(0.01, 0.01, sfx, ha='left', size="small")
    fig.text(0.9, 0.92, "training accuracy", ha="right", va="bottom", size="medium", color='r')
    fig.text(0.9, 0.885, "validation accuracy", ha="right", va="bottom", size="medium", color='b')

    fig.savefig('./figures/%s_acc.png' % prefixname)


plot_accuracy(fold_results, valid_fold_nb, neuralnet_used, start_epoch, last_epoch, criterion, learning_rate, batch_size,
              w_decay, dataset_name, prefix_name, suffix)

big_end = time.time()
f.write('Total run time in s: {val:0.3f} \n'.format(val=(big_end - big_start)))
f.write('Total run time in min: {val:0.3f} \n'.format(val=(big_end - big_start) / 60))
f.write('Total run time in hours: {val:0.3f} \n'.format(val=(big_end - big_start) / 3600))
#"""
f.close()
