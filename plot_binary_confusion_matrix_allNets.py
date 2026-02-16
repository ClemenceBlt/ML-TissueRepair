import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, ConcatDataset
from torch import nn
import h5py
import os
import seaborn as sn
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# trace de la matrice de confusion a partir du fichier tar contenant le meilleur model d'un entrainement.
# Se base sur le fichier h5 de l'ensemble de test.

# parametres
start_epoch = 0
last_epoch = 1499
batch_size = 16
learning_rate = 1e-4
w_decay = 1e-5
test_fold_train = 1
echeance = "j03j10"
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
output_size = 1
dataset_name = '230323_trainmosTV'
suffix = 'divmos'
neuralnet_used = 'resnet_CV'
#neuralnet_used = 'SqueezeNet'
#neuralnet_used = 'vgg11'

testset_path = './h5_files/dataset230323_trainmosTe_5folds_j03j10_ch12_divmos.h5'

if testset_path == './h5_files/dataset230323_trainmosTV_5folds_j03j10_ch12_divmos.h5':
    setTVTe = 'TV'
elif testset_path == './h5_files/dataset230323_trainmosTe_5folds_j03j10_ch12_divmos.h5':
    setTVTe = 'Te'
else:
    setTVTe = 'undefSet'

testset_name = testset_path[18:35]

if suffix=='_':
    prefix_name = 'train_ch12_%s_%s_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s' \
                % (echeance, neuralnet_used, dataset_name, start_epoch, last_epoch, batch_size, learning_rate,
                    w_decay, test_fold_train)
else:
    prefix_name = 'train_ch12_%s_%s_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s_%s' \
                % (echeance, neuralnet_used, dataset_name, start_epoch, last_epoch, batch_size, learning_rate,
                    w_decay, test_fold_train, suffix)   

torch.manual_seed(0)


class ECMDataset(Dataset):
    # ECM dataset.
    # Loads images and labels.

    def __init__(self, h5path, fold_number, transform=None):
        # Args:
        #    root_dir (string): Directory with all the images.
        #    fold_number (string): Number of the fold.
        #    transform (callable, optional): Optional transform to be applied on a sample.

        self.h5path = h5path
        self.fold_number = fold_number
        self.transform = transform

    def __len__(self):
        return len(h5py.File(self.h5path, 'r')[str(self.fold_number) + 'labels'])
        # return len(h5py.File(self.h5path, 'r')[str(self.fold_number) + 'labels_y'])

    def __getitem__(self, idx):

        file = h5py.File(self.h5path, 'r')

        image = file[str(self.fold_number) + 'images'][idx]
        label = file[str(self.fold_number) + 'labels'][idx].astype(int)
        # label = file[str(self.fold_number) + 'labels_y'][idx].astype(int)

        if self.transform is None:
            self.transform = transforms.Lambda(lambda img: img)
        image = self.transform(torch.Tensor(image).view(2, 400, 400))

        return image, label


# -- Labels predits --
def get_predictions_mat(neuralnet, loss_criterion, data_loader, prefixname, dataset, set_TVTe):
    real_labels = np.array([])
    predicted_labels = np.array([])
    with torch.no_grad():
        cpt = 0
        print('Compteur jusque', len(data_loader))
        for data in data_loader:
            cpt += 1
            print(cpt)
            inputs, labels = data[0], data[1]

            outputs = neuralnet(inputs)

            real_labels = np.append(real_labels, labels.cpu().numpy())

            if 'BCE' in str(loss_criterion):  # classification binaire
                outputs_sig = torch.sigmoid(outputs)  # valable pour criterion=BCEWithLogitsLoss()
                for x in outputs_sig:
                    predicted_labels = np.append(predicted_labels, int(x + 0.5))
                    # donne 0 si x<0.5 et 1 si x >= 0.5
            else:  # classification multi-classes
                _, predicted = torch.max(outputs.data, 1)
                predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())"

    # Accuracy
    test_acc = accuracy_score(real_labels, predicted_labels)
    print("Accuracy on test set from best model:", test_acc)
    report = classification_report(real_labels, predicted_labels)
    print(report)

    ftscale = 20
    ftsz=250

    # Matrice de confusion
    array = confusion_matrix(real_labels, predicted_labels)
    print(type(array))
    #df_cm = pd.DataFrame(array, index=["naloxone", "small lipectomy"], columns=["naloxone", "small lipectomy"])
    df_cm = pd.DataFrame(array, index=["rgn", "scar."], columns=["rgn", "scar."])

    sn.set_theme(font_scale=ftscale)  # taille du texte des axes

    plt.rcParams["figure.figsize"] = [40, 31]  # 9, 5 avec texte
    fig, ax = plt.subplots()
    sn.heatmap(df_cm, cmap='Blues', annot=True, fmt='0.0f', annot_kws={"size": ftsz}, cbar=False, square=True)
    # annot_kws : taille des chiffres dans la matrice, cbar : legende des couleurs
    """
    fig.text(0.5, 0.995, 'Confusion matrix - Binary classification', ha='center', va='top', size='medium')
    fig.text(0.08, 0.95, '%s, %s, best model btwn ep %s and %s, \n%s, lr=%s, bs=%s, w_decay=%s, validfld %s'
             % (echeance, neuralnet_used, start_epoch, last_epoch, loss_criterion, learning_rate,
                batch_size, w_decay, test_fold_train), ha='left', va='top', size='small')
    fig.text(0.71, 0.92, dataset, ha='left', size="small")
    if suffix != '_':
        fig.text(0.71, 0.88, suffix, ha='left', size="small")
    fig.text(0.75, 0.5, report, wrap=True, ha='center', va='top', size='small')
    fig.text(0.75, 0.7, "Accuracy:\n%0.5s" % test_acc, wrap=True, ha='center', va='top', size='medium')  # version avec texte
    """
    fig.text(0.91, 0.705, "Acc:\n%s" % round(test_acc, 2), wrap=True, ha='center', va='top', size='medium')  # version sans texte

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # fig.tight_layout(pad=0.5, rect=[-0.33, -0.02, 0.9, 0.9])  # pour la version avec le texte
    fig.tight_layout(pad=0.01, rect=[-0.13, 0, 1.0, 1.0])  # pour la version sans le texte
    # pad : taille de la figure (plus pad petit, plus la figure est grande)
    # (rect[0], rect[1]) = (abscisse, ordonnee) du coin inferieur gauche que doit occuper la figure
    # (rect[2], rect[3]) = (abscisse, ordonnee) du coin superieur droit que doit occuper la figure

    plt.savefig('./figures_generees/%s_confmatReport_%s.png' % (prefixname, set_TVTe), dpi=300)
    # plt.plot()
    #plt.show()
    plt.close()


# Definition du reseau de neurones utilise
if neuralnet_used == 'ResNet' or neuralnet_used == 'resnet_CV':
    neural_net = models.resnet18(pretrained=False)
    neural_net.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    neural_net.fc = torch.nn.Linear(512, output_size)
elif neuralnet_used == 'SqueezeNet':
    neural_net = models.squeezenet1_1(pretrained=False)
    neural_net.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(2, 2))
    neural_net.classifier[1] = torch.nn.Conv2d(512, output_size, kernel_size=(1, 1), stride=(1, 1))
elif neuralnet_used == 'vgg11':
    neural_net = models.vgg11(pretrained=False)
    neural_net.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    neural_net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=output_size, bias=True)

for bestmodel in list(os.listdir('./tar_files/%s/best_models' % prefix_name)):
    checkpointpath = os.path.join('./tar_files/%s/best_models' % prefix_name, bestmodel)
    print("Fichier tar utilise :", checkpointpath)
    checkpoint = torch.load(checkpointpath, map_location=torch.device('cpu'))
    neural_net.load_state_dict(checkpoint['model_state_dict'])

transform_test = transforms.CenterCrop(250)
test_set = ConcatDataset([ECMDataset(h5path=testset_path, fold_number=foldn, transform=transform_test)
                         for foldn in range(5)])
# test_set = ECMDataset(h5path=testset_path, fold_number=test_fold_train, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          pin_memory=True,
                                          shuffle=False)

get_predictions_mat(neural_net, criterion, test_loader, prefix_name, testset_name, setTVTe)
