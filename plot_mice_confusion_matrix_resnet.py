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
last_epoch = 299
batch_size = 16
learning_rate = 1e-3
w_decay = 1e-5
valid_fold_train = 4
echeance = "j03j10"
losscriterion = "CrossEntropyLoss"
dataset_name = '230330_miceTV'
suffix = 'SGDColorAug'

testset_path = './h5_files/mice_230330_miceTe_5folds_j03j10_22mice.h5'
testset_name = testset_path[16:29]

if suffix == '_':
    prefix_name = 'mice_j03j10_resnet_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s' \
                  % (dataset_name, start_epoch, last_epoch, batch_size, learning_rate, w_decay, valid_fold_train)
else:
    prefix_name = 'mice_j03j10_resnet_%s_ep%s_%s_bs%s_lr%s_wd%s_validfld%s_%s' \
                  % (dataset_name, start_epoch, last_epoch, batch_size, learning_rate, w_decay, valid_fold_train, suffix)

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

    def __getitem__(self, idx):

        file = h5py.File(self.h5path, 'r')

        image = file[str(self.fold_number) + 'images'][idx]
        label = file[str(self.fold_number) + 'labels'][idx].astype(int)

        if self.transform is None:
            self.transform = transforms.Lambda(lambda img: img)
        image = self.transform(torch.Tensor(image).view(2, 400, 400))

        return image, label


# -- Labels predits --
def get_predictions_mat(neuralnet, loss_criterion, data_loader, prefixname, dataset):
    real_labels = np.array([])
    predicted_labels = np.array([])
    neuralnet.eval()  # pour desactiver la couche de dropout dans cette phase de test
    with torch.no_grad():
        cpt = 0
        print('Compteur jusque', len(data_loader))
        for data in data_loader:
            cpt += 1
            print(cpt)
            inputs, labels = data[0], data[1]

            outputs = neuralnet(inputs)

            real_labels = np.append(real_labels, labels.cpu().numpy())

            _, predicted = torch.max(outputs, dim=-1)
            predicted_labels = np.append(predicted_labels, predicted.numpy())

    # Accuracy
    test_acc = accuracy_score(predicted_labels, real_labels)
    print("Accuracy on test set from best model:", test_acc)
    report = classification_report(real_labels, predicted_labels)
    print(report)

    ftscale = 10
    ftsz=52

    # Matrice de confusion
    array = confusion_matrix(real_labels, predicted_labels)
    print(array.tolist())
    df_cm = pd.DataFrame(array)#, index=[i for i in range(38)], columns=[i for i in range(38)])
    sn.set_theme(font_scale=ftscale)  # taille du texte des axes

    plt.rcParams["figure.figsize"] = [34.5, 31]
    fig, ax = plt.subplots()
    sn.heatmap(df_cm, cmap='Blues', annot=True, fmt='0.0f', annot_kws={"size": ftsz}, cbar=False, square=True,
               xticklabels=True, yticklabels=True)
    # size=5 si 38 souris, size=7 si 22 souris
    """
    fig.text(0.5, 0.995, 'Confusion matrix - Mice classification', ha='center', va='top', size='small')

    fig.text(0.12, 0.965, '%s, ResNet18, best model btwn ep %s and %s,\n%s, lr=%s, bs=%s, w_decay=%s, validfld %s'
             % (echeance, start_epoch, last_epoch, loss_criterion, learning_rate, batch_size, w_decay, valid_fold_train),
             ha='left', va='top', size='x-small')
    fig.text(0.11, 0.04, dataset, ha='left', size="x-small")
    if suffix != '_':
        fig.text(0.11, 0.01, suffix, ha='left', size="x-small")
    """
    #fig.text(0.88, 0.55, "Accuracy:\n%0.5s" % test_acc, wrap=True, ha='center', va='top', size='medium')
    fig.text(0.948, 0.61, "Acc:\n%s" % round(test_acc, 2), wrap=True, ha='center', va='top', size='medium')

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    #fig.tight_layout(pad=1.05, rect=[0, -0.05, 1, 0.95])
    fig.tight_layout(pad=0.01, rect=[-0.064, 0, 1.0, 1.0])
    #plt.savefig('./figures_generees/%s_confusionmatrix_250301.png' % prefixname, dpi=300)
    #plt.show()
    plt.close()

# Definition du reseau de neurones utilise
neural_net = models.resnet18(pretrained=False)
neural_net.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
neural_net.fc = torch.nn.Linear(512, 22)

for bestmodel in list(os.listdir('./tar_files/%s/best_models' % prefix_name)):
    checkpointpath = os.path.join('./tar_files/%s/best_models' % prefix_name, bestmodel)
    checkpoint = torch.load(checkpointpath, map_location=torch.device('cpu'))
    neural_net.load_state_dict(checkpoint['model_state_dict'])

transform_test = transforms.CenterCrop(250)
test_set = ConcatDataset([ECMDataset(h5path=testset_path, fold_number=foldn, transform=transform_test)
                         for foldn in range(5)])
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          pin_memory=True,
                                          shuffle=False)

get_predictions_mat(neural_net, losscriterion, test_loader, prefix_name, testset_name)
