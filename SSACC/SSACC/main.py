import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import os
import sys
sys.path.append('../global_module/')
import network
import train
from generate_pic import aa_and_each_accuracy, sampling, sampling_att, load_dataset, load_dataset1, load_dataset2, generate_png, generate_iter
from Utils import fdssc_model, record, extract_samll_cubic

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seeds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 10]
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

Dataset = {'IP'}
data_hsi, gt_hsi, TOTAL_SIZE, TEST_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)
# Dataset = {'PU'}
# data_hsi, gt_hsi, TOTAL_SIZE, TEST_SIZE,VALIDATION_SPLIT = load_dataset1(Dataset)
# Dataset = {'sv'}
# data_hsi, gt_hsi, TOTAL_SIZE, TEST_SIZE,VALIDATION_SPLIT = load_dataset2(Dataset)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)

CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
epoch = 10

lr, num_epochs, batch_size = 0.00050, 200, 16

loss1 = torch.nn.CrossEntropyLoss()
loss2 = torch.nn.MSELoss()

img_channels = data_hsi.shape[2] #200
INPUT_DIMENSION = data_hsi.shape[2] #200
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]#145*145 21025
TRAINING_TIME = []
TESTING_TIME = []

KAPPA1 = []
OA1 = []
AA1 = []
ELEMENT_ACC3 = np.zeros((epoch, CLASSES_NUM))

data = preprocessing.scale(data)
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
for index_iter in range(epoch):
    print('epoch:', index_iter)
    net1 = network.DENSE(BAND, CLASSES_NUM)
    net2 = network.CAM(60, CLASSES_NUM)
    a = 0.1
    optimizer1 = optim.Adam(net1.parameters(), lr=lr, amsgrad=False)  # , weight_decay=0.0001)
    optimizer2 = optim.Adam(net2.parameters(), lr=lr, amsgrad=False)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)  # (0.97,(21025,))
    _, total_indices = sampling(1, gt)  # (1,)
    _1, total_att = sampling_att(1, gt)
    TRAIN_SIZE = len(train_indices)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    VAL_SIZE = int(TRAIN_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    p1 = 3
    p2 = 5
    train_iter, valida_iter, test_iter, all_iter, att_iter = generate_iter(p1, p2, total_att, TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE, whole_data, INPUT_DIMENSION, batch_size, gt)

    tic1 = time.clock()

    train.train(a, net1, net2, train_iter, valida_iter, loss1, loss2, optimizer1, optimizer2, device, epochs=num_epochs)
    toc1 = time.clock()
    pred_test_fdssc = []
    pred_test_fdssc1 = []
    pred_test_fdssc2 = []
    tic2 = time.clock()

    with torch.no_grad():

        for X1, y1, X2, y2 in test_iter:
            X1 = X1.to(device)
            X2 = X2.to(device)
            net1.eval()
            net2.eval()

            y1, y2 = net1(X1, X2)
            y_hat1, x_att1 = net2(y1)
            y_hat2, x_att2 = net2(y2)
            record.record_attention(x_att1, x_att2, '/home/nlabtest08/DBDA/result/records/' + 'IN_attention_' + 'split:' + str(VALIDATION_SPLIT) + 'lr:' + str(lr) + '.txt')
            y_hat = np.sum((y_hat1, y_hat2), 0)
            y_hat = y_hat/2

            pred_test_fdssc.extend(np.array(y_hat.cpu().argmax(axis=1)))

    toc2 = time.clock()

    collections.Counter(pred_test_fdssc)
    gt_test = gt[test_indices] - 1
    overall_acc_fdssc3 = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])

    confusion_matrix_fdssc3 = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
    each_acc_fdssc3, average_acc_fdssc3 = aa_and_each_accuracy(confusion_matrix_fdssc3)
    kappa3 = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    torch.save(net2.state_dict(), "/home/nlabtest08/DBDA/net/" + str(round(overall_acc_fdssc3, 3)) + '.pt')
    KAPPA1.append(kappa3)
    OA1.append(overall_acc_fdssc3)
    AA1.append(average_acc_fdssc3)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC3[index_iter, :] = each_acc_fdssc3


print("--------" + net2.name + " Training Finished-----------")
record.record_output(OA1, AA1, KAPPA1, ELEMENT_ACC3, TRAINING_TIME, TESTING_TIME,
                     '/home/nlabtest08/DBDA/result/records/' + net2.name + day_str + '_' + 'split:' + str(
                         VALIDATION_SPLIT) + 'lr:' + str(lr) + '.txt')

generate_png(all_iter, net1, net2, gt_hsi, Dataset, device, total_indices)


