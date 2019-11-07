import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from numpy.random import normal

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=False)
    
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def gray_rgb(fixed_mnist, batch_size, image_size):
    fixed_mnist_rgb = torch.FloatTensor(batch_size, 3, image_size, image_size)
    fixed_mnist_rgb[:,0,:,:]=fixed_mnist
    fixed_mnist_rgb[:,1,:,:]=fixed_mnist
    fixed_mnist_rgb[:,2,:,:]=fixed_mnist
    return fixed_mnist_rgb

def test_mnistg(num_iter, data_iter, batch_size, domain, model, code, image_size):
    temp_model = model.train(False)
    correct_num = 0.0
    total_num = 0.0
    for _ in range(num_iter):
        mnist_val, mnist_val_labels = data_iter.next()
        mnist_valrgb = gray_rgb(mnist_val, batch_size, image_size)
        mnist_val, mnist_val_labels = to_var(mnist_valrgb), to_var(mnist_val_labels)
        d2_im, t2_feat, d2_class = temp_model(mnist_val, code)
        prob, predict = torch.max(d2_class, 1)
        correct_num += (predict == mnist_val_labels).sum().data[0]
        total_num += mnist_val.size(0)
        
    acc = float(correct_num)/float(total_num)
    print ('Testing %s %.4f' %(domain, acc))
    temp_model = model.train(True)
    return acc

def find_mnist(mnist, codet, svhn, codes, model1, thres = 0.95, thres2 = 0.95):
    gval = model1.train(False)
    fake_st, f_, etout = gval(svhn, codet[:svhn.size(0),...])
    fake_ss, f_, etout = gval(svhn, codes[:svhn.size(0),...])
    fake_ts, f_, etout = gval(mnist, codes[:mnist.size(0),...])
    fake_tt, f_, etout = gval(mnist, codet[:mnist.size(0),...])
    etout = F.softmax(etout)
    fake_sts, f_, eetout = gval(fake_st, codes[:svhn.size(0),...])
    fake_tst, f_, eetout = gval(fake_ts, codet[:mnist.size(0),...])
    eetout = F.softmax(eetout)
    eprobability, epredict = torch.max(etout, 1)
    eeprobability, eepredict = torch.max(eetout, 1)
    index_posi0 = np.where( to_data(eepredict.squeeze()) == to_data(epredict.squeeze()))
    ep_data = to_data(eprobability.squeeze())
    index_posi1 = np.where( ep_data > thres )
    index_posi = np.intersect1d(index_posi0,index_posi1)
    prelen = len(index_posi)
    prelenth = len(index_posi1[0])
    
    if len(index_posi)<mnist.size(0)*0.5:
        index_posi1 = np.where( ep_data > thres2 )
        index_posi = np.intersect1d(index_posi1,index_posi1)
    index_posi = torch.LongTensor(index_posi)
    if torch.cuda.is_available():
        index_posi = index_posi.cuda()
    gval=model1.train(True)
    return epredict.squeeze(), index_posi.squeeze(), prelen, prelenth, len(index_posi0[0]), len(index_posi1[0])#shape[0]

    
def gen_dcode(domain, batch, size):
    #domain=1 src; domain=0 tgt
    code=torch.rand(batch, 2, size[0], size[1])
    code[:,0,:,:]=domain
    code[:,1,:,:]=1-domain
    code=to_var(code)
    return code
