import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
from model import G
from model import D1,D2
import torch.nn.functional as F
from test import gray_rgb, test_mnistg, gen_dcode, find_mnist

from numpy.random import normal

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)
        
class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader, mnist_val_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.mnist_val_loader = mnist_val_loader
        self.g = None
        self.d1 = None
        self.d2 = None
        self.gc_optimizer = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.l2_feat_loss = config.l2_feat_loss
        self.l1_feat_loss = config.l1_feat_loss
        self.l2_reconst_loss = config.l2_reconst_loss
        self.l1_reconst_loss = config.l1_reconst_loss
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.num_classes = config.num_classes
        self.image_size = config.image_size
        self.build_model()
        
    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g = G(conv_dim=self.g_conv_dim)
        self.d1 = D1(conv_dim=self.d_conv_dim)
        self.d2 = D2(conv_dim=self.d_conv_dim)
        
        g_params = list(self.g.parameters())
        d1_params = list(self.d1.parameters())
        d_params = list(self.d1.parameters())+list(self.d2.parameters())
        self.gc_optimizer = optim.Adam(g_params, 0.001, [0.5, 0.999])
        self.g_optimizer = optim.Adam(g_params, self.lr, [0.5, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [0.5, self.beta2])
        
        if torch.cuda.is_available():
            self.g.cuda()
            self.d1.cuda()
            self.d2.cuda()
    
    def merge_images(self, sources, targets, tar2, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row*h, row*w*3])
        for idx, (s, t, t2) in enumerate(zip(sources, targets, tar2)):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*3)*h:(j*3+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*3+1)*h:(j*3+2)*h] = t
            merged[:, i*h:(i+1)*h, (j*3+2)*h:(j*3+3)*h] = t2
        return merged.transpose(1, 2, 0)
    
    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=False)
    
    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        
    def train(self):
        self.d1.apply(gaussian_weights_init)
        self.d2.apply(gaussian_weights_init)
        self.g.apply(gaussian_weights_init)
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        mnist_val_iter = iter(self.mnist_val_loader)
        svhn_per_epoch = len(svhn_iter)
        mnist_per_epoch = len(mnist_iter)
        mnistval_per_epoch = len(mnist_val_iter)
        iter_per_epoch = min(svhn_per_epoch, mnist_per_epoch)
        print (svhn_per_epoch, mnist_per_epoch, mnistval_per_epoch)
        
        # fixed mnist and svhn for sampling
        fixed_svhn_iter = svhn_iter.next()
        fixed_mnist_iter = mnist_iter.next()
        fixed_svhn = self.to_var(fixed_svhn_iter[0])
        fixed_mnist = self.to_var(gray_rgb(fixed_mnist_iter[0], fixed_mnist_iter[0].size(0), self.image_size))
        
        criterion = nn.CrossEntropyLoss()
        #precode
        codes=gen_dcode(1, 128, [1,1])
        codet=gen_dcode(0, 128, [1,1])

        f_labels = torch.LongTensor(128)
        f_labels[...] = 10
        f_labels = self.to_var(f_labels).long().squeeze()

        t_labels = torch.LongTensor(128)
        t_labels[...] = 1
        t_labels = self.to_var(t_labels).long().squeeze()

        #pretrain
        log_pre = 50
        thres = -1
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        if not os.path.exists(os.path.join(self.model_path, 'preg-20000.pkl')) :
            for step in range(20000+1):           
                if (step+1) % (svhn_per_epoch) == 0:
                    svhn_iter = iter(self.svhn_loader)
                if (step+1) % (mnist_per_epoch) == 0:
                    mnist_iter = iter(self.mnist_loader) 
                # load svhn and mnist dataset
                svhn, s_labels = svhn_iter.next()
                mnist, m_labels = mnist_iter.next()
                mnist_rgb = gray_rgb(mnist, mnist.size(0), self.image_size)
                mnist, m_labels = self.to_var(mnist_rgb), self.to_var(m_labels).long().squeeze()
                s_labels[torch.eq(s_labels, 10)]=0
                svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()
                self.reset_grad()
                fake_ss, f_, _ = self.g(svhn, codes[:svhn.size(0),...])
                loss_svhn_class = criterion(_, s_labels)
                fake_st, f_, _ = self.g(svhn, codet[:svhn.size(0),...])
                loss_svhn_class += criterion(_, s_labels)
                loss_svhn_class.backward()
                self.gc_optimizer.step()
                self.gc_optimizer.zero_grad()
                fake_tt, f_, _ = self.g(mnist, codet[:mnist.size(0),...])
                fake_ts, f_, _ = self.g(mnist, codes[:mnist.size(0),...])
                
                if (step + 1) % log_pre==0:
                    print ("[%d/20000] %.4f" %(step+1, loss_svhn_class.data[0]))
                if (step+1) % 5000 == 0:
                    p_path = os.path.join(self.model_path, 'preg-%d.pkl' %(step+1))
                    torch.save(self.g.state_dict(), p_path)
                    mnist_val_iter = iter(self.mnist_val_loader)
                    acc=test_mnistg(mnistval_per_epoch, mnist_val_iter, 100, 'tgtg', self.g, codet[:100,...], self.image_size)
                    
        else:
            print ("Start to load pretrained model...")
            self.g.load_state_dict(torch.load(os.path.join(self.model_path, 'preg-20000.pkl')))
            mnist_val_iter = iter(self.mnist_val_loader)
            acc=test_mnistg(mnistval_per_epoch, mnist_val_iter, 100, 'tgtg', self.g, codet[:100,...], self.image_size)
            
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        maxacc=0.0
        for step in range(20000):
            
            
            if (step+1) % (mnist_per_epoch) == 0:
                mnist_iter = iter(self.mnist_loader)
            if (step+1) % (svhn_per_epoch) == 0:
                svhn_iter = iter(self.svhn_loader)
            
            # load svhn and mnist dataset
            svhn, s_labels = svhn_iter.next()
            
            s_labels[torch.eq(s_labels, 10)]=0
            svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()#must squeeze
            mnist, m_labels = mnist_iter.next()
            mnist_rgb = gray_rgb(mnist, mnist.size(0), self.image_size)            
            mnist, m_labels = self.to_var(mnist_rgb), self.to_var(m_labels).long().squeeze()
            
            #============ train D ============#
            
            self.reset_grad()
            
            predict, index_posi, prelen, preth1, inter, posi = find_mnist(mnist, codet, svhn, codes, self.g, thres=0.99, thres2 = 0.992)
            
            out = self.d1(svhn)
            d_real_loss = criterion(out, s_labels)
            
            d2tout = self.d2(mnist)
            d_real_loss += criterion(d2tout[index_posi], predict[index_posi])
            
            fake_st, f_, _ = self.g(svhn, codet[:svhn.size(0),...])
            out = self.d2(fake_st)
            d_fake_loss = criterion(out, f_labels[:svhn.size(0)])         

            fake_ts, f_, _ = self.g(mnist, codes[:mnist.size(0),...])
            out = self.d1(fake_ts)
            d_fake_loss += criterion(out, f_labels[:mnist.size(0)])
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            self.reset_grad()
            
            
            #============ train G ============#
            
            self.reset_grad()
            
            predict, index_posi, prelen, preth1, inter, posi  = find_mnist(mnist, codet, svhn, codes, self.g, thres=0.99, thres2=0.992)
            fake_ss, f_, svhn_class = self.g(svhn, codes[:svhn.size(0),...])
            
            fake_st, fake_s_feat,  _  = self.g(svhn, codet[:svhn.size(0),...])
            out = self.d2(fake_st)
            g_loss = criterion(out, s_labels) + 0.2*criterion(_, s_labels)
            

            reconst_svhn = 0.0
            if self.l2_reconst_loss:
                reconst_svhn = torch.mean((svhn - fake_ss)**2)
            elif self.l1_reconst_loss:
                reconst_svhn = torch.mean(torch.abs(svhn - fake_ss))
            
            fake_tt, f_,  _  = self.g(mnist, codet[:mnist.size(0),...])
            
            fake_ts, fake_t_feat, _  = self.g(mnist, codes[:mnist.size(0),...])
            out = self.d1(fake_ts) 
            g_loss += (criterion(out[index_posi], predict[index_posi]))+0.15*criterion(_[index_posi], predict[index_posi])#bijing false
            
            
            reconst_mnist = 0.0
            if self.l2_reconst_loss:
                reconst_mnist = torch.mean((mnist - fake_tt)**2)                
            elif self.l1_reconst_loss:
                reconst_mnist= torch.mean(torch.abs(mnist - fake_tt)) 
                
            g_total_loss = g_loss + 10.0*(reconst_svhn + reconst_mnist)
            g_total_loss.backward()
            self.g_optimizer.step()
            self.reset_grad()

            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, '
                      'reconst_mnist: %.4f, reconst_svhn: %.4f, g_loss: %.4f, selected tgt: %d, '
                      'prelen: %d, preth1: %d, interset: %d, posith2: %d.'
                      %(step+1, self.train_iters, d_real_loss.data[0], d_fake_loss.data[0], 
                        reconst_mnist.data[0], reconst_svhn.data[0], g_loss.data[0], index_posi.size(0),
                        prelen, preth1, inter, posi))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                '''fake_svhn_to_svhn, f_, _  = self.g(fixed_svhn, codes[:self.batch_size,...])
                fake_mnist_to_mnist, f_, _  = self.g(fixed_mnist, codet[:self.batch_size,...])
                fake_mnist_to_svhn, f_, _  = self.g(fixed_mnist, codes[:self.batch_size,...])
                fake_svhn_to_mnist, f_, _  = self.g(fixed_svhn, codet[:self.batch_size,...])
                
                mnist, fake_mnist_to_mnist, fake_mnist_to_svhn = self.to_data(fixed_mnist), self.to_data(fake_mnist_to_mnist), self.to_data(fake_mnist_to_svhn)
                svhn , fake_svhn_to_svhn, fake_svhn_to_mnist = self.to_data(fixed_svhn), self.to_data(fake_svhn_to_svhn), self.to_data(fake_svhn_to_mnist)
                
                merged = self.merge_images(mnist, fake_mnist_to_mnist, fake_mnist_to_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(svhn, fake_svhn_to_svhn, fake_svhn_to_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-s-m.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)'''

                mnist_val_iter = iter(self.mnist_val_loader)
                acc=test_mnistg(mnistval_per_epoch, mnist_val_iter, 100, 'tgt', self.g, codet[:100,...], self.image_size)
                maxacc=max(maxacc, acc)
                print ('Max accuracy %.4f' %(maxacc))
                # save the model parameters for each epoch
                g_path = os.path.join(self.model_path, 'g-%d.pkl' %(step+1))
                d1_path = os.path.join(self.model_path, 'd1-%d.pkl' %(step+1))
                d2_path = os.path.join(self.model_path, 'd2-%d.pkl' %(step+1))
                torch.save(self.g.state_dict(), g_path)
                torch.save(self.d1.state_dict(), d1_path)
                torch.save(self.d2.state_dict(), d2_path)
                
        for step in range(20000, self.train_iters+1):
            
            
            if (step+1) % (mnist_per_epoch) == 0:
                mnist_iter = iter(self.mnist_loader)
            if (step+1) % (svhn_per_epoch) == 0:
                svhn_iter = iter(self.svhn_loader)
            
            svhn, s_labels = svhn_iter.next()
            
            s_labels[torch.eq(s_labels, 10)]=0
            svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()
            mnist, m_labels = mnist_iter.next()
            mnist_rgb = gray_rgb(mnist, mnist.size(0), self.image_size)            
            mnist, m_labels = self.to_var(mnist_rgb), self.to_var(m_labels).long().squeeze()
            
            #============ train D ============#
            
            self.reset_grad()
            
            predict, index_posi, prelen, preth1, inter, posi = find_mnist(mnist, codet, svhn, codes, self.g, thres=0.99, thres2 = 0.992)
            
            out = self.d1(svhn)
            d_real_loss = criterion(out, s_labels)
            
            d2tout = self.d2(mnist)
            d_real_loss += criterion(d2tout[index_posi], predict[index_posi])
            
            fake_st, f_, _ = self.g(svhn, codet[:svhn.size(0),...])
            out = self.d2(fake_st)
            d_fake_loss = criterion(out, f_labels[:svhn.size(0)])         

            fake_ts, f_, _ = self.g(mnist, codes[:mnist.size(0),...])
            out = self.d1(fake_ts)
            d_fake_loss += criterion(out, f_labels[:mnist.size(0)])
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            self.reset_grad()
            
            
            #============ train G ============#
            
            self.reset_grad()
            
            predict, index_posi, prelen, preth1, inter, posi  = find_mnist(mnist, codet, svhn, codes, self.g, thres=0.99, thres2=0.992)
            fake_ss, f_, svhn_class = self.g(svhn, codes[:svhn.size(0),...])
            
            fake_st, fake_s_feat,  _  = self.g(svhn, codet[:svhn.size(0),...])
            
            out = self.d2(fake_st)
            g_loss = criterion(out, s_labels) + 0.2*criterion(_, s_labels)
            fake_sts, fake_st_feat, _  = self.g(fake_st, codes[:svhn.size(0),...])
            g_loss += 0.2*(criterion(_, s_labels))

            reconst_svhn = 0.0
            if self.l2_reconst_loss:
                reconst_svhn = torch.mean((svhn - fake_ss)**2)
            elif self.l1_reconst_loss:
                reconst_svhn = torch.mean(torch.abs(svhn - fake_ss))
            
            fake_tt, f_,  _  = self.g(mnist, codet[:mnist.size(0),...])
            
            fake_ts, fake_t_feat, _  = self.g(mnist, codes[:mnist.size(0),...])
            out = self.d1(fake_ts) 
            g_loss += (criterion(out[index_posi], predict[index_posi]))+0.15*criterion(_[index_posi], predict[index_posi])#bijing false
            fake_tst, fake_ts_feat, _  = self.g(fake_ts, codet[:mnist.size(0),...])
            g_loss += 0.15*(criterion(_[index_posi], predict[index_posi]))
            
            reconst_mnist = 0.0
            if self.l2_reconst_loss:
                reconst_mnist = torch.mean((mnist - fake_tt)**2)                
            elif self.l1_reconst_loss:
                reconst_mnist= torch.mean(torch.abs(mnist - fake_tt)) 
                
            g_total_loss = g_loss + 10.0*(reconst_svhn + reconst_mnist)
            g_total_loss.backward()
            self.g_optimizer.step()
            self.reset_grad()

            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, '
                      'reconst_mnist: %.4f, reconst_svhn: %.4f, g_loss: %.4f, selected tgt: %d, '
                      'prelen: %d, preth1: %d, interset: %d, posith2: %d.'
                      %(step+1, self.train_iters, d_real_loss.data[0], d_fake_loss.data[0], 
                        reconst_mnist.data[0], reconst_svhn.data[0], g_loss.data[0], index_posi.size(0),
                        prelen, preth1, inter, posi))#index_posi.size(0)

            if (step+1) % self.sample_step == 0:
                '''fake_svhn_to_svhn, f_, _  = self.g(fixed_svhn, codes[:self.batch_size,...])
                fake_mnist_to_mnist, f_, _  = self.g(fixed_mnist, codet[:self.batch_size,...])
                fake_mnist_to_svhn, f_, _  = self.g(fixed_mnist, codes[:self.batch_size,...])
                fake_svhn_to_mnist, f_, _  = self.g(fixed_svhn, codet[:self.batch_size,...])
                
                mnist, fake_mnist_to_mnist, fake_mnist_to_svhn = self.to_data(fixed_mnist), self.to_data(fake_mnist_to_mnist), self.to_data(fake_mnist_to_svhn)
                svhn , fake_svhn_to_svhn, fake_svhn_to_mnist = self.to_data(fixed_svhn), self.to_data(fake_svhn_to_svhn), self.to_data(fake_svhn_to_mnist)
                
                merged = self.merge_images(mnist, fake_mnist_to_mnist, fake_mnist_to_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(svhn, fake_svhn_to_svhn, fake_svhn_to_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-s-m.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)'''

                mnist_val_iter = iter(self.mnist_val_loader)
                acc=test_mnistg(mnistval_per_epoch, mnist_val_iter, 100, 'tgt', self.g, codet[:100,...], self.image_size)
                maxacc=max(maxacc, acc)
                print ('Max accuracy %.4f' %(maxacc))
                # save the model parameters for each epoch
                g_path = os.path.join(self.model_path, 'g-%d.pkl' %(step+1))
                d1_path = os.path.join(self.model_path, 'd1-%d.pkl' %(step+1))
                d2_path = os.path.join(self.model_path, 'd2-%d.pkl' %(step+1))
                torch.save(self.g.state_dict(), g_path)
                torch.save(self.d1.state_dict(), d1_path)
                torch.save(self.d2.state_dict(), d2_path)
