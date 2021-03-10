
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet3D


from datagenerator import DataGenerator
from datagenerator_valid import DataGeneratorValid
from utils import fetch_data_files, get_validation_split
from loss import tv_loss
from fmLayer import CalFMLayer
from NDI import NDIErr, DoMask
import numpy as np
import nibabel as nib
import kornia

config = dict()
config["pool_size"] = (2,2,2)           # pool size for the max pooling operations
config["n_base_filters"] = 24           # num of base kernels
config["conv_kernel"] = (3,3,3)         # convolutional kernel shape
config["layer_depth"] = 5               # unet depth
config["deconvolution"] = False         # if False, will use upsampling instead of deconvolution
config["batch_normalization"] = False    # Using batch norm
config["activation"] = "linear"

config["train_patch_size"] = [96, 96, 96]
config["test_patch_size"] = [160, 160, 160]
config["voxel_size"] = [1, 1, 1]

config["initial_learning_rate"] = 0.0001
config["batch_size"] = 3
config["epochs"] = 10

config['data_path'] = r'D:\Projects\QSM\Data'
config['data_filenames'] = ['RDF.nii.gz', 'Mask.nii.gz', 'iMag.nii.gz']



def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
        
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)  
    
def readNifti(fileName):
    img = nib.load(fileName)
    return img.get_fdata()

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    device = torch.device("cuda")
    #device = torch.device("cpu")
    
    torch.backends.cudnn.benchmark = True
    
    # get the data files
    data_files = fetch_data_files(config['data_path'], config['data_filenames'])
    print("num of datasets %d" % (len(data_files)))

    # -------------------------------
    # create data generator for training and validatation, it can load the data from memory pretty fast using multiple workers and buffers if you need to load your data batch by batch
    training_list, validation_list = get_validation_split(data_files,
                                                          'training.pkl',
                                                          'val.pkl',
                                                          data_split=0.8,
                                                          overwrite=True)
    # To make sure the num of training and validation cases is dividable by num_gpus when doing multi-GPUs training
    training_set = DataGenerator(data_files,
                                 training_list,
                                 patch_size = config["train_patch_size"],
                                 voxel_size = config["voxel_size"],
                                 batch_size=config["batch_size"],
                                 shuffle=True)
    validation_set = DataGeneratorValid(data_files,
                                        validation_list,
                                        patch_size = config["test_patch_size"],
                                        voxel_size = config["voxel_size"],
                                        batch_size=1,
                                        shuffle=False)
    
    params_training = {'batch_size': config["batch_size"], 'shuffle': True, 'num_workers': 4}
    params_valid = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
    training_generator = torch.utils.data.DataLoader(training_set, **params_training)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_valid)

    model = UNet3D(in_channels = 1, 
                    out_channels = 1, 
                    final_sigmoid=False, 
                    f_maps=config["n_base_filters"], 
                    layer_order='cl',
                    num_groups=1, 
                    num_levels=config["layer_depth"], 
                    is_segmentation=False)
    if torch.cuda.is_available():
        model.cuda()
        
    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config["initial_learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        iteration = 0
        
        for batch in training_generator:
            optimizer.zero_grad()
            
            rdf, mask, w, dipole_kernel = batch[0][0].to(device, dtype=torch.float), batch[1][0].to(device, dtype=torch.float), batch[2][0].to(device, dtype=torch.float), batch[3][0].to(device, dtype=torch.float)
            
            chi = model(rdf)
            chi = DoMask()([chi, mask])
            
            fm_chi = CalFMLayer()([chi, dipole_kernel])
            fm_chi = DoMask()([fm_chi, mask])
            
            
            ndi = NDIErr()([rdf, fm_chi, w])
            
            loss_ndi = torch.mean(torch.square(ndi))
            loss_tv = tv_loss(chi)
            
            loss = loss_ndi + 0.001*loss_tv  
           
            
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
    
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} NDI Loss: {:.4f} TV Loss: {:.4f} ".format(epoch, iteration, len(training_generator), loss.item(), loss_ndi.item(), loss_tv.item()))
            iteration += 1
        
        scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
            
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_generator)))
  
 
        model_out_path = "model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        model_out_path = "model.pth"
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        
        avg_loss = 0
        with torch.no_grad():
            idx = 0
            for batch in validation_generator:
                rdf, mask, w, dipole_kernel = batch[0][0].to(device, dtype=torch.float), batch[1][0].to(device, dtype=torch.float), batch[2][0].to(device, dtype=torch.float), batch[3][0].to(device, dtype=torch.float)
            
                chi = model(rdf)
                chi = DoMask()([chi, mask])
                
                chi_pred = (chi.cpu().detach().numpy()[0,0])/100*3.0
                saveNifti(chi_pred, 'chi_pred_'+'epoch'+str(epoch)+'_data'+str(idx)+'_.nii.gz')

            
                fm_chi = CalFMLayer()([chi, dipole_kernel])
                fm_chi = DoMask()([fm_chi, mask])
            
                ndi = NDIErr()([rdf, fm_chi, w])
                
                loss_ndi = torch.mean(torch.square(ndi))
                loss_tv = tv_loss(chi)
                 
                loss = loss_ndi + 0.001*loss_tv
                
                avg_loss += loss.item()
                idx += 1
                
        print("===> Avg. Loss: {:.4f}".format(avg_loss / len(validation_generator)))
    
if __name__ == "__main__":
    main()
