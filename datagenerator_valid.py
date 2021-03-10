import numpy as np
import os
import torch
from random import shuffle
import nibabel as nib
    
class DataGeneratorValid(torch.utils.data.Dataset):
    def __init__(self, data_files, validation_list, patch_size=[160,160,160], voxel_size=[1,1,1], batch_size=1, shuffle=False):
        'Initialization'
        self.data_files = data_files
        self.indexes    = validation_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.voxel_size = voxel_size
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index:index+self.batch_size]
        
        # Generate data
        X1,m1,w1,k = self.__data_generation(indexes)

        return [X1,m1,w1,k]
    
    def on_epoch_end(self):
        pass

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x1_list = list()
        w1_list = list()
        m1_list = list()
        k_list = list()
        
        
        # Generate data
        for i, index in enumerate(indexes):           
            image_list = list()     
            for k, image_file in enumerate(self.data_files[index]):
                image = nib.load(os.path.abspath(image_file))
                image_list.append(image)
               
            subject_data = [image.get_data() for image in image_list]
            
            rdf = np.asarray(subject_data[0]) 
            m = np.asarray(subject_data[1]) 
            mag = np.asarray(subject_data[2]) 

            TE = 25000 * 1e-6;           
            B0 = 3;                      
            gyro = 2*np.pi*42.58;
            rdf /= (TE*B0*gyro)
            mag /= mag.max()
            m = (rdf!=0)

            d1 = np.max(np.max(m, axis=1), axis=1)
            d1first = np.nonzero(d1)[0][0]
            d1last = np.nonzero(d1)[0][-1]
        
            d2 = np.max(np.max(m, axis=0), axis=1)
            d2first = np.nonzero(d2)[0][0]
            d2last = np.nonzero(d2)[0][-1]
        
            d3 = np.max(np.max(m, axis=0), axis=0)
            d3first = np.nonzero(d3)[0][0]
            d3last = np.nonzero(d3)[0][-1]                        
            
            maskV = m[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            rdfV = rdf[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            magV = mag[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]            
            
            cnnx, cnny, cnnz = self.patch_size[0], self.patch_size[1], self.patch_size[2]
            nx, ny, nz = rdfV.shape
            
            if nx>cnnx:
                maskV  = maskV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                rdfV = rdfV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                magV  = magV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
            elif nx<cnnx:
                maskV  = np.pad(maskV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                rdfV = np.pad(rdfV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                magV  = np.pad(magV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        
            if ny>cnny:
                maskV  = maskV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                rdfV = rdfV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                magV  = magV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
            elif ny<cnny:
                maskV = np.pad(maskV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                rdfV = np.pad(rdfV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                magV  = np.pad(magV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        
            if nz>cnnz:
                maskV  = maskV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                rdfV = rdfV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                magV  = magV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
            elif nz<cnnz:
                maskV  = np.pad(maskV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                rdfV  = np.pad(rdfV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                magV  = np.pad(magV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))

            maskV = maskV[np.newaxis, :]
            rdfV = rdfV[np.newaxis, :]
            magV = magV[np.newaxis, :]
            
            # -------------------------------------------------
            #dipole kernel
            voxel_size = self.voxel_size[0], self.voxel_size[1], self.voxel_size[2]
            Nx,Ny,Nz = cnnx, cnny, cnnz
            FOV = [Nx*voxel_size[0], Ny*voxel_size[1], Nz*voxel_size[2]]
            kx_squared = np.fft.ifftshift(np.arange(-Nx/2.0, Nx/2.0)/float(FOV[0]))**2
            ky_squared = np.fft.ifftshift(np.arange(-Ny/2.0, Ny/2.0)/float(FOV[1]))**2
            kz_squared = np.fft.ifftshift(np.arange(-Nz/2.0, Nz/2.0)/float(FOV[2]))**2

            [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
            kernel = 3*(1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D + 1e-15))
            kernel[0,0,0] = 0
            kernel = kernel[np.newaxis,:]
                
            x1_list.append(100*rdfV*(maskV!=0))
            m1_list.append((maskV!=0))
            w1_list.append((magV)*(maskV!=0))
            k_list.append(kernel)
                

        return np.asarray(x1_list), np.asarray(m1_list), np.asarray(w1_list), np.asarray(k_list)