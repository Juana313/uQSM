import numpy as np
import os
import torch
from random import shuffle
import nibabel as nib
    
class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data_files, training_list, patch_size=[96,96,96], voxel_size=[1,1,1], batch_size=32, shuffle=True):
        'Initialization'
        self.data_files = data_files
        self.indexes    = training_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.voxel_size = voxel_size
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 16*len(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.random.choice(self.indexes, size=1)
        
        # Generate data
        X1,m1,w1,k = self.__data_generation(indexes)

        return [X1,m1,w1,k]
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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
            
            for ii in range(self.batch_size):
                while 1:
                    ix = np.random.random_integers(rdfV.shape[0]-cnnx, size=1)[0] - 1
                    iy = np.random.random_integers(rdfV.shape[1]-cnny, size=1)[0] - 1
                    iz = np.random.random_integers(rdfV.shape[2]-cnnz, size=1)[0] - 1
                    
                    maskPatch = maskV[ix:ix+cnnx, iy:iy+cnny, iz:iz+cnnz]
                    rdfPatch = rdfV[ix:ix+cnnx, iy:iy+cnny, iz:iz+cnnz]
                    magPatch = magV[ix:ix+cnnx, iy:iy+cnny, iz:iz+cnnz]
                    
                    if maskPatch.sum() > 0.5*cnnx*cnny*cnnz:
                        break
                
                nx, ny, nz = rdfPatch.shape
                
                if nx>cnnx:
                    maskPatch  = maskPatch[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                    rdfPatch = rdfPatch[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                    magPatch  = magPatch[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                elif nx<cnnx:
                    maskPatch  = np.pad(maskPatch, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    rdfPatch = np.pad(rdfPatch, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    magPatch  = np.pad(magPatch, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            
                if ny>cnny:
                    maskPatch  = maskPatch[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                    rdfPatch = rdfPatch[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                    magPatch  = magPatch[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                elif ny<cnny:
                    maskPatch  = np.pad(maskPatch, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    rdfPatch = np.pad(rdfPatch, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    magPatch  = np.pad(magPatch, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            
                if nz>cnnz:
                    maskPatch  = maskPatch[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                    rdfPatch = rdfPatch[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                    magPatch  = magPatch[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                elif nz<cnnz:
                    maskPatch  = np.pad(maskPatch, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    rdfPatch  = np.pad(rdfPatch, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    magPatch  = np.pad(magPatch, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
    
                maskPatch = maskPatch[np.newaxis, :]
                rdfPatch = rdfPatch[np.newaxis, :]
                magPatch = magPatch[np.newaxis, :]
                
                # -------------------------------------------------
                #dipole kernel
                voxel_size = self.voxel_size[0],self.voxel_size[1],self.voxel_size[2] 
                Nx,Ny,Nz = cnnx, cnny, cnnz
                FOV = [Nx*voxel_size[0], Ny*voxel_size[1], Nz*voxel_size[2]]
                kx_squared = np.fft.ifftshift(np.arange(-Nx/2.0, Nx/2.0)/float(FOV[0]))**2
                ky_squared = np.fft.ifftshift(np.arange(-Ny/2.0, Ny/2.0)/float(FOV[1]))**2
                kz_squared = np.fft.ifftshift(np.arange(-Nz/2.0, Nz/2.0)/float(FOV[2]))**2
    
                [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
                kernel = 3*(1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D + 1e-15))
                kernel[0,0,0] = 0
                kernel = kernel[np.newaxis,:]
                
                x1_list.append(100*rdfPatch*(maskPatch!=0))
                m1_list.append((maskPatch!=0))
                w1_list.append((magPatch)*(maskPatch!=0))
                k_list.append(kernel)
                

        return np.asarray(x1_list), np.asarray(m1_list), np.asarray(w1_list), np.asarray(k_list)