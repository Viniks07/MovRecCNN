import numpy as np
import time

#############################################|Função de espelhamento da imagem|#####################################################
mirroring = lambda cam_frame: cam_frame[:,::-1,:]

#######################################|Função para a conversão em escala de cinza|#################################################
def gray_scale(cam_frame):
    if cam_frame.ndim <= 2:
        raise ValueError('O Frame deve ter canais de cores')
    return np.dot(cam_frame[:,:,:],[0.114, 0.587, 0.299]).astype(np.uint8)
 
####################################################|Função Binarização|############################################################
def binarization(cam_frame,limiar = 127):
    if cam_frame.ndim != 2:
        raise ValueError('O Frame deve ser bidimensional (grayscale).')

    cam_frame = cam_frame.copy()
    return np.where(cam_frame < limiar, 0, 255).astype(np.uint8)

##############################################|Função Background Subtraction|#######################################################
def background_subtraction(cam_frame, start_time=3, limiar=15):
    
    cam_frame = cam_frame.copy()

    if cam_frame.ndim != 2:
        raise ValueError('O Frame deve ser bidimensional (grayscale).')

    if not hasattr(background_subtraction, "start"):
        background_subtraction.start = time.time()
        background_subtraction.background = None

    if background_subtraction.background is None:
        if start_time <= 0 or time.time() - background_subtraction.start >= start_time:
            background_subtraction.background = cam_frame.copy().astype(np.int16)

    cam_frame = cam_frame.astype(np.int16)

    if background_subtraction.background is not None:
        cam_frame[np.abs(cam_frame - background_subtraction.background) < limiar] = 0

    return cam_frame.astype(np.uint8)

##################################################|Função de Down Sample|###########################################################
def down_sampling(cam_frame,division= 16):
    
    visualizer = cam_frame.copy()
    
    if cam_frame.ndim == 3:
        down_sample = np.zeros(shape=(cam_frame.shape[0]//division,cam_frame.shape[1]//division,3),dtype=np.uint8)
    elif cam_frame.ndim == 2:
        down_sample = np.zeros(shape=(cam_frame.shape[0]//division,cam_frame.shape[1]//division),dtype=np.uint8)

    for i in range(0, cam_frame.shape[0], division):
        for j in range(0, cam_frame.shape[1], division):

            block = visualizer[i:i+division, j:j+division]

            if cam_frame.ndim == 3:
                block_mean = block.mean(axis=(0, 1)).astype(np.uint8)
            else:
                block_mean = int(block.mean())

            visualizer[i:i+division, j:j+division] = block_mean

            down_sample[i // division, j // division] = block_mean

    return (visualizer,down_sample)

##################################################| D E S C O N T I N U A D A S |###################################################

#Função Target(Descontinuada)
def color_target(cam_frame,size = 10,height_width = (5,5),color = 'red'):
    
    height,width = height_width

    cp =(cam_frame.shape[0]*height//10,cam_frame.shape[1]*width//10)

    target_values = cam_frame[cp[0]-size+1:cp[0]+size-1,cp[1]-size+1:cp[1]+size-1]
    cam_frame_target = cam_frame.copy()

    if color == 'blue':
        color = [255,0,0]
    elif color == 'green':
        color = [0,255,0]
    else:
        color = [0,0,255]

    cam_frame_target[cp[0]-size:cp[0]+size,cp[1]-size:cp[1]+size] = color
    cam_frame_target[cp[0]-size+1:cp[0]+size-1,cp[1]-size+1:cp[1]+size-1] = target_values

    target_values = cam_frame[cp[0]-size+1:cp[0]+size-1,cp[1]-size+1:cp[1]+size-1]
    
    target_RGB_means = (np.mean(target_values[:,:,2]),np.mean(target_values[:,:,1]),np.mean(target_values[:,:,0]))
    target_RGB_stds = (np.std(target_values[:,:,2]),np.std(target_values[:,:,1]),np.std(target_values[:,:,0]))

    return (cam_frame_target,target_RGB_means,target_RGB_stds)

#Função de Chroma Key (Descontinuada)
def chroma_key(cam_frame,target_RGB_means=(125,125,125),target_RGB_stds =(25,25,25),limiar = 8):

    condition = ( (target_RGB_means[0] - target_RGB_stds[0] * limiar < cam_frame[:, :, 2]) &
                  (target_RGB_means[0] + target_RGB_stds[0] * limiar > cam_frame[: ,: ,2]) &
                  (target_RGB_means[1] - target_RGB_stds[1] * limiar < cam_frame[:, :, 1]) &
                  (target_RGB_means[1] + target_RGB_stds[1] * limiar > cam_frame[:, :, 1]) &
                  (target_RGB_means[2] - target_RGB_stds[2] * limiar < cam_frame[:, :, 0]) &
                  (target_RGB_means[2] + target_RGB_stds[2] * limiar > cam_frame[:, :, 0]) )

    mask_cam = cam_frame.copy()
    mask_cam[condition] = 0
    return mask_cam