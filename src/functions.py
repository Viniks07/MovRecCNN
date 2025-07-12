import numpy as np
import time

#############################################|Função de espelhamento da imagem|#####################################################
def mirroring(cam_frame):
    return cam_frame[:,::-1,:].copy()

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
    
    visualizer = visualizer.astype(np.uint8)
    down_sample = down_sample.astype(np.uint8)

    return (down_sample,visualizer)

##################################################|Função Bounding Box|#############################################################
def bounding_box(cam_frame, frame_vizualizer=None,division = 16):
    
    if frame_vizualizer is None:
        frame_vizualizer = cam_frame.copy()
    else:
        frame_vizualizer = frame_vizualizer.copy()

    if frame_vizualizer.ndim == 2:
        frame_vizualizer = np.stack([frame_vizualizer]*3, axis=-1)
    
    if cam_frame.ndim != 2:
        cam_frame = gray_scale(cam_frame).copy()
    
    y, x = np.where(cam_frame == 255)
    if len(x) == 0 or len(y) == 0:
        return (division, division, division, division),frame_vizualizer

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    frame_vizualizer[y_min:y_max+1, x_min] = [255,0,150] #Esquerda 
    frame_vizualizer[y_min:y_max+1, x_max] = [255,0,150] #Direita 
    frame_vizualizer[y_min, x_min:x_max+1] = [255,0,150] #Cima
    frame_vizualizer[y_max, x_min:x_max+1] = [255,0,150] #Baixo
#               CIMA               BAIXO                      
    return (y_min//division,(y_max-division+1)//division,x_min//division,(x_max-division +1)//division),frame_vizualizer

###################################################|Função de Gradeamento|###########################################################
def grid(cam_frame):
    cam_frame = cam_frame.copy()

    cam_frame[:,np.r_[160,480],:] = [255,255,255]
    cam_frame[160,160:480,:] = [255,255,255]
    cam_frame[240,np.r_[0:160,480:640],:] = [255,255,255]

    return cam_frame

##################################################|Função de Vetorização|###########################################################
def vectorize(sample, bbox_points, division= 16):

    cut_image = sample[bbox_points[0]:bbox_points[1], bbox_points[2]:bbox_points[3]]
    somas = np.sum(cut_image, axis=0)

    if np.any(somas > 0):
        center_points = np.argmax(somas)

        total_width_length = 640 // division 
        target_center = total_width_length // 2

        left_length = center_points
        right_length = (cut_image.shape[1]) - center_points

        pad_left = target_center - left_length
        pad_right = total_width_length - target_center - right_length

        vector = np.pad(
            cut_image,
            pad_width=(
                (abs((480 // division) - cut_image.shape[0]), 0),
                (abs(pad_left), abs(pad_right))
            ),
            mode='constant',
            constant_values=0
        )

        vector = vector // 255

        vector = vector.flatten()
    else:
        vector = np.zeros((480 // division, 640 // division), dtype=np.uint8)
    return vector

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