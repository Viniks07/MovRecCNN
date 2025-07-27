import numpy as np
import time


def mirroring(cam_frame):
    return cam_frame[:,::-1,:].copy()


def gray_scale(cam_frame):
    if cam_frame.ndim <= 2:
        raise ValueError('O Frame deve ter canais de cores')

    return np.dot(cam_frame[:,:,:],[0.114, 0.587, 0.299]).astype(np.uint8)


def background_subtraction(cam_frame, start_time=0.1, limiar=30):
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


def binarization(cam_frame,limiar = 5):
    if cam_frame.ndim != 2:
        raise ValueError('O Frame deve ser bidimensional (grayscale).')

    cam_frame = cam_frame.copy()
   
    return np.where(cam_frame < limiar, 0, 255).astype(np.uint8)


def dilate(frame, kernel = np.ones(dtype=np.uint8,shape=(16,16)), iterations=1):
    frame = frame.copy()
    for _ in range(iterations):
        padded = np.pad(frame, 
                        ((kernel.shape[0]//2, kernel.shape[0]//2),
                         (kernel.shape[1]//2, kernel.shape[1]//2)),
                        mode='constant', constant_values=0)
        result = np.zeros_like(frame)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                if np.any(region[kernel ==1] == 255):
                    result[i, j] = 255
        frame = result
    return frame


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


def bounding_box(cam_frame, frame_vizualizer=None,division = 16):
    cam_frame = cam_frame.copy()

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
                      
    return (y_min//division,(y_max-division+1)//division,x_min//division,(x_max-division +1)//division),frame_vizualizer


def centralize(sample, bbox_points):
    cut_image = sample[bbox_points[0]:bbox_points[1], bbox_points[2]:bbox_points[3]].copy()
    total_height,total_width = sample.shape
    ys, xs = np.nonzero(cut_image)

    if len(xs) == 0 and len(ys) == 0:

        return np.zeros((30, 40), dtype=np.uint8)

    cm_x = int(np.round(np.mean(xs))) + bbox_points[2]
    center_x_cut = cm_x - bbox_points[2]

    left_pad = max(total_width//2 - center_x_cut, 0)
    right_pad = max(total_width - (cut_image.shape[1] + left_pad), 0)
    pad_top = max(total_height - cut_image.shape[0], 0)

    cut_image = np.pad(
        cut_image,
        ((pad_top, 0), (left_pad, right_pad)),
        mode='constant',
        constant_values=0
    )

    cut_image = cut_image[:total_height,:total_width]

    return cut_image.astype(np.uint8)