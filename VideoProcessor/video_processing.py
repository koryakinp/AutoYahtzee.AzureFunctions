import cv2
import numpy as np
import logging
import tempfile
import os


def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def get_diffs(data, smoothing):

    diffs = []
    i = 0
    while i < len(data) - 1:
        diffs.append(((data[i]/255 - data[i + 1]/255)**2).sum())
        i += 1

    smoothed_diffs = running_mean(diffs, 5).tolist()
    return smoothed_diffs

def get_range(data, threshold, offset):
    i = 0
    while i < len(data):
        if data[i] >= threshold:
            break
        i += 1

    first = i

    i = len(data) - 1
    while i >= 0:
        if data[i] >= threshold:
            break
        i -= 1

    last = i
    
    first -= offset
    last += offset + 10
    
    if first < 0:
        first = 0
        
    if last > len(data) - 1:
        last = len(data) - 1
    
    return first, last

def get_grayscale_frames(filename):
    
    frames_color = []
    frames_gray = []
    
    cap = cv2.VideoCapture(filename)
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(1000)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
        frames_color.append(frame)
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            cv2.waitKey(1000)
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
            
    return frames_color, frames_gray

def save_video(name, data, fps, codec):
    shape = data[0].shape
    shape = (shape[1], shape[0])
    
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*codec), fps, shape)

    for i in range(len(data)):
        out.write(data[i])

    out.release()

def save_image(name, data):
    cv2.imwrite(name, data)
    
def rotate_image(image, angle, center):
    image_center = tuple(np.array(image.shape)/2)
    shape = tuple(image.shape)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    image = np.asarray(image[:,:])
    return cv2.warpAffine(image, rot_mat, (480, 480), flags=cv2.INTER_LINEAR)

def shift_image(img, shape, top, left):
    M = np.float32([[1,0,top],[0,1,left]])
    return cv2.warpAffine(img,M,shape)

def crop_image(image,y,h,x,w):
    return image[y:y+h, x:x+w]

def process_video(
    file_src, 
    still_threshold=30, still_avg=3, angle=-3, top=20, left=30):
    frames_color, frames_gray = get_grayscale_frames(file_src)
    smoothed_diffs = get_diffs(frames_gray, still_avg)
    first, last = get_range(smoothed_diffs, still_threshold, still_avg)
    processed_frames = frames_color[first:last]
    processed_frames = [rotate_image(frame, angle, (240, 240)) for frame in processed_frames]
    processed_frames = [shift_image(frame, (480, 480), top, left) for frame in processed_frames]
    processed_frames = [crop_image(frame, 40,410,30,410) for frame in processed_frames]

    mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    webm = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    jpg = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

    logging.info('Frames to processed: ' + str(len(processed_frames)))

    save_video(webm.name, processed_frames, 25, 'vp80')
    logging.info('WEBM saved: ' + str(os.path.isfile(webm.name)))

    save_video(mp4.name, processed_frames, 25, 'mp4v')
    logging.info('MP4 saved: ' + str(os.path.isfile(mp4.name)))

    save_image(jpg.name, processed_frames[-1])
    logging.info('JPG saved: ' + str(os.path.isfile(jpg.name)))

    return mp4.name, webm.name, jpg.name



