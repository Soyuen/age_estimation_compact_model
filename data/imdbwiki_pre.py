import tensorflow.compat.v1 as tf    
from mtcnn.mtcnn import MTCNN
import cv2
import argparse
from tqdm import tqdm
import glob
import numpy as np
from PIL import Image
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="imdb",
                        help="dataset; wiki or imdb")
    args = parser.parse_args()
    return args
def birth(path):
    img_name = path
    Split=img_name.split("_")
    Birth=Split[-2].split("-")
    Y_birth=int(Birth[0])
    M_birth=int(Birth[1])
    Photo=Split[-1].split(".")
    Photo=int(Photo[0])
    if M_birth>7:
        age=Photo-Y_birth-1
    else:
        age=Photo-Y_birth
    return age
def mtdetect(img):
    global detector
    try:
        face_list = detector.detect_faces(img) # face detect and alignment
    except:    
        return np.array([-1])
    if face_list==[]:
        return np.array([-1])
    elif np.size(face_list)>1:
        return np.array([-1])
    else:
        for face in face_list:
            box = face["box"]
            box=np.array(box)
            box[box<0]=0
            x,y,w,h = box
            if 6400<h*w<45000:
                return box
            else:
                return np.array([-1])

def main():
    args = get_args()
    db = args.db
    cwd=os.getcwd()
    root_path = cwd+"/{}_crop_1".format(db)
    img_size=64
    out_ages,out_imgs,out_20ns,out_4ns,out_1ns = [],[],[],[],[]
    for i in tqdm(sorted(glob.glob(root_path+"/**/*.jpg"))):
        img = cv2.imread(i)
        img = cv2.resize(img,(240,240))
        box= mtdetect(img)
        k=i.split("\\")
        k=k[-1]
        if ~(np.any(box==-1)):
            x,y,w,h = box
            cropped = img[y:y+h, x:x+w]
            del x,y,w,h
        else:
            continue
        out_ages.append(int(birth(i)))
        out_imgs.append(cv2.resize(cropped, (64, 64)))
        out_20ns.append(int(birth(i))//20*2)
        out_4ns.append(int(birth(i))%20//4*4)
        out_1ns.append(int(birth(i))%20%4)
        del cropped,box
        
    np.savez("{}.npz".format(db),image=np.array(out_imgs), age=np.array(out_ages), img_size=img_size,age20=np.array(out_20ns),age4=np.array(out_4ns),age1=np.array(out_1ns))        
if __name__ == '__main__':
    detector = MTCNN()
    main()
