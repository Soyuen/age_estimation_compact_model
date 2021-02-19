import numpy as np
import os
import argparse
from tqdm import tqdm
from TYY_utils import get_meta
import shutil

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    db = args.db
    min_score = args.min_score

    root_path = "./{}_crop/".format(db)
    dst_path="./{}_crop_1".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])):
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        t_dir=str(full_path[i][0])[0:2]
        fln=str(full_path[i][0])[3:]
        dst1=os.path.join(dst_path,t_dir)
        if not os.path.exists(dst1):
            os.mkdir(dst1)
        dstf=os.path.join(dst1,fln)
        ff=root_path + str(full_path[i][0])
        shutil.copy(ff,dstf)

if __name__ == '__main__':
    main()
