import cv2
import numpy as np

# resize all images to (3, 224, 224) for batch process
def img_load_resize(root, num):
    for i in range(num):
        if (i % 500 == 0):
            print(i)
        path = root + str(i) + ".jpg"
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = img.reshape(3, 224, 224)
        img = img / 256
        np.save(root + str(i), img)

# pack all images into a single tensor to reduce disk access latency
def img_pack(root, num, dict_path):
    imgs = []
    for i in range(num):
        img_path = root + str(i) + ".npy"
        im = np.load(img_path, allow_pickle=True)
        if (i % 500 == 0):
            print(img_path, im.shape)
        imgs.append(im)

    imgs = np.stack(imgs, axis=0)
    print(imgs.shape)
    np.save(dict_path, imgs)

# use coco dataset val2017
# the size of data is about 200MB before preprocess, 1GB after preprocessed
# the latency of low-level memory (disk) access is eliminated
# by loading all data into memory before running
# and the focus is on high level memory access optimization and parallel optimization
if __name__ == "__main__":
    root = "./data/image/"
    dict_path = "./data/img_dict.npy"
    num = 1000
    img_load_resize(root, num)
    img_pack(root, num, dict_path)
