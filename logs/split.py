import cv2
import os

if __name__ == "__main__":
    path = "./rain800_test"  # change this dirpath.
    listdir = os.listdir(path)

    newdir = path
    if (os.path.exists(newdir) == False):
        os.mkdir(newdir)

    for i in listdir:
        # if i.split('.')[1] == "jpg":  # the format of zed img.
        #     filepath = os.path.join(path, i)
        #     filename = i.split('.')[0]
        leftpath = "./test/Rain800/norain/"+i
        rightpath = "./test/Rain800/rain/"+i

        img = cv2.imread(path+'/'+i)
        [h, w] = img.shape[:2]
        limg = img[:, :int(w / 2), :]
        rimg = img[:, int(w / 2):, :]

        cv2.imwrite(leftpath, limg)
        cv2.imwrite(rightpath, rimg)
