import numpy as np
import cv2 as cv

class Utils:
    @staticmethod
    def oneHotEncoding(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)
    
    @staticmethod
    def imgPreprocess(img_path, image_size):
        img = cv.imread(img_path)
        img = cv.resize(img, (image_size, image_size))
        img = img.astype("float32")
        img /= 255
        return img

    @staticmethod
    def show(image):
        cv.namedWindow("view", cv.WINDOW_AUTOSIZE)
        cv.imshow("view", image)
        cv.waitKey(0)
        cv.destroyWindow("view")
    