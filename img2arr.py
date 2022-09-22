import os 
import sys
import shutil
import numpy as np
from Utils import *
from Config import *

argv = sys.argv[1:]
if len(argv) < 2:
    print("Error")

else:
    input_path = argv[0]
    output_path = argv[1]

if not os.path.exists(output_path):
    os.makedirs(output_path)

for f in os.listdir(input_path):
    if f.find(".png") != -1:
        img = Utils.imgPreprocess("{}/{}".format(input_path, f), IMAGE_SIZE)
        filename = f[:f.find(".png")]

        np.savez_compressed("{}/{}".format(output_path, filename), features = img)
        features = np.load("{}/{}.npz".format(output_path, filename))["features"]

        assert np.array_equal(img, features)
        shutil.copyfile("{}/{}.gui".format(input_path, filename), "{}/{}.gui".format(output_path, filename))

print("Arrays in: {}".format(output_path))