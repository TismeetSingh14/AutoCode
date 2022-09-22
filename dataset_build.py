import os
import sys
import hashlib
import shutil
import numpy as np
from importlib.metadata import distribution
# from classes.Sampler import *
argv = sys.argv[1:]
if len(argv) < 1:
    print("Error")
else:
    input_path = argv[0]

distribution = 6 if len(argv) < 2 else argv[1]
TRAINING_SET = "TRAINING_SET"
EVALUATION_SET = "EVALUATION_SET"

paths = []
for f in os.listdir(input_path):
    if f.find(".gui") != -1:
        path_gui = "{}/{}".format(input_path, f)
        filename = f[:f.find(".gui")]

        if os.path.isfile("{}/{}.png".format(input_path, filename)):
            pathimg = "{}/{}.png".format(input_path, filename)
            paths.append(filename)

count_evaluation_samples = len(paths)/(distribution + 1)
count_training_samples = count_evaluation_samples * distribution

assert count_training_samples + count_evaluation_samples == len(paths)

print("Data Split")
print("Training Samples: {}".format(count_training_samples))
print("Evaluation Samples: {}".format(count_evaluation_samples))

np.random.shuffle(paths)

eval_set = []
train_set = []
hashes = []

for path in paths:
    if sys.version_info >= (3,):
        f = open("{}/{}.gui".format(input_path, path), 'r', encoding = "utf-8")
    else:
        f = open("{}/{}.gui".format(input_path, path), 'r')

    with f:
        chars = ""
        for line in f:
            chars += line
        
        content_hash = chars.replace(" ", "").replace("\n","")
        content_hash = hashlib.sha256(content_hash.encode("utf-8")).hexdigest()

        if len(eval_set) == count_evaluation_samples:
            train_set.append(path)

        else:
            isUnique = True
            for h in hashes:
                if h is content_hash:
                    isUnique = False
                    break
            if isUnique:
                eval_set.append(path)
            else:
                train_set.append(path)

        hashes.append(content_hash)
assert len(eval_set) == count_evaluation_samples
assert len(train_set) == count_training_samples

if not os.path.exists("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET)):
    os.makedirs("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET))

if not os.path.exists("{}/{}".format(os.path.dirname(input_path), TRAINING_SET)):
    os.makedirs("{}/{}".format(os.path.dirname(input_path), TRAINING_SET))

for path in eval_set:
    shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), EVALUATION_SET, path))
    shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), EVALUATION_SET, path))

for path in train_set:
    shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), TRAINING_SET, path))
    shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), TRAINING_SET, path))

print("Training_Set:{}/training_set".format(os.path.dirname(input_path), path))
print("Evaluation_Set:{}/eval_set".format(os.path.dirname(input_path), path))