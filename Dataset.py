from this import s
from Vocabulary import *
from Utils import *
from Config import *
import os

class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words  = []

        self.voc = Vocabulary()
        self.size = 0


    @staticmethod
    def pathLoader(path):
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gp = "{}/{}".format(path, f)
                gui_paths.append(gp)
                filename = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, filename)):
                    pathimg = "{}/{}.png".format(path, filename)
                    img_paths.append(pathimg)

                elif os.path.isfile("{}/{}.npz".format(path, filename)):
                    pathnpz = "{}/{}.npz".format(path, filename)
                    img_paths.append(pathnpz) 


        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths
    
    def loadData(self, path, isBinSeq = False):
        for f in os.listdir(path):
            if f.find(".gui" != -1):
                gui = open("{}/{}".format(path, f), 'r')
                filename = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, filename)):
                    img = Utils.imgPreprocess("{}/{}.png".format(path, filename), IMAGE_SIZE)
                    self.append(filename, gui, img)

                elif os.path.isfile("{}/{}.npz".format(path, filename)):
                    arr = np.load("{}/{}.npz".format(path, filename))["features"]
                    self.append(filename, gui, arr)
        
        self.voc.serializeBinaryConvert()
        self.next_words = self.oneHotEncoding(self.next_words, self.voc)
        if isBinSeq:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else :
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset Size: {}".format(self.size))
        print("Vocabulary Size:{}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input Shape: {}".format(self.input_shape))
        print("Output Size: {}".format(self.output_size))
        