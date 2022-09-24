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
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                filename = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, filename)):
                    img = Utils.imgPreprocess("{}/{}.png".format(path, filename), IMAGE_SIZE)
                    self.append(filename, gui, img)

                elif os.path.isfile("{}/{}.npz".format(path, filename)):
                    arr = np.load("{}/{}.npz".format(path, filename))["features"]
                    self.append(filename, gui, arr)
        
        self.voc.serializeBinaryConvert()
        self.next_words = self.makeSparseLabels(self.next_words, self.voc)
        if isBinSeq:
            self.partial_sequences = self.createBinary(self.partial_sequences, self.voc)
        else :
            self.partial_sequences = self.makeIndices(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset Size: {}".format(self.size))
        print("Vocabulary Size:{}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input Shape: {}".format(self.input_shape))
        print("Output Size: {}".format(self.output_size))
        
    def arrayConvert(self):
        # print("Converting Arrays...")

        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self,sample_id, gui, img, to_show = False):
        if to_show:
            pic = img*255
            pic = np.array(pic,dtype=np.uint8)
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(","," ,").replace("\n"," \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)

        token_sequence.append(END_TOKEN)

        suffix = [PLACEHOLDER]*CONTEXT_LENGTH

        a = np.concatenate([suffix,token_sequence])
        for j in range(0,len(a) - CONTEXT_LENGTH):
            context = a[j:j+CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)
    
    @staticmethod
    def makeIndices(parital_sequence,voc):
        temp = []
        for sequence in parital_sequence:
            sparse_vector_seq = []
            for token in sequence:
                sparse_vector_seq.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vector_seq))
        return temp
    
    @staticmethod
    def createBinary(partial_sequence,voc):
        temp = []
        for sequence in partial_sequence:
            sparse_vector_seq = []
            for token in sequence:
                sparse_vector_seq.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vector_seq))
        return temp
    
    @staticmethod
    def makeSparseLabels(next_words,voc):
        print("hello")
        temp = []
        for label in next_words:
            # print(label)
            temp.append(voc.binary_vocabulary[label])
        return temp
    
    def saveMetadata(self,path):
        np.save("{}/meta_datatset".format(path),np.array([self.input_shape, self.output_size,self.size]))