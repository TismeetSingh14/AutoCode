import sys
import numpy as np

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "
SEPARATOR = "->"

class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}
        self.token_lookup = {}
        self.size = 0

        self.addToken(START_TOKEN)
        self.addToken(END_TOKEN)
        self.addToken(PLACEHOLDER)


    def addToken(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1

    def binaryConvert(self):
        items = self.vocabulary.items()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary
    
    def serializeBinaryConvert(self):
        if len(self.binary_vocabulary) == 0:
            self.binaryConvert()
        
        mystr = ""
        items = self.binary_vocabulary.items()
        for key, value in items:
            strarr = np.array2string(value, separator=',', max_line_width = self.size*self.size)
            mystr += "{}{}{}\n".format(key, SEPARATOR, strarr[1:len(strarr) - 1])
    
        return mystr

    def save(self, path):
        output_file = "{}/words.vocab".format(path)
        file = open(output_file, 'w')
        file.write(self.serializeBinaryConvert())
        file.close()
    
    def dataRead(self, path):
        input_file = "{}/words.vocab".format(path)
        file = open(input_file, 'r')
        buffer = ""
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(SEPARATOR)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(SEPARATOR):]
                value = np.fromstring(value, sep = ',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = np.where(value == 1)[0][0]
                self.token_lookup[np.where(value == 1)[0][0]] = key

                buffer = ""
            except ValueError:
                buffer += line
        file.close()
        self.size= len(self.vocabulary)