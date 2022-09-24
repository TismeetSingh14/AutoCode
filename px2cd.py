from email import generator
from tabnanny import verbose
from keras.layers import *
from keras.models import *
from Config import *
from keras import *
from keras.optimizers import *
from keras.optimizers import Adam
from AModel import *

class AutoCode():
    def __init__(self, input_shape, output_size, path):
        AModel.__init__(self, input_shape, output_size, path)
        self.name = "Code-Automaton"
    
        model_i = Sequential()
        model_i.add(Conv2D(32, (3,3), padding = "valid", activation="relu", input_shape = input_shape))
        model_i.add(Conv2D(32, (3,3), padding = "valid", activation="relu"))
        model_i.add(MaxPooling2D(pool_size = (2,2)))
        model_i.add(Dropout(0.3))

        model_i.add(Conv2D(64, (3,3), padding = "valid", activation="relu"))
        model_i.add(Conv2D(64, (3,3), padding = "valid", activation="relu"))
        model_i.add(MaxPooling2D(pool_size = (2,2)))
        model_i.add(Dropout(0.25))

        model_i.add(Conv2D(128, (3,3), padding = "valid", activation="relu"))
        model_i.add(Conv2D(128, (3,3), padding = "valid", activation="relu"))
        model_i.add(MaxPooling2D(pool_size = (2,2)))
        model_i.add(Dropout(0.3))

        model_i.add(Flatten())

        model_i.add(Dense(1024, activation = "relu"))
        model_i.add(Dropout(0.3))
        model_i.add(Dense(1024, activation = "relu"))
        model_i.add(Dropout(0.3))

        model_i.add(RepeatVector(CONTEXT_LENGTH))

        img_inp = Input(shape = input_shape)
        img_enc = model_i(img_inp)

        model_l = Sequential()
        model_l.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        model_l.add(LSTM(128, return_sequences=True))

        txt_inp = Input(shape = (CONTEXT_LENGTH, output_size))
        txt_enc = model_l(txt_inp)

        decoder = concatenate([img_enc, txt_enc])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation = "softmax")(decoder)

        self.model = Model(inputs = [img_inp, txt_inp], outputs = decoder)

        optimizer = Adam(lr = 0.0001)
        self.model.compile(loss = "categorical_crossentropy", optimizer=optimizer)
    
    def fit(self, images, captions, next_words):
        self.model.fit([images, captions], next_words, shuffle = False, epochs = EPOCHS, batch_size=BATCH_SIZE,verbose = True)
        self.save()

    def fit_generator(self, generator, steps_per_epoch):
        self.model.fit_generator(generator, steps_per_epoch= steps_per_epoch, epochs = EPOCHS, verbose = True)
        self.save()
    
    def predict(self, image, captions):
        return self.model.predict([image, captions], verbose = False)[0]
    
    def predict_batch(self, image, captions):
        return self.model.predict([image, captions], verbose = False)
    