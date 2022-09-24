import tensorflow as tf
import sys
# session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from Generator import *
from px2cd import *

def run(input_path, output_path, memory_intensive, pretrained_model):
    np.random.seed(11)
    
    dataset = Dataset()
    dataset.loadData(input_path, True)
    dataset.saveMetadata(output_path)
    dataset.voc.save(output_path)

    if not memory_intensive:
        dataset.arrayConvert()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images),len(dataset.partial_sequences),len(dataset.next_words))
        print(dataset.input_images.shape,dataset.partial_sequences.shape,dataset.next_words.shape)
    
    else:
        gui_paths, img_paths = Dataset.pathLoader(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size/BATCH_SIZE

        voc = Vocabulary()
        voc.dataRead(output_path)
        generator = Generator.data_generator(voc,gui_paths,img_paths,BATCH_SIZE,True)
    
    model = AutoCode(input_shape,output_size,output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)
    
    if not memory_intensive:
        model.fit(dataset.input_images,dataset.partial_sequences,dataset.next_words)
    
    else:
        model.fit_generator(generator,steps_per_epoch)


if __name__  == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]

        use_gen = False if len(argv) < 3 else True if int(argv[2]) == 1 else False
        pretrained_weights = None if len(argv) < 4 else argv[3]

    run(input_path,output_path,use_gen,pretrained_weights)
