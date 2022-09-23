from keras.models import model_from_json

class AModel:

    def __init__(self, input_shape, output_size, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        
        self.name = ""
        
    def save(self):
        model_json = self.model.to_json()
        
        with open("{}/{}.json".format(self.output_path, self.name), "w") as f:
            f.write(model_json)
        self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))
    
    def load(self, name = ""):
        output_name = self.name if name == "" else name
        
        with open("{}/{}.json".format(self.output_path, output_name), "r") as f:
            model_json = f.read()
        self.model = model_from_json(model_json)
        self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))
        

        
        