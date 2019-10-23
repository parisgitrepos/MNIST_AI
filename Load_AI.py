from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
model = keras.models.load_model('MNIST_Number_AI.h5')

class MNIST_AI:
    def __init__(self, x, model = model):
        self.model = model
        self.x = x/255
        self.x = x.reshape(1, 28, 28, 1)
        self.x_for_plt = x
        
    def matplotlib_plt(self):
        plt.imshow(self.x_for_plt)
        
    def model_predict(self):
        prediction = model.predict(self.x)[0]
        prediction = prediction.tolist()
        prediction = prediction.index(max(prediction))
        print(prediction)

a = MNIST_AI(x = test_x[23])
a.matplotlib_plt()
a.model_predict()