import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './model_garbage/modelo.h5'
pesos_modelo = './model_garbage/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("pred: Aprovechable")
    elif answer == 1:
        print("pred: No Aprovechable")
    elif answer == 2:
        print("pred: Organico biodegradable")
    elif answer == 3:
        print("pred: Residuo especial")
    elif answer == 4:
        print("pred: Residuo peligroso")
    return answer

predict('data/training_garbage/Aprovechable/alpina3.jpg')
