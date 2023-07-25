import keras
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# GRADED FUNCTION: image_generator
def image_generator(path):
    ### START CODE HERE

    # Instantiate the ImageDataGenerator class.
    # Remember to set the rescale argument.
    train_datagen = ImageDataGenerator(rescale=1/255)

    # Specify the method to load images from a directory and pass in the appropriate arguments:
    # - directory: should be a relative path to the directory containing the data
    # - targe_size: set this equal to the resolution of each image (excluding the color dimension)
    # - batch_size: number of images the generator yields when asked for a next batch. Set this to 10.
    # - class_mode: How the labels are represented. Should be one of "binary", "categorical" or "sparse".
    #               Pick the one that better suits here given that the labels are going to be 1D binary labels.
    train_generator = train_datagen.flow_from_directory(directory=path,
                                                        target_size=(150, 150),
                                                        batch_size=10,
                                                        class_mode='binary')
    ### END CODE HERE

    return train_generator


def create_conv_model():
    model=tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(64,(3,3),input_shape=(150,150,3)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)

        ]
    )
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=tf.keras.losses.binary_crossentropy,metrics=['accuracy'])
    return model

class MyCallback (tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Halts the training after reaching 60 percent accuracy

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    # Check accuracy
    if(logs.get('accuracy') >0.995):

      # Stop if threshold is met
      print("\nAcc is higher than 0.995 so cancelling training!")
      self.model.stop_training = True

def print_loss(history):
    loss = history.history['loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)

    plt.show()

if __name__=='__main__':
    gen=image_generator('C:\Progetti\Personali\MachineLearning\Basic\Cousera\HappyorSad\data')
    model=create_conv_model()
    callback=MyCallback()
    history=model.fit(x=gen, epochs=500,callbacks=[callback])
    print_loss(history)
