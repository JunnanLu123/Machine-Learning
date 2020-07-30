"""@title Licensed under the Apache License, Version 2.0 (the "License"); 
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""Setup numpy, tensorflow, and Matplotlib
   Import Tensorflow and other required python
   module. 
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

"""Iris Machine Learning Class"""
class IrisLearningMachine:
    """Class Constructor"""
    def __init__(self):
        self.features=[]
        self.label=[]

    """ Helper function for visualizing the flower data 
        from the dataset batch.The batch size is 32 rows 
        of feature and label pairs.
    """
    def displayFig(self):
        plt.scatter(self.features['petal_length'], self.features['petal_width'], c=self.label, cmap='viridis')
        plt.xlabel('petal_length')
        plt.ylabel('petal_width')
        plt.show()

    """ Helper function to Unpack the dataset to feature 
        and label pairs.
    """
    def unpackData(self, dataset):
        features, labels = next(iter(dataset))
        return features, labels

    """Pack the data into a single array."""
    def pack(self, features, label):
        features = tf.stack(list(features.values()), axis=1)
        return features, label

    """Making trainable dataset from CSV file."""
    def makeDataFromCSV(self, name, batchSize):
        # Column order in CSV file
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        feature_names = column_names[:-1]
        # Each label is associate with class name
        label_name = column_names[-1]
        # Parsing the data into suitable format
        dataset = tf.data.experimental.make_csv_dataset(name, batch_size=batchSize, column_names = column_names, label_name = label_name, num_epochs = 1)
        # Pack feature and label pair into training dataset
        dataset = dataset.map(self.pack)
        return dataset

    """Creating the learning model with Keras."""
    def model_training(self):
        """Parsing the CSV file into suitable format 
        with batch size at 32."""
        dataset = self.makeDataFromCSV("C:/Users/JunnanLu/iris_training.csv", 32)

        """Keep result for plotting."""
        loss_value_list=[]
        model_accuracy_list=[]

        """Creating a linear stack of layers with sequential model."""
        model = tf.keras.Sequential()
        # The first dense hidden layer with 10 nodes.
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,))) 
        # The second dense hidden layer with 10 nodes
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu)) 
        # The final output layer with 3 nodes denoting three flower classes
        model.add(tf.keras.layers.Dense(3)) 

        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # Calculate the loss and return average loss across the examples.
        train = tf.keras.metrics.Mean(name='mean') # Model metric
        # Calculate the gradients to optimize the model
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) 

        # Model Training Process Using Batch of 32 Samples
        # Each epoch is one pass through the dataset
        for epochs in range(100):
            loss_value = tf.keras.metrics.Mean()
            accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            """Training the model through each of the training sample
               Iterate through each example of the feature and label of
               the dataset. 
            """
            for features, label in dataset:
                """GradientTape automatically watch trainable variable,
                   the resource will be released when GradientTape.gradient()
                   method is called. Using gradient to calculate and optimize 
                   gradients of the model.
                """
                with tf.GradientTape() as Tape:
                    # Set training to true if layers have different behaviours
                   prediction = model(features, training=True)
                   loss = loss_object(label, prediction)
                grads = Tape.gradient(loss, model.trainable_variables)
                # Optimizer implemented the Stochastic Gradient Decent method.
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # Tracking the stats of the loss and accuracy value for visualization
                loss_value.update_state(loss)
                accuracy.update_state(label, model(features, training=True))
                
            # Append the loss value and accuracy result to the lists
            loss_value_list.append(loss_value.result())
            model_accuracy_list.append(accuracy.result())

        """Visualize the loss in time elapsed fashion.
           TensorBoard can also help to visualize the training metrics.
           We are using matplotlib to create basic metrics chart.
        """
        fig1, axes = plt.subplots(2, sharex=True, figsize=(12,8))
        fig1.suptitle("Train Metrics")
        axes[0].set_ylabel("LOSS", fontsize=14)
        axes[0].set_xlabel("EPOCHS", fontsize=14)
        axes[0].plot(loss_value_list)

        axes[1].set_ylabel("ACCURACY", fontsize=14)
        axes[1].set_xlabel("EPOCHS", fontsize=14)
        axes[1].plot(model_accuracy_list)
        plt.show()
       

        """Model evaluation using test data
           Using single epoch to test the model.
           Dataset is downloaded and store in local machine.
           Parsing the test data.
        """
        dataset = self.makeDataFromCSV("C:/Users/JunnanLu/iris_test.csv", 100)

        # Tracking model performace
        accuracy = tf.keras.metrics.Accuracy()
        test_accuracy_list=[]

        """We iterate every sample from the dataset and compare to the
           model's prediction against the actual to measure model's 
           accuracy across the entire test dataset.
        """
        for features, label in dataset:
            # Set training to false if layers have different behaviours
            prediction = model(features, training=False)
            prediction = tf.nn.softmax(prediction, axis=1,name='softmax')
            prediction = tf.argmax(prediction, axis=1, output_type=tf.dtypes.int32)
            print (prediction)
            accuracy.update_state(label, prediction)
            print (accuracy.result())
            test_accuracy_list.append(accuracy.result())
        print ("TEST Accuracy {:1.2%}".format(accuracy.result()))

        """Extracting features from the dataset. In here,
           we take 10 samples from the features data
           letting our model to predict.
        """
        predict_data = self.makeDataFromCSV("C:/Users/JunnanLu/iris1.csv",10)
        features, labels = self.unpackData(predict_data)

        # List of three class names.
        class_name = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

        """We trained the model. The model is good if not perfect.
           Now we are go to us the trained model to predict unlabeled data.
           The prediction stores model result.
        """
        prediction = model(features, training=False)

        # Iterating through the model result.
        for index, logits in enumerate(prediction):
            # Record the class index with maximum probability.
            class_index = tf.argmax(logits).numpy()
            # Calculate the class softmax probability.
            probability = tf.nn.softmax(logits)[class_index] 
            # Display the index, class name, and predicted class probability
            name = class_name[class_index]
            print ("index {}, class: {}, probability {:1.2%}".format(index,name,probability))

# Unit test
if __name__=='__main__':
    model = IrisLearningMachine()
    model.model_training()
