import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import cv2
import tensorflow as tf
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Display:
    def __init__(self):
        self.images_list = []  # list that contains generated images

    @staticmethod
    def fig_to_array(fig):
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert the rendered image to a NumPy array
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # Close the figure to release resources
        plt.close()
        return image_array

    def add_image_to_list(self, fig):
        img = self.fig_to_array(fig)
        self.images_list.append(img)
        return

    @staticmethod
    def aff_dataset(x: np.array, y: np.array, num_first_img: int, nb_images: int, class_names: list) -> None:
        """
        affiche une partie des images du dataset

        Parameters
        ----------
        x : np.array
            dataset des images
        y : np.array
            dataset des labels associé au dataset d'images
        num_first_img : int
            numéro de la premiere image affichée
        nb_images : int (max 9)
            nombre d'images affichées (max 9)
        class_names : list of string
            nom des classes contenues dans le dataset
        Returns
        -------
        None.

        """
        plt.figure(figsize=(nb_images, nb_images))
        if nb_images > 9:
            nb_images = 9
        for i in range(nb_images):
            plt.subplot(3, 3, i + 1)
            plt.imshow(x[num_first_img + i])
            plt.title(class_names[y[num_first_img + i]])
            plt.axis("off")
        return

    def aff_loss(self, model_history):
        """
        affiche les courbes de loss de train et val après l'entrainement'

        Parameters
        ----------
        model_history : tf. ?
            variable contenant l'historique de l'apprentissage (en sortie de la fonction model.train)

        Returns
        -------
        list of images generated

        """
        fig = plt.figure(figsize=(15, 10))
        loss_train_curve = model_history.history["loss"]
        loss_val_curve = model_history.history["val_loss"]
        plt.plot(loss_train_curve, label="Train")
        plt.plot(loss_val_curve, label="Validation")
        plt.legend(loc='upper right')
        plt.title("Loss")
        self.add_image_to_list(fig)
        plt.close()
        return

    def aff_accuracy(self, model_history):
        """
        affiche les courbes de précision de train et val après l'entrainement
        précision : pourcentage d'images bien prédites

        Parameters
        ----------
        model_history : tf. ?
            variable contenant l'historique de l'apprentissage (en sortie de la fonction model.train)

        Returns
        -------
        None.

        """
        fig = plt.figure(figsize=(15, 10))
        plt.plot(model_history.history["accuracy"])
        plt.plot(model_history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        self.add_image_to_list(fig)
        plt.close()
        return

    @staticmethod
    def plot_image(predictions_array, true_label, img, shape, class_names):
        """
        fonction spécifique à la fonction afficher_predictions
        """
        img = img.reshape(shape)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)
        return

    @staticmethod
    def plot_value_array(predictions_array, true_label, class_names):
        """
        fonction spécifique à afficher_predictions
        """
        plt.grid(False)
        plt.xticks(ticks=range(len(class_names)), labels=class_names)
        plt.yticks([])
        this_plot = plt.bar(class_names, predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        this_plot[predicted_label].set_color('red')
        this_plot[true_label].set_color('blue')
        return

    def aff_predictions(self, x, y, predictions, nb_predictions, class_names):
        """
        affiche un certain nombre de prédictions

        Parameters
        ----------
        x : np.array
            dataset d'images
        y : np.array
            dataset de label associé à x
        predictions : np.array ?
            tableau contenant les predictions (sortie de model.evaluate)
        nb_predictions : int
            nombre de prédictions affichées
        class_names : tab de string
            tableau contenant le nombre de classes
        Returns
        -------
        None.

        """
        shape = x[0].shape
        for i in range(nb_predictions):
            fig = plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            Display.plot_image(predictions[i], y[i], x[i], shape, class_names)
            plt.subplot(1, 2, 2)
            Display.plot_value_array(predictions[i], y[i], class_names)
            self.add_image_to_list(fig)
            plt.close()
        return

    def aff_mat_confusion(self, y_test, predictions, class_names):
        """
        affiche la matrice de confusion

        Parameters
        ----------
        y_test : array de label
            vrai labels.
        predictions : array de label
            labels predits.
        class_names : array de string
            DESCRIPTION.

        Returns
        -------
        None.

        """
        mat = metrics.confusion_matrix(y_test, np.argmax(predictions, axis=1))
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(mat, cmap='YlGn')

        # Affichage des valeurs de la matrice
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, str(mat[i, j]), ha='center', va='center')

        # Configuration des axes
        plt.xticks(ticks=range(len(class_names)), labels=class_names)
        plt.yticks(ticks=range(len(class_names)), labels=class_names)
        plt.ylabel('Vrai labels')
        plt.xlabel('Predictions')
        plt.title("Matrice de confusion")
        self.add_image_to_list(fig)
        plt.close()
        return
    
    @staticmethod
    def get_heatmap(model, layer_name, img_array):
        img_array = img_array/255.
        img_array = np.expand_dims(img_array, axis=0)
        
        gradModel = tf.keras.Model(
            inputs = [model.inputs],
            outputs = [model.get_layer(layer_name).output, model.output]
            )
        inputs = tf.cast(img_array, tf.float32)
        print(model.predict(inputs))
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # forward propagate the image through the gradient model, and grab the loss
            # associated with the specific class index
            (convOutputs, predictions) = gradModel(inputs)
            print(predictions)
            loss = predictions[:, tf.argmax(predictions[0])]
            print(loss)
            
        # use automatic differentiation to compue the gradient
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # drop the batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        heatmap= cam.numpy()
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        eps = 1e-15
        plt.imshow(heatmap)
        plt.show()
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        
        (w, h) = (img_array.shape[2], img_array.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        return heatmap
    
    # V2 of get_heatmap
    def make_gradcam_heatmap(model, list_conv_layers, img_array, max_layers = 10, pred_index=None):
        """
        calcul la heatmap en fonction du model, d'une image et d'une liste de couche de convolution

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        list_conv_layers : TYPE
            DESCRIPTION.
        img_array : TYPE
            DESCRIPTION.
        pred_index : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        mean_heatmap : TYPE
            DESCRIPTION.

        """
        # normalisation de l'image pour afficher la heatmap et ajout de la dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.cast(img_array/255., tf.float32)
        
        # si il y a moins de 2*max_layers couches de convolution, on utilise moins de couches 
        num_layers = len(list_conv_layers)
        if num_layers < (2*max_layers):
          max_layers = int(num_layers/2.)
        
        # First, we create a model that maps the input image to the activations
        # of the list of conv layers as well as the output predictions
        output_layers = []
        for layer in reversed(list_conv_layers):
            output_layers.append(model.get_layer(layer).output)
        
        grad_model = tf.keras.models.Model(
            model.inputs, [output_layers, model.output]
        )
    
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        with tf.GradientTape(persistent=True) as tape:
            list_layers_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        i = 0
        list_heatmaps = []
        for layer_output in list_layers_output:
            if i < max_layers:
                i += 1
            else:
                break
            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            grads = tape.gradient(class_channel, layer_output)
            
            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the heatmap class activation
            layer_output = layer_output[0]
            heatmap = layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            
            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = np.array(heatmap)
            if not (heatmap.shape == ()):
                (w, h) = (img_array.shape[2], img_array.shape[1])
                heatmap = cv2.resize(np.array(heatmap), (w, h))
                list_heatmaps.append(heatmap)
        
        del tape
        
        # compute mean of heatmaps
        list_heatmaps = np.array(list_heatmaps)
        mean_heatmap = np.mean(list_heatmaps, axis= 0)

        # normalize and color mean_heatmap
        # we erase negative value due to relu as activation function
        mean_heatmap = tf.maximum(mean_heatmap, 0) / tf.math.reduce_max(mean_heatmap)
    
        mean_heatmap = cv2.applyColorMap(np.uint8(255 * mean_heatmap), cv2.COLORMAP_JET)
        mean_heatmap = cv2.cvtColor(mean_heatmap, cv2.COLOR_BGR2RGB)
        return mean_heatmap
    
    def aff_heatmap(self, img, label, prediction, model, class_names, alpha=0.5, fontsize=10):
        """
        affiche la heatmap d'une image avec la prediction associée

        Parameters
        ----------
        img : np.array
            une image
        label : int
            label associé à x (entier qui correspond à la place dans le tab class_names)
        prediction : ?
            prediction associé à x
        model : tf.Model
            model utilisé
        alpha : float, optional
            transparence de la heatmap lors de la superposition avec l'image. The default is 0.5.
        fontsize : int, optional
            Taille du titre. The default is 10.
        class_names : tab de string, optional
            tableau contenant les classes. The default is class_names.

        Returns
        -------
        None.

        """

        # recuperation du nom de la derniere couche conv2D
        list_conv_layer = list(filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), model.layers))
        list_conv_layer_name = []
        for layer in list_conv_layer:
            list_conv_layer_name.append(layer.name)
            
        # calcul de la heatmap
        heatmap = Display.make_gradcam_heatmap(model, list_conv_layer_name, img)
        
        # affichage de la prediction
        self.aff_predictions([img], [label], [prediction], 1, class_names)

        # affichage de la heatmap
        fig = plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(heatmap)
        plt.axis('off')
        plt.title("heatmap", fontsize=fontsize)

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(heatmap, alpha=alpha)
        plt.axis('off')
        plt.title("image +heatmap", fontsize=fontsize)
        # plt.show()
        plt.close()

        # concatenate the two images of pred and heatmaps into a single one
        img_pred = self.images_list[-1]
        img_heatmap = self.fig_to_array(fig)
        concatenated_image = np.vstack((img_pred, img_heatmap))
        self.images_list[-1] = concatenated_image
        return
    
    def display_images(self, keys, folder):
        # plot images
        if "display_images" in keys:
            for i, image in enumerate(self.images_list):
                plt.imshow(image)
                plt.show()

        # save images
        if "save_images_in_a_folder" in keys:
            # creation of the folder to save images
            if not (os.path.exists(folder)):
                os.mkdir(folder)

            # save images
            for i, image in enumerate(self.images_list):
                image_path = f'{folder}/img_{i}.png'
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, image)
