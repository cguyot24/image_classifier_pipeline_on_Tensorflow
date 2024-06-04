# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:26:14 2023

@author: cguyot



Ce script permet de classifier des images en utilisant le deep learning.
Régler les paramètres dans la partie PARAMETRES et exécuter le script.

La structure du dossier des images importées doit être la suivante :
\
    -images\
        -nom_classe_1\
            -img1
            -img2
            ...
        -nom_classe_2\
            -img1
            -img2
        ...
\

"""
import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import yaml
import cv2

from packages.utils_dataset import DatasetArray
from packages.utils_dataset import DatasetTF
from packages.display import Display
from packages.display import InConsole
from packages.model import Model
from packages.config_settings import ConfigSettings

    
class Main:
    def __init__(self):
        # read the config file
        with open('config.yml', 'r') as file:
            self.data_yml = yaml.load(file, Loader=yaml.FullLoader)

        # on charge les paramètres
        self.p = ConfigSettings(self.data_yml)

        # save dataset_path for training
        self.dataset_path = None

        # save training_folder for evaluation
        self.training_folder = None
          
        # to save images and terminal output to a folder
        self.display = Display()
        self.inconsole = InConsole()
        
    @staticmethod
    def save_yml(data_yml, output_path):
        """
        save a yml file to a destination folder

        """
        with open(output_path, 'w') as file:
            yaml.dump(data_yml, file)
        
    def load_image_and_process(self, img_name, label, input_folder, output_folder, scale, augment=False):
        """
        charge une image, augmente et process l'image, et renvoie un tableau contenant les images générées

        Parameters
        ----------
        img_name : string
            DESCRIPTION.
        label : string
            DESCRIPTION.
        input_folder : string
            path to the folder where the image is located
        output_folder : string
            path to the folder to save the image
        scale : tuple of int
            DESCRIPTION.
        augment : boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        images, labels : tableaux contenant les images générées avec leur label

        """
        # load image
        img = cv2.imread(str(input_folder + img_name), cv2.IMREAD_COLOR)

        if img is None:
            return None
        else:
            # rescale before augmentation to accelerate processus
            if self.p.rescale_before_augmentation is not None:
                img = cv2.resize(img, self.p.rba_scale)

            # augmentation
            if augment:
                label_int = self.p.class_names.index(label)
                images, labels = DatasetArray.augm_data_np_array([img], [label_int],
                                                                 self.p.class_names,
                                                                 self.p.nb_images_aug,
                                                                 self.p.transform,
                                                                 self.p.class_names)
            else:
                images = [img]

            # rescale images and save
            for i, img in enumerate(images):
                img = cv2.resize(img, scale)  # rescale
                name, ext = os.path.splitext(img_name)
                filename = output_folder + '/' + label + '/' + name + '-' + str(i) + ext
                cv2.imwrite(filename, img)  # save

    def create_dataset(self):
        """
        créer le dataset en fonction de parametres.py et le sauvegarde dans un dossier

        Returns
        -------
        None.

        """
        print("!!!! DEBUT CREATION DATASET !!!!")
        # creation des dossiers
        to_create, class_names_check = DatasetArray.creer_dossiers(self.p.output_cd, self.p.input_cd)

        # test erreur class_names
        if self.p.class_names != class_names_check:
            print("ERROR : le nom des dossiers ne correspond pas à la variable class_names")
            print(self.p.class_names)
            print(class_names_check)
            sys.exit(1)

        # creation du dataset par class
        for label in self.p.class_names:
            print(f'!!!!création du dataset sur la classe {label}')
            # recuperation du nom des images
            all_images = os.listdir(f'{self.p.input_cd}/{label}')
            len_images = len(all_images)

            # melange du dataset
            np.random.shuffle(all_images)

            # calcul du nombre d'images dans train, val et test pour ce label/classe
            end_train = int(len_images * self.p.train_split_size)
            end_val = int(len_images * (self.p.train_split_size + self.p.val_split_size))

            input_folder = self.p.input_cd + '/' + label + '/'

            # ajout dans train
            num_img = 1
            print("!!!!creation de train....")
            output_folder = self.p.output_cd + '/train/'
            for img_name in all_images[:end_train]:
                self.load_image_and_process(img_name, label, input_folder, output_folder, self.p.scale, self.p.do_augmentation)
                print(f'image {num_img}/{len_images}')
                num_img+=1

            # ajout dans validation
            print("!!!!creation de validation....")
            output_folder = self.p.output_cd + '/validation/'
            for img_name in all_images[end_train:end_val]:
                self.load_image_and_process(img_name, label, input_folder, output_folder, self.p.scale, False)
                print(f'image {num_img}/{len_images}')
                num_img+=1

            # ajout dans test
            print("!!!!creation de test....")
            output_folder = self.p.output_cd + '/test/'
            for img_name in all_images[end_val:]:
                self.load_image_and_process(img_name, label, input_folder, output_folder, self.p.scale, False)
                print(f'image {num_img}/{len_images}')
                num_img+=1

            # save dataset_path for training
            self.dataset_path = self.p.output_cd
        print("!!!!CREATION DATASET REUSSI!!!!")

    def train_model(self):
        """
        entraine le modèle et sauvegarde les résultats dans un dossier

        Returns
        -------
        None.

        """
        print("!!!! DEBUT TRAIN MODEL !!!!")
        # si creer_dataset et train_network sont appeles en meme temps, on recupere le chemin du dataset
        if self.dataset_path is not None:
            dataset_dir = self.dataset_path
        else:
            dataset_dir = self.p.input_tm

        # chargement du dataset train
        train_dir = dataset_dir + '/train/'
        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                image_size=self.p.scale,
                batch_size=self.p.batch_size,
                class_names=self.p.class_names,
            )
        except ValueError:
            print("!!!!ERROR : probablement un mauvais chemin pour le dataset")
            raise

        val_dir = dataset_dir + '/validation/'
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=self.p.scale,
            batch_size=self.p.batch_size,
            class_names=self.p.class_names,
        )

        train_ds = DatasetTF.preprocess_dataset(train_ds)
        val_ds = DatasetTF.preprocess_dataset(val_ds)
         
        print("!!!!model choisi : ", self.p.model_type)
        train_name = 'training_' + datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss') + '_' + self.p.model_type
        
        # création du modèle
        print("!!!!creation du modele...")
        shape = (self.p.scale[0], self.p.scale[1], 3)
        model = Model.create_model(self.p.model_type, shape, self.p.learning_rate, self.p.class_names, self.p.imagenet)
       
        # affichage du modele
        # model.summary()

        # creation du dossier d'entrainement
        print("!!!!nom du dossier d'entrainement : " + train_name)
        train_folder = os.path.join(self.p.output_tm, train_name)
        os.makedirs(train_folder)

        best_epoch_filename = train_folder + '/best_epoch/best_epoch'
        epoch_save_filename = train_folder + '/epoch_{epoch:02d}/epoch_{epoch:02d}'

        # parametres de sauvegardes du modele et d'arret
        callbacks = [
            # save best epoch
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_epoch_filename,
                monitor=self.p.monitor,
                save_best_only=True,
                save_weights_only=True),

            # save epoch every X epochs
            tf.keras.callbacks.ModelCheckpoint(
                filepath=epoch_save_filename,
                monitor=self.p.monitor,
                save_freq='epoch',
                period=self.p.save_parametres_step,
                save_weights_only=True),

            # stop the model if it doesn't learn anymore (after "patience" epoch)
            tf.keras.callbacks.EarlyStopping(
                monitor=self.p.monitor,
                patience=self.p.patience,
                verbose=1,
                restore_best_weights=False)]

        # charger les poids du modele a partir d'un fichier
        if self.p.load_weights:
            print("!!!!restoration du modele...")
            model.load_weights(self.p.weights_file_tm)

        # créer un log tensorboard pour voir les résultats
        log_dir = train_folder + '/tensorboard_logs/'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback) # on ajoute tensorboard au callbacks pour l'affichage
        bash_command = "tensorboard --logdir=" + os.getcwd() + '/' + log_dir
        print("!!!!COMMANDE POUR LANCER TENSORBOARD : " + bash_command)

        # usefull if evaluate is called next
        self.training_folder = train_folder

        # entrainement du modele
        print("!!!!entrainement du modele...")
        model_history = model.fit(train_ds, validation_data=val_ds, epochs=self.p.epochs,
                                  callbacks=callbacks)

        # affichage des courbes de loss (pertes) et accuracy
        print("!!!!entrainement terminé, affichage courbes loss et accuracy...")
        self.display.aff_loss(model_history)
        self.display.aff_accuracy(model_history)
        
        # sauvegarde des parametres de config
        output_config_path = f'{train_folder}/config_saved.yml'
        Main.save_yml(self.data_yml, output_config_path)
        
        print("!!!!ENTRAINEMENT REUSSI!!!!")
    
    @staticmethod
    def load_model(model, training_folder, weights_folder, load_best_epoch, epoch_number):
        """
        load model using keys in the .yml to load the weights of the model, usefull for evaluate and export

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        training_folder : TYPE
            DESCRIPTION.
        weights_folder : TYPE
            DESCRIPTION.
        load_best_epoch : TYPE
            DESCRIPTION.
        epoch_number : TYPE
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION.
        training_folder : TYPE
            DESCRIPTION.

        """
        print("!!!!on charge les poids du modele..")
        # si l'entrainement à été fait juste avant, on utilise le fichier d'entrainement juste créé
        # sinon, on utilise weights_folder
        if training_folder is None:
            training_folder = weights_folder

        # meilleure epoch ou epoch choisie qui sera chargée
        if load_best_epoch:
            weights_file = f'{training_folder}/best_epoch/best_epoch'
        else:
            weights_file = f'{training_folder}/epoch_{epoch_number}/epoch_{epoch_number}'
        print("!!!!fichier charger : " + weights_file)

        # expect.partial() pour supprimer les warnings a l'affichage
        try:
            model.load_weights(weights_file).expect_partial()
        except tf.errors.NotFoundError:
            print("!!!!ERROR : filename_load_cpkt ne contient pas le nom d'un fichier correct pour charger les poids")
            raise
        except ValueError as e:
            error_message = str(e)
            if "incompatible tensor with shape" in error_message:
                print("!!!!ERROR : le modèle choisi ne correspond pas au modèle chargé dans le fichier de poids")
                raise
            else:
                raise
        return model, training_folder
    
    def evaluate_model(self):
        """
        evalue le modele sur le dataset de test

        Returns
        -------
        None.

        """
        print("!!!! DEBUT EVALUATE MODEL !!!!")
        # si creer_dataset et train_model et evaluate_model sont appeles en meme temps, on recupere le chemin du dataset
        if self.dataset_path is not None:
            test_dir = self.dataset_path+'/test/'
        else:
            test_dir = self.p.dataset_test_directory

        # chargement du dataset test
        try:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                image_size=self.p.scale,
                batch_size=self.p.batch_size,
                class_names=self.p.class_names,
                shuffle=False,
            )
        except ValueError:
            print("!!!!ERROR : probablement un mauvais chemin pour le dataset")
            raise
        
        # conversion en np_array pour les affichages
        x_test, y_test = DatasetTF.dataset_into_np_array(test_ds)
        x_test = x_test.astype(int)
        y_test = y_test.astype(int)
        
        # preprocess du dataset
        test_ds = DatasetTF.preprocess_dataset(test_ds)
       
        # creation et chargement du modele
        shape = (self.p.scale[0], self.p.scale[1], 3)
        model = Model.create_model(self.p.model_type, shape, self.p.learning_rate, self.p.class_names, self.p.imagenet)
        
        # load the model
        model, self.training_folder = Main.load_model(model, self.training_folder,
                                                            self.p.weights_folder_em,
                                                            self.p.load_best_epoch_em,
                                                            self.p.epoch_number_em)
        
        # afficher la structure du model
        # model.summary()

        # tester le modele sur le dataset de test
        model.evaluate(test_ds)

        # faire des predictions sur le dataset de test
        #probability_model = tf.keras.Sequential([model,
        #                                         tf.keras.layers.Softmax()])
        predictions = model.predict(test_ds)

        # calcul de la matrice de confusion
        print("!!!!affichage matrice de confusion...")
        self.display.aff_mat_confusion(y_test, predictions, self.p.class_names)
        print(x_test[0].shape)
        # affichage des valeurs de tests
        if len(self.p.class_names) == 2:
            self.inconsole.aff_metrics_2_classes(y_test, predictions)

        # test du temps d'inférence (temps pour traverser le modèle 1 fois)
        self.inconsole.test_inference(x_test[0], y_test[0], model=model)
        
        # calcul des predictions et heatmaps
        print('!!!!affichage des predictions....')
        if self.p.display_all_predictions:
            nb_preds = len(y_test)
        else:
            nb_preds = min(self.p.nb_pred, len(y_test))

        for i in range(nb_preds):
            print(f'calcul de la heatmap {i+1}/{nb_preds}')
            self.display.aff_heatmap(x_test[i], y_test[i], predictions[i], model, self.p.class_names)
            
        # calcul des mauvaises predictions
        if self.p.display_bad_results:
            print("!!!!affichage mauvaises predictions+heatmap...")
            for i in range(len(y_test)):
                if np.argmax(predictions[i]) != y_test[i]:
                    self.display.aff_heatmap(x_test[i], y_test[i], predictions[i], model, self.p.class_names)
        
        print("!!!!EVALUATION TERMINEE!!!!")

    def export_model(self):
        print("!!!!DEBUT EXPORT MODEL!!!!")
        # creation et chargement du modele
        shape = (self.p.scale[0], self.p.scale[1], 3)
        model = Model.create_model(self.p.model_type, shape, self.p.learning_rate, self.p.class_names, self.p.imagenet)
        
        # load the model
        model, export_folder = Main.load_model(model, self.training_folder,
                                self.p.weights_folder_ex,
                                self.p.load_best_epoch_ex,
                                self.p.epoch_number_ex)
        
        # on définit le chemin où sera exporté le model 
        export_folder = export_folder+'/export_model/'
          
        # sauvegarde du modele au format standard
        if "export_default_model" in self.p.keys_export:
            model.save(export_folder+'/default_model')
            
        # sauvegarde du modele au format TensorflowLite
        if "export_lite_model" in self.p.keys_export:
            # Convert the model to TensorFlow Lite format
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save the TensorFlow Lite model to a file
            with open(f'{export_folder}/tflite_model.tflite', 'wb') as f:
                f.write(tflite_model)
        
        print("!!!!EXPORT TERMINE!!!!")


m = Main()
if 'create_dataset' in m.p.action:
    m.create_dataset()
if 'train_model' in m.p.action:
    m.train_model()
if 'evaluate_model' in m.p.action:
    m.evaluate_model()
if 'export_model' in m.p.action:
    m.export_model()
    
# affichage et enregistrement de toutes les images et de l'output terminal générés
if 'train_model' in m.p.action or 'evaluate_model' in m.p.action:
  m.display.display_images(m.p.keys_images, m.training_folder+'/evaluation_results/')
  m.inconsole.save_output_to_folder(m.training_folder+'/evaluation_results/evaluation_output.txt')
