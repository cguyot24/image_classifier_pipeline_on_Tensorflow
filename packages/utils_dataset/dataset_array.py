# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:40:46 2023

@author: cguyot
"""
import os
import random
from shutil import copyfile
from shutil import rmtree

import cv2
import numpy as np
import tensorflow as tf


class DatasetArray:
    def __init__(self):
        pass

    @staticmethod
    def creer_dossiers(output_dir, source, creer_dataset=True):

        """
        creer les dossiers pour le dataset
    
        Parameters
        ----------
        source : string
            dossier contenant le dataset d'images (trié avec comme sous-dossiers les classes)
        output_dir : string
            dossier ou sera enregistrer le dataset.
        creer_dataset : boolean, optional
            The default is True.
    
        Returns
        -------
        to_create : dictionnary
            contient les dossiers qui seront créés
    
        """
        to_create = {
            'train': output_dir + '/train/',
            'validation': output_dir + '/validation/',
            'test': output_dir + '/test/',

        }

        # recupere le nom des classes
        class_names = []
        for directory in os.listdir(source):
            class_names.append(directory)
        print(class_names)

        dossiers = ['train_', 'validation_', 'test_']
        a = 0
        to_create1 = to_create.copy()
        for j in to_create1.values():
            for i in class_names:
                to_create.update({dossiers[a] + i: j + i})
            a = a + 1

        # si dataset deja cree, on le supprime et on le cree
        if creer_dataset:
            print("!!!!créaton des dossiers...")
            if os.path.exists(to_create.get('train')):
                rmtree(to_create.get('train'))
            if os.path.exists(to_create.get('validation')):
                rmtree(to_create.get('validation'))
            if os.path.exists(to_create.get('test')):
                rmtree(to_create.get('test'))

            if not (os.path.exists(output_dir)):
                os.mkdir(output_dir)
            # création des dossiers
            for directory in to_create.values():
                try:
                    os.mkdir(directory)
                    print(directory, 'created')  # iterating through dictionary to make new dirs
                except:
                    print(directory, 'failed')

        return to_create, class_names

    @staticmethod
    def split_data(source, train_path, validation_path, test_path, train_split_size):

        """
        sépare le dataset en train, validation et test
        
        Parameters
        ----------
        source : string
            dossier où est situé le dataset
        train_path : string
            dossier ou est enregistré le dataset train
        validation_path : string
            dossier ou est enregistré le dataset validation
        test_path : string
            dossier ou est enregistré le dataset test
        train_split_size : float (entre 0 et 1)
            pourcentage du dataset pour train (val et test seront séparés en deux avec le reste)
    
        Returns
        -------
        None.
    
        """
        print("!!!!séparation du dataset en train, val et test...")
        all_files = []
        for image in os.listdir(source):
            image_path = os.path.join(image, source)
            if os.path.getsize(image_path):
                all_files.append(image)
            else:
                print('{} has zero size, skipping'.format(image))

        total_files = len(all_files)
        end_train = int(total_files * train_split_size)
        end_validation = int(total_files * (train_split_size + (1 - train_split_size) / 2))

        shuffled = random.sample(all_files,
                                 total_files)  # sample n number of files randomly from the given list of files
        train = shuffled[:end_train]  # slicing from start to split point
        validation = shuffled[end_train:end_validation]
        test = shuffled[end_validation:]

        for image in train:  # copy files from one path to another
            copyfile(os.path.join(source, image), os.path.join(train_path, image))

        for image in validation:
            copyfile(os.path.join(source, image), os.path.join(validation_path, image))

        for image in test:
            copyfile(os.path.join(source, image), os.path.join(test_path, image))
        return

    @staticmethod
    def split_data_loaded(images, labels, train_split_size, val_split_size):
        """
        split data loaded into the memory (RAM or VRAM)

        Parameters
        ----------
        images : np.array
            tableau contenant les images
        labels : np.array
            tableau contenant les labels
        train_split_size : float (between 0 and 1)
            pourcentage du dataset qui servira a l'entrainement
        val_split_size : float (between 0 and 1)
            pourcentage du dataset qui servira a la validation

        Returns
        -------
        x_train, y_train, x_val, y_val, x_test, y_test : images and labels for each dataset

        """
        len_images = len(images)
        end_train = int(len_images * train_split_size)
        end_val = int(len_images * (train_split_size + val_split_size))

        # shuffle the dataset
        loc = np.arange(0, len_images)
        np.random.shuffle(loc)
        images_shuffled = []
        labels_shuffled = []
        for i in range(len_images):
            images_shuffled.append(images[loc[i]])
            labels_shuffled.append(labels[loc[i]])

        images_shuffled = np.array(images_shuffled)
        labels_shuffled = np.array(labels_shuffled)

        # split the dataset
        x_train = images_shuffled[:end_train]
        y_train = labels_shuffled[:end_train]
        x_val = images_shuffled[end_train:end_val]
        y_val = labels_shuffled[end_train:end_val]
        x_test = images_shuffled[end_val:]
        y_test = labels_shuffled[end_val:]

        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def data_load_from_folder(root_path, scale=None):

        """
        importe le dataset depuis un dossier avec pour sous_dossiers les classes
    
        Parameters
        ----------
        root_path : string
            dossier ou est situé le dataset
        scale : (x:int, y:int), optional
            taille dans laquelle les images seront redimensionnées. 
            si None, les images ne sont pas redimensionnées
    
        Returns
        -------
        x, y
            x : dataset d'images ;
            y : labels correspondant
    
        """
        print("!!!!importation des donnees...")

        categories = os.listdir(root_path)
        x = []
        y = []

        num_cat = 0
        for i, cat in enumerate(categories):
            num_cat += 1
            img_path = os.path.join(root_path, cat)
            images = os.listdir(img_path)

            num_img = 0
            for image in images:
                num_img += 1
                img = cv2.imread(os.path.join(img_path, image), cv2.IMREAD_COLOR)
                if img is None:
                    print("une image est vide, c'est l'image ", num_img, " du dossier ", num_cat, ", on skip l'image")
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if scale is not None:
                        img = cv2.resize(img, scale)
                    x.append(img)
                    y.append(i)
        return np.array(x), np.array(y)

    @staticmethod
    def creer_dataset(dataset_dir, images_dir, train_split_size, scale=(256, 256), creer_dataset=True):

        """
        creer et importer le dataset

        Parameters
        ----------
        dataset_dir : string
            dossier où sera créé le dataset (avec en sous-dossiers : train, val, test)
        images_dir : string
            dossier contenant les images (qui sont dans un sous-dossier avec la classe)
        train_split_size : int (entre 0 et 1)
            pourcentage du dataset pour train (val et test seront séparés en deux avec le reste)
        scale : (x:int, y:int)
            taille dans laquelle les images seront redimensionnées
        creer_dataset : boolean, optional
            Si True, créé (ou recréé) un nouvel repartition pour le dataset. The default is True.

        Returns
        -------
        x_train : TYPE
            DESCRIPTION.
        y_train : TYPE
            DESCRIPTION.
        x_val : TYPE
            DESCRIPTION.
        y_val : TYPE
            DESCRIPTION.
        x_test : TYPE
            DESCRIPTION.
        y_test : TYPE
            DESCRIPTION.

        """
        print("!!!!création du dataset")

        # creation des dossiers
        to_create, class_names = DatasetArray.creer_dossiers(dataset_dir, images_dir, creer_dataset)

        if not creer_dataset:
            print("dataset deja créé, on ne fait qu'importer les données")
        else:
            # separation du dataset pour toutes les classes
            for label in class_names:
                DatasetArray.split_data((images_dir + '/' + label), to_create.get('train_' + label),
                                        to_create.get('validation_' + label),
                                        to_create.get('test_' + label), train_split_size)

        # import les 3 datasets et leurs labels
        x_train, y_train = DatasetArray.data_load_from_folder(to_create.get('train'), scale=scale)
        x_val, y_val = DatasetArray.data_load_from_folder(to_create.get('validation'), scale=scale)
        x_test, y_test = DatasetArray.data_load_from_folder(to_create.get('test'), scale=scale)

        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def preprocess_dataset(x, y, batch_size, shuffle=False):

        """
        preprocess le dataset (normalization, conversion en tf.dataset, etc)

        Parameters
        ----------
        x : np.array
            contient le dataset d'images
        y : np.array
            contient le dataset de label
        batch_size : int
            nombre d'images par batch (lots)
        shuffle : boolean, optional
            si True, créer un dataset mélangé. The default is False.

        Returns
        -------
        dataset : tf.Dataset
            Dataset pret à etre rentré dans le réseau de neurones

        """
        print("!!!!preprocess du dataset...")
        # normalisation
        x = x / 255.
        x = np.array(x, dtype=np.float32)
        # ajout de la dim channel pour l'entrainement
        x = np.expand_dims(x, 4)

        # conversion tf.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        # melange du dataset
        if shuffle:
            dataset = dataset.shuffle(len(x))

        # mise sous la forme de paquet
        dataset = (dataset.batch(batch_size)
                   .cache()
                   .prefetch(tf.data.experimental.AUTOTUNE))

        return dataset

    @staticmethod
    def augm_data_np_array(data, label, class_aug, nb_img_augm, transform, class_names):
        """
        augmnente le dataset en créant de nouvelles images

        Parameters
        ----------
        data : np.array
            tableau contenant les images à augmenter
        label : np.array of int 
            tableau contenant les labels associés à data 
        class_aug : list de string
            liste contenant les classes qui sont augmentées.
        nb_img_augm : int
            nombre d'images créées pour une image de data
        transform : Albumentation.Compoe()
            fonction de transformation pour augmenter les images ; contient toutes
            augmentations applicables sur les images
        class_names : list of string
            contient le nom des classes du dataset

        Returns
        -------
        data_aug, label_aug
            Tuple comprenant la data augmentée et les labels augmentée

        """
        aug_data = []
        aug_label = []
        class_aug_int = []
        for i in range(len(class_names)):
            if class_names[i] in class_aug:
                class_aug_int.append(i)

        for i in range(len(data)):
            aug_data.append(data[i])
            aug_label.append(label[i])
            if label[i] in class_aug_int:
                for j in range(nb_img_augm):
                    augms = transform(image=data[i])
                    augm_img = augms['image']
                    aug_data.append(augm_img)
                    aug_label.append(label[i])
        return np.array(aug_data), np.array(aug_label)

    @staticmethod
    def dataset_into_np_array(dataset):
        x = []
        y = []
        for images_batch, labels_batch in dataset.as_numpy_iterator():
            for i in range(images_batch.shape[0]):
                x.append(images_batch[i])
                y.append(labels_batch[i])
        return np.array(x), np.array(y)
