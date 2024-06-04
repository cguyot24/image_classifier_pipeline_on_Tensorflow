#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:30:34 2023

@author: corentin
"""
import numpy as np
import tensorflow as tf


class DatasetTF:
    def __int__(self):
        pass

    @staticmethod
    def dataset_into_np_array(dataset):
        x = []
        y = []
        for images_batch, labels_batch in dataset.as_numpy_iterator():
            for i in range(images_batch.shape[0]):
                x.append(images_batch[i])
                y.append(labels_batch[i])
        return np.array(x), np.array(y)

    @staticmethod
    def aug_fn(image, img_size, transforms):
        data = {"image": image}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img / 255.0, tf.float32)
        aug_img = tf.image.resize(aug_img, size=img_size)
        return aug_img

    @staticmethod
    def augm_data(image, label, img_size):
        aug_img = tf.numpy_function(func=DatasetTF.aug_fn, inp=[image, img_size], Tout=tf.float32)
        return aug_img, label

    @staticmethod
    def preprocess_dataset(ds):
        def normalize_element(element, label):
            normalization_layer = tf.keras.layers.Rescaling(1. / 255)
            element = normalization_layer(element)
            return element, label
        
        ds = ds.map(normalize_element)
        ds.prefetch(tf.data.AUTOTUNE)
        return ds
