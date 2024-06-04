import tensorflow as tf
import tensorflow.keras.layers as layers

class Model:
    def __init__(self):
        pass

    """
    @staticmethod
    def augm_model(input_shape):
        #augmentation model
        augm_model = tf.keras.Sequential([
                #layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomBrightness(0.2, value_range=(0.0,1.0)),
                layers.RandomContrast(0.2),
            ])
        return augm_model
    """

    @staticmethod
    def small_model(input_shape, class_names):
        
        size = 1 # taille du model (1 ou 2 ou 3)
        nb_neurones = 32 * size
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(nb_neurones, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(nb_neurones, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(nb_neurones, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4 * nb_neurones, activation='relu'),
            layers.Dropout(0.05),
            layers.Dense(len(class_names), activation='softmax')
        ])
        return model

    @staticmethod
    def alex_net(input_shape, class_names):
        model = tf.keras.models.Sequential([
            layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(4, 4), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(class_names), activation='softmax')
        ])
        return model

    # pour les models pre entraines
    @staticmethod
    def unfreeze_model(model, nb_layers):
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in model.layers[-nb_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        return model

    @staticmethod
    def resnet_50(input_shape, class_names, load_weights):
        model_0 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=load_weights,
                                                          input_shape=input_shape)
        
        for layer in model_0.layers:
            layer.trainable = False

        # rend entrainable 10 couches
        model_0 = Model.unfreeze_model(model_0, 10)
        
        x = model_0.output
        x = layers.Flatten()(x)
        x = layers.Dense(len(class_names), activation='softmax')(x)
        model = tf.keras.Model(inputs=model_0.input, outputs=x)
        return model

    @staticmethod
    def vgg16(input_shape, class_names):
        model_0 = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        
        model_0.trainable = False
        # on reinitialise la 5 partie cnn du reseau pour entrainer uniquement celle la
        x = model_0.layers[-5].output

        x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(len(class_names), activation='softmax')(x)

        model = tf.keras.Model(inputs=model_0.input, outputs=x)
        return model

    @staticmethod
    def efficientnet(input_shape, class_names, model_name, load_weights):
        if model_name == "efficientnet_xs":
            model_0 = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False,
                                                                        weights=load_weights,
                                                                        input_shape=input_shape,
                                                                        pooling="avg"
                                                                        )
        elif model_name == "efficientnet_s":
            model_0 = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False,
                                                                        weights=load_weights,
                                                                        input_shape=input_shape,
                                                                        pooling="avg"
                                                                        )
        elif model_name == "efficientnet_m":
            model_0 = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False,
                                                                        weights=load_weights,
                                                                        input_shape=input_shape,
                                                                        pooling="avg"
                                                                        )
        elif model_name == "efficientnet_l":
            model_0 = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False,
                                                                        weights=load_weights,
                                                                        input_shape=input_shape,
                                                                        pooling="avg"
                                                                        )
        else:
            model_0 = None
            print("Error : wrong model_name, end of program")

        outputs = layers.Dense(len(class_names), activation="softmax")(model_0.output)
        model = tf.keras.Model(inputs=model_0.input, outputs=outputs)
        return model
      
    @staticmethod
    def densenet121(input_shape, class_names, load_weights):
      model_0 =  tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                            weights=load_weights,
                                                            input_shape=input_shape,
                                                            pooling="avg"
                                                            )

      outputs = layers.Dense(len(class_names), activation="softmax")(model_0.output)
      
      model = tf.keras.Model(inputs=model_0.input, outputs=outputs)
      return model
  
    @staticmethod
    def mobilenetV3Small(input_shape, class_names, load_weights):
      model_0 =  tf.keras.applications.MobileNetV3Small(include_top=False,
                                                            weights=load_weights,
                                                            input_shape=input_shape,
                                                            pooling="avg"
                                                            )
      outputs = layers.Dense(len(class_names), activation="softmax")(model_0.output)
      model = tf.keras.Model(inputs=model_0.input, outputs=outputs)
      return model
    
    @staticmethod
    def mobilenetV3Large(input_shape, class_names, load_weights):
      model_0 =  tf.keras.applications.MobileNetV3Large(include_top=False,
                                                            weights=load_weights,
                                                            input_shape=input_shape,
                                                            pooling="avg"
                                                            )
      outputs = layers.Dense(len(class_names), activation="softmax")(model_0.output)
      model = tf.keras.Model(inputs=model_0.input, outputs=outputs)
      return model
    
    @staticmethod
    def mobilenetV2(input_shape, class_names, load_weights):
      model_0 =  tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                            weights=load_weights,
                                                            input_shape=input_shape,
                                                            pooling="avg"
                                                            )
      outputs = layers.Dense(len(class_names), activation="softmax")(model_0.output)
      model = tf.keras.Model(inputs=model_0.input, outputs=outputs)
      return model
    
    @staticmethod
    def preprocess_model(input_shape):
        # preprocess model
        preprocess_model = tf.keras.Sequential([
            layers.Resizing(input_shape[0], input_shape[1]),
            layers.Rescaling(1. / 255),
        ])
        return preprocess_model
      

    @staticmethod
    def create_model(model_name, input_shape, learning_rate, class_names, imagenet = True):
        if imagenet == True:
          load_weights = 'imagenet'
        else:
          load_weights = None
      
        # get the base model
        if model_name == 'small_model':
            model = Model.small_model(input_shape, class_names)
        elif model_name == 'alexnet':
            model = Model.alex_net(input_shape, class_names)
        elif model_name == 'resnet50':
            model = Model.resnet_50(input_shape, class_names, load_weights)
        elif model_name == 'vgg16':
            model = Model.vgg16(input_shape, class_names)
        elif model_name[:12] == "efficientnet":
            model = Model.efficientnet(input_shape, class_names, model_name, load_weights)
        elif model_name == 'densenet121':
          model = Model.densenet121(input_shape, class_names, load_weights)
        elif model_name =='mobilenetV2':
          model = Model.mobilenetV2(input_shape, class_names, load_weights)
        else:
            print("Error : wrong model_name, end of program")
            model = ()
            
        # compile the model
        # in most of the cases, this loss and this optimizer are the best
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer=tf.keras.optimizers.Adam(learning_rate),
                      metrics=['accuracy'])
        return model
