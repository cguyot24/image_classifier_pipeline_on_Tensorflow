import sys
import pathlib
import albumentations as A


class ConfigSettings:
    @staticmethod
    def error_config(var, var_name, var_type):
        print(f'Error on variable "{var_name}": {var} is not a {var_type}')
        sys.exit(1)

    @staticmethod
    def is_str(var, var_name):
        if not isinstance(var, str):
            ConfigSettings.error_config(var, var_name, "string")
        return var

    @staticmethod
    def is_int(var, var_name):
        if not isinstance(var, int):
            ConfigSettings.error_config(var, var_name, "int")
        return var

    @staticmethod
    def is_list(var, var_name):
        if not isinstance(var, list):
            ConfigSettings.error_config(var, var_name, "list")
        return var

    @staticmethod
    def is_dict(var, var_name):
        if not isinstance(var, dict):
            ConfigSettings.error_config(var, var_name, "dict")
        return var

    @staticmethod
    def is_path(path, var_name):
        path_str = ConfigSettings.is_str(path, var_name)
        # convert to a unix path
        path_str = pathlib.PureWindowsPath(path_str).as_posix()
        # path_str = path_str + '/'
        # TODO check valid path
        return path_str

    @staticmethod
    def is_float(var, var_name):
        if not isinstance(var, float):
            ConfigSettings.error_config(var, var_name, "float")
        return var

    @staticmethod
    def is_percentage(var, var_name):
        ConfigSettings.is_float(var, var_name)
        if not (0. <= var <= 1.):
            ConfigSettings.error_config(var, var_name, "between 0 and 1")
        return var

    @staticmethod
    def is_boolean(var, var_name):
        if not ((var is True) or (var is False)):
            print(f'Error on variable "{var_name}": {var} is not "true" or "false"')
            sys.exit(1)
        return var

    def __init__(self, d):
        self.action = self.is_list(d['action'], 'action')

        # create dataset
        cd = d['create_dataset']
        self.class_names = self.is_list(cd['class_names'], 'class_names')

        self.input_cd = self.is_path(cd['input_directory'], 'input_directory')
        self.output_cd = self.is_path(cd['output_directory'], 'output_directory')

        self.train_split_size = self.is_percentage(cd['train_split_size'], 'train_split_size')
        self.val_split_size = self.is_percentage(cd['val_split_size'], 'val_split_size')

        self.scale = []  # scale = input_size
        self.scale.append(self.is_int(cd['input_size']['width'], 'input_size'))
        self.scale.append(self.is_int(cd['input_size']['height'], 'input_size'))
        
        # augmentation
        self.do_augmentation = self.is_boolean(cd['do_augmentation'], 'do_augmentation')

        aug = cd['augmentation']
        self.nb_images_aug = self.is_int(aug["nb_images_aug"], "nb_images_aug")
        self.rescale_before_augmentation = self.is_boolean(aug['rescale_before_augmentation'],
                                                           'rescale_before_augmentation')

        self.rba_scale = []
        self.rba_scale.append(self.is_int(aug['res_bef_aug_scale']['width'], 'res_bef_aug_scale'))
        self.rba_scale.append(self.is_int(aug['res_bef_aug_scale']['height'], 'res_bef_aug_scale'))

        trans = aug["transform"]
        transform_dict = {
            'RandomCrop': A.RandomCrop(**trans["RandomCrop"]),
            'CenterCrop': A.CenterCrop(**trans["CenterCrop"]),
            'Blur': A.Blur(**trans["Blur"]),
            'Rotate': A.Rotate(**trans["Rotate"]),
            'HorizontalFlip': A.HorizontalFlip(**trans["HorizontalFlip"]),
            'VerticalFlip': A.VerticalFlip(**trans["VerticalFlip"]),
            'RGBShift': A.RGBShift(**trans["RGBShift"]),
            'ColorJitter': A.ColorJitter(**trans["ColorJitter"]),
            'Resize': A.Resize(self.rba_scale[1], self.rba_scale[0], always_apply=True)
        }

        # add OneOf to transform_dict
        list_one_of = []
        for one_of_transform in trans["OneOf"]["list_OneOf"]:
            list_one_of.append(transform_dict[one_of_transform])
        transform_dict["OneOf"] = A.OneOf(list_one_of, trans["OneOf"]["p"])

        # add Resize to every augmentation to rescale to the original size in case of Crop
        # it's because all images should have the same size after augmentation
        aug["list_transform"].append("Resize")

        # create the transform object use by albumentations
        transform = []
        for tr in aug["list_transform"]:
            transform.append(transform_dict[tr])
        self.transform = A.Compose(transform)

        # train_model
        tm = d["train_model"]
        self.input_tm = self.is_path(tm["dataset_directory"], "dataset_directory")
        self.output_tm = self.is_path(tm["output_directory"], "output_directory")

        model = tm["model"]
        self.model_type = self.is_str(model["model_type"], "model_type")

        self.load_weights = self.is_boolean(model["load_weights"], "load_weights")
        self.weights_file_tm = self.is_path(model["weights_file_tm"], "weights_file_tm")

        self.imagenet = self.is_boolean(model['use_imagenet_weights'], 'use_imagenet_weights')
         
        hp = tm["hyperparametres"]
        self.epochs = self.is_int(hp["epochs"], "epochs")
        self.patience = self.is_int(hp["patience"], "patience")
        self.learning_rate = self.is_float(hp["learning_rate"], "learning_rate")
        self.batch_size = self.is_int(hp["batch_size"], "batch_size")
        self.monitor = self.is_str(hp["monitor"], "monitor")
        self.save_parametres_step = self.is_int(hp["save_parametres_step"], "save_parametres_step")

        # evaluate model
        em = d["evaluate_model"]
        em_lm = em["load_model"]
        self.weights_folder_em = self.is_path(em_lm["weights_folder"], "weights_folder")
        self.load_best_epoch_em = self.is_boolean(em_lm["load_best_epoch"], "load_best_epoch")
        self.epoch_number_em = self.is_int(em_lm["epoch_number"], "epoch_number")
        
        self.dataset_test_directory = self.is_path(em["dataset_test_directory"], "dataset_test_directory")
        self.keys_images = self.is_list(em["keys"], "evaluate/keys")

        self.display_all_predictions = self.is_boolean(em["display_all_predictions"], "display_all_predictions")
        self.nb_pred = self.is_int(em["nb_pred"], "nb_pred")
        self.display_bad_results = self.is_boolean(em["display_bad_results"], "display_bad_results")
        
        # export the model
        ex = d["export_model"]
        ex_lm = ex["load_model"]
        self.weights_folder_ex = self.is_path(ex_lm["weights_folder"], "weights_folder")
        self.load_best_epoch_ex = self.is_boolean(ex_lm["load_best_epoch"], "load_best_epoch")
        self.epoch_number_ex = self.is_int(ex_lm["epoch_number"], "epoch_number")
        
        self.keys_export = self.is_list(ex["keys"], "export/keys")
