import time

import numpy as np
import sklearn.metrics as metrics


class InConsole:
    def __init__(self):
      self.output_string_list = []
      
    # fonctions pour afficher dans le terminal et sauvegarder cette affichage dans un fichier
    def save_output_to_folder(self, folder_path):
      """
      save self.output_string_list to a .txt file
      """
      with open(folder_path, "w") as f:
        for string in self.output_string_list:
            f.write(string + "\n")
            
    def custom_print(self, *args, **kwargs):
        """
        put the string in the list and print it 
        """
        # save the string to the list
        output = ""
        for arg in args:
            output = output + str(arg)
        self.output_string_list.append(output)

        # Call the original print function to display the output in the terminal
        print(*args, **kwargs)
        
        
    def aff_metrics_2_classes(self, y_test, predictions):
        """
        affiche precision, recall et f1-score pour la classification a deux classes

        Parameters
        ----------
        y_test : tableau de label
            tableau des vrais labels.
        predictions : tableau de label
            tableau des labels predits.

        Returns
        -------
        None.

        """
        max_pred = np.argmax(predictions, axis=1)
        self.custom_print("-------------------")
        self.custom_print("Precision ", round(metrics.precision_score(y_test, max_pred), 2))
        self.custom_print("Recall ", round(metrics.recall_score(y_test, max_pred), 2))
        self.custom_print("f1_score ", round(metrics.f1_score(y_test, max_pred), 2))
        self.custom_print("-------------------")
        self.custom_print("precision = % de predictions positives qui sont vraiment positifs")
        self.custom_print("recall = % de vrais positifs qui ont été prédits positifs")
        self.custom_print("f1_score = qualité du modele ; combinaison de precision et recall")
        self.custom_print("-------------------")
        return

    @staticmethod
    def wait():
        time.sleep(10)
        
    def test_inference(self, x, y, model):
        """
        test le temps d'inference du model

        Parameters
        ----------
        x : np.array
            une image du dataset
        y : np.array
            le label associé à x
        rgb : boolean
            True si image couleur, false sinon.
        model : tf.Model
            modele à evaluer

        Returns
        -------
        None.

        """
        self.custom_print("!!!!test inférence...")
        tab_ms = []
        #test_inf = DatasetArray.preprocess_dataset(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), 1)
        test_inf = np.expand_dims(x, axis=0)
        for i in range(5):
            model(test_inf, training = False)
        for i in range(5):
            start_time = time.time()
            model(test_inf, training=False)
            end_time = time.time()
            tr_duration = end_time - start_time
            hours = tr_duration // 3600.
            minutes = (tr_duration - (hours * 3600.)) // 60.
            seconds = tr_duration - ((hours * 3600.) + (minutes * 60.))
            tab_ms.append(seconds)

        for i in range(5):
            ms = int(tab_ms[i] * 1000)
            msg = f'temps inference : {str(ms)} ms'
            self.custom_print(msg)
        self.custom_print('temps moyen : ' + str(round(np.mean(tab_ms) * 1000)) + ' ms')
        return
