# Tensorflow classifier by Corentin Guyot

Create a virtual environnement (optional but recommended):

	--> install Miniconda :
	https://docs.conda.io/en/latest/miniconda.html
	
	--> create a python virtual environnement named "tf" :
	conda create --name tf python=3.9
	
	--> activate the env (to deactivate : conda deactivate tf):
	conda activate tf    


Install GPU support in the virtual environnement (optionnal but recommended):

	--> conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
	
	Next package correct some tensorflow-errors:
	--> conda install -c nvidia cuda-nvcc

Install dependencies to run the code (mandatory):
If you create a virtual env, you have to run these commands in the virtual env !!!

	--> install the dependencies :
	pip install -r PATH_TO_THE_FILE/requirements.txt
	
	
Test the gpu installation (optional):
	
	--> python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


# Run the code
If your create a virtual env, you have to run the code in the virtual env 
OR if you use Spyder, Pycharm, ..., set your python environnement path to the virtual environnement you created

	--> Edit the "config.yml" file to setup your project

	--> To run the code : python main.py
	

SOURCES :
https://www.tensorflow.org/install/pip?hl=fr#windows-native
	