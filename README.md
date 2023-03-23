# Initial Setup (needed if you run on your PC and not on the server)

1. Create VirtualEnv : `python3 -m venv .venv `
2. Activate VirtualEnv: `source .venv/bin/activate`
3. Install Packages : `pip install -r requirements.txt`

# Git Commands

1. `git add [filename]`
2. `git commit -m [message]`
3. `git push`
4. `git pull`

# File Structure explained
`data` folder is used to store all the images categorized by their label.

`cache` folder is used to store the image downloaded from the web in the file `run_model.py`.

`deliverable` folder is used to store the final model that has been trained to later be used in `run_model.py`.

`.gitignore` file is used to specify to Git which files and folders to ignore (these files and folders will never be uploaded to Git).

`main.py` is the main file where we load the images, sort them by label, create the NN architecture, train the model, evaluate it and save it (in `deliverable` folder as a `.h5` file).

`run_model.py` is the file where we can download an image from the web and use the classifier to predict a label.

`requirements.txt` is a file that is used to download all the Python packages needed to run the scripts (with `pip install -r requirements.txt`).

# Script `main.py` explained

1. First we import all the packages that we need in the script.

2. Set variables such as which GPU to use or model hyperparameters.

3. `data_augmentation` defines sequences of layers to augment the data with eg `RandomRotation` (it does not really make a difference. still testing it).

4. `sort_images()` method reads the excel file to know which image corresponds to which label. Then for each label available, it creates a folder and puts its images in it. Since we have `dcm` images, we convert them before into `png` images.

5. `load_model()` method creates and return the model. Initially we define the feature extraction part structure (with Conv2D, Activation, ...). Then we have the dense layers to analyze the features extracted before.

6. We create train, validation and test set (80%, 10%, 10%).

7. We compile the model with the chosen optimizer, learning rate and metrics to evaluate.

8. We train the model using the `train_ds` and `validation_ds`.

9. We evaluate the score of the model using the `test_ds` (dataset never seen before by the model -> unbiased evaluation).

10. We save the model so that we can re-use it later.

# Script `run_model.py` explained

1. Import needed packages.

2. Load the model that we saved before.

3. Set the URL to download the image to evaluate.

4. Set the path of the location where we store the image.

5. Convert the image into array.

6. Predict the label with the classifier.

7. Print result and confidence.