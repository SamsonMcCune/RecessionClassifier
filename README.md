# Recession Classifier
An approach to recession classification using machine learning.

**Project Overview**

Using FRED-provided data, this project classifies each month according to the probability that it should have been classified by the National Bureau of Economic Research (NBER) as recessionary or not. It does so using either a random forest model through scikit-learn, or an ensemble method which includes classifier models such as XGBoost, LightGBM, and CatBoost.

**Data**

All data is taken from the FRED database.

https://www.stlouisfed.org/research/economists/mccracken/fred-databases

https://fred.stlouisfed.org/series/USREC

**To Install and Run:**

This project uses Python. Make sure that you have both Python 3.14 and the project downloaded, as well as the libraries listed in requirements.txt. To install libraries, use:

> pip install -r requirements.txt

in your terminal window, ensuring that you are operating in the same directory as the project.

After this, you can run either using the run button in your IDE or directly through your terminal by running:

> python {project_name}

once again ensuring that you are in the project directory.

**Output**

Output data is placed into folders associated with either the random forest or the ensemble as a whole. The configuration for each run of the model is stored here. Attached to this project are my outputs and configuration data associated with a single config state of this model. These are the same results I use in my Substack piece.

**Methodology and Motivation**

The hope is to take a large set of economic indicators and use them to properly predict whether periods will be recessionary or not. This project uses multiple classifier models, but the first model is a random forest classifier model. This uses the scikit-learn RandomForestClassifier function, which you can read more about here:

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

This takes a set of decision trees, another classifier model, and randomly samples input data to then create an ensemble of these decision trees and classifies the output data by averaging the probabilities from each decision tree. Read more about decision trees here:

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

For the "ensemble" model (in quotes because a random forest is technically already an ensemble), we add on other methods that explore adding extra trees and boosting, either through gradient boosting, XGBoost, LightGBM, or CatBoost.


