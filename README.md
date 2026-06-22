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

Read more about these here:

https://en.wikipedia.org/wiki/XGBoost
https://catboost.ai/
https://en.wikipedia.org/wiki/LightGBM
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
https://en.wikipedia.org/wiki/Gradient_boosting


**Findings and Interpretation**

This program is attempting to take any given month and classify the probability that there will be a recession in the 12 months following that chosen month. Thus, this is an inherently predictive model. Because, however, this returns the probability that there will be a recession anywhere in the next 12 months, this could reasonably be interpreted as a nowcasting model as well as a forecasting model. This is how the model is able to "predict" COVID when the economy had no knowledge that a global pandemic was coming. Further, the classification of recessions is based on an unpredictable classification provided by NBER. This determines that the post COVID era was not recessionary and thus returns a set of "false positives" while I argue that in reality these could, or rather should, be seen as true positives. To see the findings directly, feel free to explore either output folder.

**Disclaimer**

This work was done in collaboration with ChatGPT for some parts of the programming. None of this README or the Substack post were written using the help of an LLM.
