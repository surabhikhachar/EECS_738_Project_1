# EECS 738 Project 1

This is Andre Kurait and Surabhi Khachar's first project for EECS 738, Machine Learning in Spring 2019. For the project one, we built a K-Means algorithm using Python and tested it on datasets from Kaggle.

The project requirements were the following:

1. Set up a new git repository in your GitHub account
2. Pick two datasets from
https://www.kaggle.com/uciml/datasets
3. Choose a programming language (Python, C/C++, Java)
4. Formulate ideas on how machine learning can be used to
model distributions within the dataset
5. Build a heuristic and/or algorithm to model the data using
mixture models of probability distributions
programmatically
6. Document your process and results
7. Commit your source code, documentation and other
supporting files to the git repository in GitHub

<h1>Data</h1>
The data chosen for this project was from the Iris dataset and Biomechanical Features dataset.

https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients
  * Various biomechanical features of orthopedic patients such as pelvic radius, pelvic incidence, lumbar lordosis angle
  * Trying to classify patients based on if they have Disk Hernia or Spondylolisthesis (abnormal) or if they don't have either (normal)
  
https://www.kaggle.com/uciml/iris
  * Trying to predict species based on different properties of the flowers 
  * Sepal length, sepal width, petal length, petal width

<h1>K-Means Algorithm:</h1>
The algorithm we constructed is a basic algorithm to identify clusters given a certain set of features. It follows the discussion in this article: http://benalexkeen.com/k-means-clustering-in-python/
The K-Means algorithm has been written in a .py file and is in the 'Python' folder. The algorithm was then used in the Jupyter notebooks so that we were able to explore the datasets first prior to creating K-Means clusters. By creating a class we were easily able to call it for both datasets.
