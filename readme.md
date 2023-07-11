# [Machine Learning | Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

- [Discourse  - DeepLearning.AI](https://community.deeplearning.ai/top?period=daily)

## 1. [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning/home/info)

### 1.1: Introduction to Machine Learning

#### 1.1.1 What is machine learning?

> Field of study that gives computers the ability learn without being explicitly pogrammed -- Arthur Samuel

Supervised vs Unsupervised

#### 1.1.2 Supervised Learning: Regression Algorithms 

> learn x to y or input to output mappings

| Input (x)      | Output (Y)             | Application         |
| -              | -                      | -                   |
| email          | spam? (0/1)            | spam filtering      |
| audio          | text transcripts       | speech recognition  |
| Enlish         | Spanish                | machine translation |
| ad, user       | click? (0/1)           | online advertising  |
| image, radar   | position of other cars | self-driving car    |
| image of phone | defect? (0/1)          | visual inspection   |

![](img/supervised.learning.regression.housing.price.prediction.png)
- Use different algorithms to predict price of house based on data

#### 1.1.3 Supervised Learning: Classification 

`Regression` attempts to predict ininite possible results
`Classification` **predicts categories** ie from limited possible results

e.g. Breast cancer detection
![](img/supervised.classification.breast.cancer.png)

![](img/supervised.learning.classification.malignant.png)

![](img/classification.multiple.inputs.png)
- we can have multiple inputs
- we can draw a boundary line to separate our output

Supervised Learning

|         | Regression | Classification |
| -       | -          | -              |
| Predict | number     | categories     |
| Outputs | infinite   | limited        |

#### 1.1.4 Unsupervised Learning 

![](img/unsupervised.clusturing.png)
- With unsupervised we don't have predetermined expected output
- we're trying to find structure in the pattern
- in this example it's `clustering`
- e.g. Google News will "cluster" news related to pandas given a specific article about panda/birth/twin/zoo

![](img/clustering.dna.microarray.png)
- e.g. given set of individual genes, "cluster" similar genes

![](img/clustering.grouping.customers.png)
- how deeplearning.ai categorizes their learners

> `Unsupervised Learning`: Data only comes with inputs x, but not output labels y. 
> Algorithm has to find _structure_ in the data  
> - Clustering: Group similar data points together
> - Anomaly Detection: Find unusual data points
> - Dimensionality Reduction: Compress data using fewer numbers

###### Question: Of the following examples, which would you address using an unsupervised learning algorithm?  (Check all that apply.)
- [ ] Given a set of news articles found on the web, group them into sets of articles about the same stories.
- [ ] Given email labeled as spam/not spam, learn a spam filter.
- [ ] Given a database of customer data, automatically discover market segments and group customers into different market segments.
- [ ] Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

#### 1.1.5 **Lab:** Python and Jupyter Notebooks

[Python and Jupyter Notebooks | Coursera](https://www.coursera.org/learn/machine-learning/ungradedLab/rNe84/python-and-jupyter-notebooks/lab?path=%2Fnotebooks%2FC1_W1_Lab01_Python_Jupyter_Soln.ipynb)

xxx

