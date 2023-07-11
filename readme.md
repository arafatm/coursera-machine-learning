#![](img/ [Machine Learning | Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

- [Discourse  - DeepLearning.AI](https://community.deeplearning.ai/top?period=daily)

## 1. [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning/home/info)

### 1.1: Introduction to Machine Learning

#### 1.1.01 What is machine learning?

> Field of study that gives computers the ability learn without being explicitly pogrammed -- Arthur Samuel

Supervised vs Unsupervised

#### 1.1.02 Supervised Learning: Regression Algorithms 

> learn x to y or input to output mappings

| Input (x)      | Output (Y)             | Application         |
| -              | -                      | -                   |
| email          | spam? (0/1)            | spam filtering      |
| audio          | text transcripts       | speech recognition  |
| Enlish         | Spanish                | machine translation |
| ad, user       | click? (0/1)           | online advertising  |
| image, radar   | position of other cars | self-driving car    |
| image of phone | defect? (0/1)          | visual inspection   |

Use different algorithms to predict price of house based on data
![](img/supervised.learning.regression.housing.price.prediction.png)

#### 1.1.03 Supervised Learning: Classification 

`Regression` attempts to predict ininite possible results
`Classification` **predicts categories** ie from limited possible results

e.g. Breast cancer detection
![](img/supervised.classification.breast.cancer.png)

we can have multiple outputs
![](img/supervised.learning.classification.malignant.png)

we can draw a boundary line to separate our output
![](img/classification.multiple.inputs.png)

Supervised Learning

|          | Regression | Classification |
| -        | -          | -              |
| Predicts | numbers    | categories     |
| Outputs  | infinite   | limited        |

#### 1.1.04 Unsupervised Learning 

With unsupervised we don't have predetermined expected output
- we're trying to find structure in the pattern
- in this example it's `clustering`
- e.g. Google News will "cluster" news related to pandas given a specific article about panda/birth/twin/zoo
![](img/unsupervised.clusturing.png)

e.g. given set of individual genes, "cluster" similar genes
![](img/clustering.dna.microarray.png)

e.g. how deeplearning.ai categorizes their learners
![](img/clustering.grouping.customers.png)

> `Unsupervised Learning`: Data only comes with inputs x, but not output labels y. 
> Algorithm has to find _structure_ in the data  
> - `Clustering`: Group similar data points together
> - `Anomaly Detection`: Find unusual data points
> - `Dimensionality Reduction`: Compress data using fewer numbers

###### Question: Of the following examples, which would you address using an unsupervised learning algorithm?  (Check all that apply.)

- [ ] Given a set of news articles found on the web, group them into sets of articles about the same stories.
- [ ] Given email labeled as spam/not spam, learn a spam filter.
- [ ] Given a database of customer data, automatically discover market segments and group customers into different market segments.
- [ ] Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

#### 1.1.05 **Lab:** Python and Jupyter Notebooks

[Python and Jupyter Notebooks | Coursera](https://www.coursera.org/learn/machine-learning/ungradedLab/rNe84/python-and-jupyter-notebooks/lab?path=%2Fnotebooks%2FC1_W1_Lab01_Python_Jupyter_Soln.ipynb)

#### **Quiz:** Supervised vs Unsupervised Learning

Which are the two common types of supervised learning (choose two)
- [ ] Classificaiton
- [ ] Regression
- [ ] Clustering

Which of these is a type of unsupervised learning?
- [ ] Clustering
- [ ] Regression
- [ ] Classification

#### 1.1.06 Linear regression model part 1

> `Linear Regression Model` => a **Supervised Learning Model** that simply puts a line through a dataset
- most commonly used model

e.g. Finding the right house price based on dataset of houses by sq ft.
01.01.house.size.and.price.png

| `Training Set` | data used to train the model    |
| -:             | :=                              |
| `x`            | *input* variable or **feature**
| `y`            | *output* variable or **target**
| `m`            | number of training examples
| `(x,y)`        | single training example
| `(xⁱ,yⁱ)`      | i-th training example

#### 1.1.07 Linear regression model part 2
#### 1.1.07 Lab: Optional lab: Model representation
#### 1.1.09 Cost function formula
#### 1.1.10 Cost function intuition
#### 1.1.11 Visualizing the cost function
#### 1.1.12 Visualization examples
#### 1.1.13 Lab: Optional lab: Cost function
ting 
