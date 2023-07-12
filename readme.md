#![](img/ [Machine Learning | Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

- [Discourse  - DeepLearning.AI](https://community.deeplearning.ai/top?period=daily)

##  [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning/home/info)

### 1: Introduction to Machine Learning

#### What is machine learning?

> Field of study that gives computers the ability learn without being explicitly pogrammed -- Arthur Samuel

Supervised vs Unsupervised

#### Supervised Learning: Regression Algorithms 

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

#### Supervised Learning: Classification 

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

#### Unsupervised Learning 

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

#### **Lab:** Python and Jupyter Notebooks

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

#### Linear regression model part 1

> `Linear Regression Model` => a **Supervised Learning Model** that simply puts a line through a dataset
- most commonly used model

e.g. Finding the right house price based on dataset of houses by sq ft.
01.01.house.size.and.price.png

| Terminology  |                                 |
| -:           | :-                              |
| Training Set | data used to train the model    |
| x            | *input* variable or **feature**
| y            | *output* variable or **target**
| m            | number of training examples
| (x,y)        | single training example
| (xⁱ,yⁱ)      | i-th training example

#### Linear regression model part 2

```mermaid
flowchart TD

A[training set] --> B[learning algorithm]
B --> F[f (function)]
```

`f` is a linear function with _one_ variable
- $`f_{w,b}(x) = wx + b`$ is equivalent to 
- $`f(x) = wx + b`$

> `Univariate` linear regression => one variable

01.01.linear.regression.png

#### [Lab: Optional lab: Model representation](https://www.coursera.org/learn/machine-learning/ungradedLab/PhN1X/optional-lab-model-representation/lab)
Here is a summary of some of the notation you will encounter.  

| General Notation        | Python (if applicable) | Description                                                                                                   |
| :--                     | :--                    | :--                                                                                                           |
| $`a`$                  |                        | scalar, non bold                                                                                              |
| $`\mathbf{a}`$         |                        | vector, bold                                                                                                  |
| **Regression**          |                        |                                                                                                               |  |
| $`\mathbf{x}`$         | `x_train`              | Training Example feature values (in this lab - Size (1000 sqft))                                              |
| $`\mathbf{y}`$         | `y_train`              | Training Example  targets (in this lab Price (1000s of dollars))
| $`x^{(i)}$, $y^{(i)}`$ | `x_i`, `y_i`           | $`i_{th}`$ Training Example                                                                                      |
| m                       | `m`                    | Number of training examples                                                                                   |
| $`w`$                  | `w`                    | parameter: weight                                                                                             |
| $`b`$                  | `b`                    | parameter: bias                                                                                               |
| $`f_{w,b}(x^{(i)})`$   | `f_wb`                 | The result of the model evaluation at $`x^{(i)}`$ parameterized by $`w,b`$: $`f_{w,b}(x^{(i)}) = wx^{(i)}+b`$ |

Code
- `NumPy`, a popular library for scientific computing
- `Matplotlib`, a popular library for plotting data
  - `scatter()` to plot on a graph
    - `marker` for symbol to use
    - `c` for color

#### Cost function formula

1st step to implement linear function is to define `Cost Function`
- $`f_{w,b}(x) = wx + b`$
- `w` is the **slope**
- `b` is the **y-intercept** ie where the line intersects x=0

01.01.parameters.png

We can play with `w` & `b` to find the best fit line
- $`\hat{y}`$
- _Cost function_ takes  predicted $`\hat{y}`$ and compares to `y`  to measure the __error__ 
  - $`\sum_{i=1}^{m} (\hat{y}^{(i) - y^i})^2`$

#### Cost function intuition
#### Visualizing the cost function
#### Visualization examples
#### Lab: Optional lab: Cost function
<!-- vim: set textwidth=0: -->
<!-- vim: set wrapmargin=0: -->
<!-- vim: set nowrap: -->
<!-- vim: set foldlevel=9: -->
