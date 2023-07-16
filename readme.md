# [Machine Learning | Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

- [Discourse  - DeepLearning.AI](https://community.deeplearning.ai/top?period=daily)


<!-- vim-markdown-toc GFM -->

* [C1 Supervised Machine Learning: Regression and Classification](#c1-supervised-machine-learning-regression-and-classification)
  * [C1_W1 Introduction to Machine Learning](#c1_w1-introduction-to-machine-learning)
    * [C1_W1_M1 Supervised vs Unsuperverside Machine Learning](#c1_w1_m1-supervised-vs-unsuperverside-machine-learning)
      * [C1_W1_M1_1 What is machine learning?](#c1_w1_m1_1-what-is-machine-learning)
      * [C1_W1_M1_2 Supervised Learning: Regression Algorithms](#c1_w1_m1_2-supervised-learning-regression-algorithms)
      * [C1_W1_M1_3 Supervised Learning: Classification](#c1_w1_m1_3-supervised-learning-classification)
      * [C1_W1_M1_4 Unsupervised Learning](#c1_w1_m1_4-unsupervised-learning)
        * [Question: Of the following examples, which would you address using an unsupervised learning algorithm?](#question-of-the-following-examples-which-would-you-address-using-an-unsupervised-learning-algorithm)
      * [Lab: Python and Jupyter Notebooks](#lab-python-and-jupyter-notebooks)
      * [Quiz: Supervised vs Unsupervised Learning](#quiz-supervised-vs-unsupervised-learning)
    * [C1_W1_M2 Regression Model](#c1_w1_m2-regression-model)
      * [C1_W1_M2_1 Linear regression model part 1](#c1_w1_m2_1-linear-regression-model-part-1)
      * [C1_W1_M2_2 Linear regression model part 2](#c1_w1_m2_2-linear-regression-model-part-2)
      * [Lab: Model representation](#lab-model-representation)
      * [C1_W1_M2_4 Cost function formula](#c1_w1_m2_4-cost-function-formula)
        * [Question: Which of these parameters of the model that can be adjusted?](#question-which-of-these-parameters-of-the-model-that-can-be-adjusted)
      * [C1_W1_M2_5 Cost Function Intuition](#c1_w1_m2_5-cost-function-intuition)
      * [C1_W1_M2_6 Visualizing the cost function](#c1_w1_m2_6-visualizing-the-cost-function)
      * [C1_W1_M2_7 Visualization examples](#c1_w1_m2_7-visualization-examples)
      * [Lab: Cost function](#lab-cost-function)
      * [Quiz: Regression Model](#quiz-regression-model)
    * [C1_W1_M3 Train the model with gradient descent](#c1_w1_m3-train-the-model-with-gradient-descent)
      * [C1_W1_M3_1 Gradient descent](#c1_w1_m3_1-gradient-descent)
      * [C1_W1_M3_2 Implementing gradient descent](#c1_w1_m3_2-implementing-gradient-descent)
      * [C1_W1_M3_3 Gradient descent intuition](#c1_w1_m3_3-gradient-descent-intuition)
      * [C1_W1_M3_4 Learning rate](#c1_w1_m3_4-learning-rate)
      * [C1_W1_M3_5 Gradient descent for linear regression](#c1_w1_m3_5-gradient-descent-for-linear-regression)
      * [C1_W1_M3_6 Running gradient descent](#c1_w1_m3_6-running-gradient-descent)
      * [Lab: Gradient descent](#lab-gradient-descent)
      * [Quiz: Train the Model with Gradient Descent](#quiz-train-the-model-with-gradient-descent)
  * [C1_W2: Regression with Multiple Input Variables](#c1_w2-regression-with-multiple-input-variables)
    * [C1_W2_M1 Multiple Linear Regression](#c1_w2_m1-multiple-linear-regression)
      * [C1_W2_M1_1 Multiple features](#c1_w2_m1_1-multiple-features)
      * [C1_W2_M1_2 Vectorization part 1](#c1_w2_m1_2-vectorization-part-1)
      * [C1_W2_M1_3 Vectorization part 2](#c1_w2_m1_3-vectorization-part-2)
      * [Lab: Python, NumPy and vectorization](#lab-python-numpy-and-vectorization)
      * [C1_W2_M1_4 Gradient descent for multiple linear regression](#c1_w2_m1_4-gradient-descent-for-multiple-linear-regression)
      * [Lab: Muliple linear regression](#lab-muliple-linear-regression)
    * [C1_W2_M2 Gradient Descent In Practice](#c1_w2_m2-gradient-descent-in-practice)

<!-- vim-markdown-toc -->

## [C1 Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning/home/info)

### C1_W1 Introduction to Machine Learning

#### C1_W1_M1 Supervised vs Unsuperverside Machine Learning

##### C1_W1_M1_1 What is machine learning?

> Field of study that gives computers the ability learn without being explicitly pogrammed -- Arthur Samuel

Supervised vs Unsupervised

##### C1_W1_M1_2 Supervised Learning: Regression Algorithms 

> learn x to y or input to output mappings

| Input (x)      | Output (Y)             | Application         |
| -              | -                      | -                   |
| email          | spam? (0/1)            | spam filtering      |
| audio          | text transcripts       | speech recognition  |
| Enlish         | Spanish                | machine translation |
| ad, user       | click? (0/1)           | online advertising  |
| image, radar   | position of other cars | self-driving car    |
| image of phone | defect? (0/1)          | visual inspection   |

![img/supervised.learning.regression.housing.price.prediction.png](img/supervised.learning.regression.housing.price.prediction.png)

- Use different algorithms to predict price of house based on data

##### C1_W1_M1_3 Supervised Learning: Classification 

`Regression` attempts to predict ininite possible results
`Classification` **predicts categories** ie from limited possible results

![img/supervised.classification.breast.cancer.png](img/supervised.classification.breast.cancer.png)

- e.g. Breast cancer detection

![img/supervised.learning.classification.malignant.png](img/supervised.learning.classification.malignant.png)

- we can have multiple outputs

![img/classification.multiple.inputs.png](img/classification.multiple.inputs.png)

- we can draw a boundary line to separate our output

Supervised Learning

|          | Regression | Classification |
| -        | -          | -              |
| Predicts | numbers    | categories     |
| Outputs  | infinite   | limited        |

##### C1_W1_M1_4 Unsupervised Learning 

![img/unsupervised.clusturing.png](img/unsupervised.clusturing.png)

- With unsupervised we don't have predetermined expected output
- we're trying to find structure in the pattern
- in this example it's `clustering`
- e.g. Google News will "cluster" news related to pandas given a specific article about panda/birth/twin/zoo

![img/clustering.dna.microarray.png](img/clustering.dna.microarray.png)

- e.g. given set of individual genes, "cluster" similar genes

![img/clustering.grouping.customers.png](img/clustering.grouping.customers.png)

- e.g. how deeplearning.ai categorizes their learners

> `Unsupervised Learning`: Data only comes with inputs x, but not output labels y. 
> Algorithm has to find _structure_ in the data  
> - `Clustering`: Group similar data points together
> - `Anomaly Detection`: Find unusual data points
> - `Dimensionality Reduction`: Compress data using fewer numbers

###### Question: Of the following examples, which would you address using an unsupervised learning algorithm?  

(Check all that apply.)
- [ ] Given a set of news articles found on the web, group them into sets of articles about the same stories.
- [ ] Given email labeled as spam/not spam, learn a spam filter.
- [ ] Given a database of customer data, automatically discover market segments and group customers into different market segments.
- [ ] Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

##### Lab: Python and Jupyter Notebooks

[Python and Jupyter Notebooks | Coursera](https://www.coursera.org/learn/machine-learning/ungradedLab/rNe84/python-and-jupyter-notebooks/lab?path=%2Fnotebooks%2FC1_W1_Lab01_Python_Jupyter_Soln.ipynb)

##### Quiz: Supervised vs Unsupervised Learning

Which are the two common types of supervised learning (choose two)
- [ ] Classificaiton
- [ ] Regression
- [ ] Clustering

Which of these is a type of unsupervised learning?
- [ ] Clustering
- [ ] Regression
- [ ] Classification

#### C1_W1_M2 Regression Model

##### C1_W1_M2_1 Linear regression model part 1

`Linear Regression Model` => a **Supervised Learning Model** that simply puts a line through a dataset
- most commonly used model

![img/01.01.house.size.and.price.png](img/01.01.house.size.and.price.png)

- e.g. Finding the right house price based on dataset of houses by sq ft.

| Terminology  |                                 |
| -:           | :-                              |
| Training Set | data used to train the model    |
| x            | *input* variable or **feature**
| y            | *output* variable or **target**
| m            | number of training examples
| (x,y)        | single training example
| (xⁱ,yⁱ)      | i-th training example

##### C1_W1_M2_2 Linear regression model part 2

```mermaid
flowchart TD

A[training set] --> B[learning algorithm]
B --> F[f (function)]
```

`f` is a linear function with _one_ variable
- $`f_{w,b}(x) = wx + b`$ is equivalent to 
- $`f(x) = wx + b`$

![img/01.01.linear.regression.png](img/01.01.linear.regression.png)

- `Univariate` linear regression => one variable

##### [Lab: Model representation](https://www.coursera.org/learn/machine-learning/ungradedLab/PhN1X/optional-lab-model-representation/lab)

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

##### C1_W1_M2_4 Cost function formula

![img/01.01.parameters.png](img/01.01.parameters.png)
- We can play with `w` & `b` to find the best fit line

![img/01.01.cost.function.png](img/01.01.cost.function.png)
- 1st step to implement linear function is to define `Cost Function`
-  Given $`f_{w,b}(x) = wx + b`$ where `w` is the `slope` and `b` is the `y-intercept` 
- `Cost function` takes  predicted $`\hat{y}`$ and compares to `y`
- ie `error` = $`\hat{y} - y`$ 
- $`\sum\limits_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^{2}`$ 
  - where `m` is the number of training examples
- Dividing by `2m` makes the calculation neater $`\frac{1}{2m} \sum\limits_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^{2}`$ 
- Also known as `squared error cost function` $`J_{(w,b)} = \frac{1}{2m} \sum\limits_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^{2}`$ 
- Which can be rewritten as $`J_{(w,b)} = \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^{2}`$ 
- Remember we want to find values of `w,b` where $`\hat{y}^{(i)}`$ is close to $`y^{(i)}`$ for all $`(x^{(i)}, y^{(i)})`$

###### Question: Which of these parameters of the model that can be adjusted?
- $`w`$ and $`b`$
- $`f_{w,b}`$
- $`w`$ only, because we should choose $`b = 0`$
- $`\hat{y}`$

##### C1_W1_M2_5 Cost Function Intuition

To get a sense of how to minimize $`J`$ we can use a simplified model

|               |                                                                                     | simplified |
| --            | --                                                                                  | --         |
| model         | $`f_{w,b}(x) = wx + b`                                                              | $`f_{w}(x) = wx` by setting $`b=0`$                                                              |
| parameters    | $`w`$, $`b`$                                                                        | $`w`$
| cost function | $`J_{(w,b)} = \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^{2}`$ | $`J_{(w)} = \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w}(x^{(i)}) - y^{(i)})^{2}`$ |
| goal          | we want to minimize $`J_{(w,b)}`$                                                   | we want to minimize $`J_{(w)}`$                                                   |

![img/01.01.04.simplified.png](img/01.01.04.simplified.png)
- we can use simplified function to find the best fit line 

![img/01.01.04.w.is.1.png](img/01.01.04.w.is.1.png)
- the 2nd graph shows that when $`w = 1`$ then $`J(1) = 0`$

![img/01.01.04.w.is.0.5.png](img/01.01.04.w.is.0.5.png)
- the 2nd graph shows that when $`w = 0.5`$ then $`J(0.5) ~= 0.58`$

![img/01.01.04.w.is.0.png](img/01.01.04.w.is.0.png)
- the 2nd graph shows that when $`w = 0`$ then $`J(0) ~= 2.33`$

![img/01.01.04.negative.w.png](img/01.01.04.negative.w.png)
- We can do this calculation for various $`w`$ even negative numbers
- when $`w = -0.5`$ then $`J(-0.5) ~= 5.25`$

![img/01.01.04.J.png](img/01.01.04.J.png)
- We can plot various values for `w` and get a graph (on the right)
- As we can see the cost function with $`w = 1`$ is the best fit line for __this data__

> :bulb: The goal of linear regression is to find the values of $`w,b`$ that
> allows us to minimize $`J_{(w,b)}`$
x
##### C1_W1_M2_6 Visualizing the cost function

|               |                                                                                     |
| --            | --                                                                                  |
| model         | $`f_{w,b}(x) = wx + b`                                                              |
| parameters    | $`w`$, $`b`$                                                                        |
| cost function | $`J_{(w,b)} = \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^{2}`$ |
| goal          | minimize $`J_{(w,b)}`$                                                              |

![](img/01.01.05.J.w.png)
- When we only have w, then we can plot `J` vs `w` in 2-dimensions

![](img/01.01.05.soup.bowl.png)
- However, when we add `b` then it's 3-dimensional
- The value of `J` is the height

![](img/01.01.05.contour.plot.png)
- this is easier to visualize as a `contour plot`

![](img/01.01.05.mount.fuji.png)
- Same as used to visualize height of mountains

![](img/01.01.05.contour.visualize.png)
- take a horizontal slice which gives you the same `J` for given `w,b`
- the center of the contour is the minimum
- __Countour__ allows us to visualize the 3-D `J` in 2-D

##### C1_W1_M2_7 Visualization examples

Here are some examples of `J`

![](img/01.01.06.bad.png)
- We can see this is a pretty bad `J`

![](img/01.01.06.less.bad.png)
- A little less bad `J`

![](img/01.01.06.good.png)
- This is pretty good and close to minimal (but not quite perfect)

In the next lab, you can click on different points on the contour to view the cost function on the graph

> `Gradient Descent` is an algorithm to train linear regression and other complex models 

##### [Lab: Cost function](https://www.coursera.org/learn/machine-learning/ungradedLab/udPHh/optional-lab-cost-function/lab?path=%2Fnotebooks%2FC1_W1_Lab03_Cost_function_Soln.ipynb)

[Lab: Cost Function](/code/C1_W1_Lab03_Cost_function_Soln.ipynb)
- `from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl`

##### Quiz: Regression Model

1. Which of the following are the inputs, or features, that are fed into the model and with which the model is expected to make a prediciton?
- [ ] $`m`$
- [ ] $`w`$ and $`b`$
- [ ] $`(x,y)`$
- [ ] $`x`$

2. For linear regression, if you find parameters $`w`$ and $`b`$ so that $`J_{(w,b)}`$ is very close to zero, what can you conclude?
- [ ] The selected values of the parameters $`w, b`$ cause the algorithm to fit the training set really well
- [ ] This is never possible. There must be a bug in the code
- [ ] The selected values of the parameters $`w, b`$ cause the algorithm to fit the training set really poorly

<details><summary>Ans</summary>4, 1</details>

#### C1_W1_M3 Train the model with gradient descent

##### C1_W1_M3_1 Gradient descent

Want a systematic way to find values of $`w,b`$ that allows us to easily find
smallest $`J`$

`Gradient Descent` is an algorithm used for any function, not just in linear
regression but also in advanced neural network models

![](/img/1.1.3.1.algorithm.png)
- start with some $`w,b`$ e.g. $`(0,0)`$
- keep changing $`w,b`$ to reduce $`J(w,b)`$
- until we settle at or near a minimume

![](/img/1.1.3.1.local.minima.png)
- Example of a more comples $`J`$
- not a squared error cost
- not linear regression
- we want to get to the lowest point in this topography
- pick a direction and take a step that is slightly lower
  - repeat until you're at lowest point
- However, depending on starting point and direction, you will end up at a different "lowest point"
  - Known as `local mimina`

> `local minima` may not be the true lowest point

##### C1_W1_M3_2 Implementing gradient descent

![](/img/1.1.3.2.algorithm.png)
- The `Gradient Descent` algorithm
- $`w = w - \alpha \frac{\partial}{\partial w} J_{(w,b)}`$
  - $`\alpha`$ == `learning rate`. ie How "big a step" you take down the hill
  - $`\frac{\partial}{\partial w} J_{(w,b)}`$ == `derivative`. ie which direction
- $`b = b - \alpha \frac{\partial}{\partial b} J_{(w,b)}`$
- We repeat these 2 steps for $`w,b`$ until the algorithm __converges__
  - ie each subsequent step doesn't change the value
- We want to _simultaneously_ update w and b at each step
  - `tmp_w =` $`w - \alpha \frac{\partial}{\partial w} J_{(w,b)}`$
  - `tmp_b =` $`b - \alpha \frac{\partial}{\partial b} J_{(w,b)}`$
  - `w = tmp_w && b = tmp_b`

##### C1_W1_M3_3 Gradient descent intuition

![](/img/1.1.3.3.min.J.png)
- We want to find minimum `w,b`
```math
\begin{aligned}
\text{repeat until convergence \{} \\ 
  &w = w - \alpha \frac{\partial}{\partial w} J_{(w,b)}\\
  &b = b - \alpha \frac{\partial}{\partial b} J_{(w,b)}\\
\}
\end{aligned}
```
- Starting with finding `min w` we can simplify to just $`J(w)`$
- Gradient descent with $`w = w - \alpha \frac{\partial}{\partial w} J_{(w)}`$
- minimize cost by adjusting just `w`: $`\min J(w)`$

![](/img/1.1.3.3.slope.png)
- Recall previous example where we set `b = 0`
- Initialize `w` at a random location
- $`\frac{\partial}{\partial w} J(w)`$ is the __slope__
  - we want to find slopes __that take us to minimum w__
- In the first case, we get $`w - \alpha (positive number)`$ which is the correct direction
- However (_2nd graph_), slope is negative, and therefore also in the correct direction

##### C1_W1_M3_4 Learning rate

![](/img/1.1.3.4.alpha.png)
- $`\alpha`$ is the __learning rate__ ie how big a step to take
  - If _too small_ then you take small steps and will take a long time to find minimum
  - If _too big_ then you might miss true minimum ie __diverge__ instead of _converge_

![](/img/1.1.3.4.local.minimum.png)
- If you're already at local minimum...
- `slope = 0` and therefore $`\frac{\partial}{\partial w} J(w) = 0`$ 
  - ie `w = w * 0`
  - further _steps_ will bring you back here

![](/img/1.1.3.4.derivative.png)
- As we get closer to local minimum, gradient descent (_derivative function_)
  will __automatically take smaller steps__

##### C1_W1_M3_5 Gradient descent for linear regression

![](/img/1.1.3.5.recap.png)
- linear regression model     $`f_{w,b}(x) = wx + b`$
- cost function               $`J_{(w,b)} = \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)} - y^{(i)})^{2}`$
- gradient descent algorithm  
  - `repeat until convergence {`
  - $`w = w - \alpha \frac{\partial}{\partial w} J_{(w,b)}`$
  - $`b = b - \alpha \frac{\partial}{\partial b} J_{(w,b)}`$
  - `}`
- where $`\frac{\partial}{\partial w} J_{(w,b)}`$ = $`\frac{1}{m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)} - y^{(i)})x^{(i)}`$
- and $`\frac{\partial}{\partial b} J_{(w,b)}`$ = $`\frac{1}{m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)} - y^{(i)})`$

![](/img/1.1.3.5.optional.png)
- We can simplify for `w`
```math
\begin{align}
\frac{\partial}{\partial w} J_{(w,b)} \\
&= \frac{\partial}{\partial w} \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)} - y^{(i)})^2 \\
&= \frac{\partial}{\partial w} \frac{1}{2m} \sum\limits_{i=1}^{m} (wx^{(i)} + b - y^{(i)})^2 \\
&= \frac{1}{2m} \sum\limits_{i=1}^{m} (wx^{(i)} + b - y^{(i)}) 2x^{(i)} \\
&= \frac{1}{m} \sum\limits_{i=1}^{m} (wx^{(i)} + b - y^{(i)}) x^{(i)} \\
&= \frac{1}{m} \sum\limits_{i=1}^{m} ((f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)}
\end{align}
```
- and for `b`
```math
\begin{align}
\frac{\partial}{\partial b} J_{(w,b)} \\
&= \frac{\partial}{\partial b} \frac{1}{2m} \sum\limits_{i=1}^{m} (f_{w,b}(x^{(i)} - y^{(i)})^2 \\
&= \frac{\partial}{\partial b} \frac{1}{2m} \sum\limits_{i=1}^{m} (wx^{(i)} + b - y^{(i)})^2 \\
&= \frac{1}{2m} \sum\limits_{i=1}^{m} (wx^{(i)} + b - y^{(i)}) 2 \\
&= \frac{1}{m} \sum\limits_{i=1}^{m} (wx^{(i)} + b - y^{(i)}) \\
&= \frac{1}{m} \sum\limits_{i=1}^{m} ((f_{w,b}(x^{(i)}) - y^{(i)})
\end{align}
```

![](/img/1.1.3.5.gradient.descent.png) 

![](/img/1.1.3.5.local.minima.png) 

![](/img/1.1.3.5.convex.png) 
- a __convex function__ will have a _single global minimum_

##### C1_W1_M3_6 Running gradient descent

![](/img/1.1.3.6.plot.png) 
- left is _plot of the model_
- right is __contour plot__ of the cost function
- bottom is the __surface plot__ of the cost function
- for this example, `w = -0.1, b = 900`
- as we take each step we get closer to the global minimum
  - the yellow line is the __best line fit__
- Given a house with `1250 sq ft`, we can predict it should sell for `$250k` per the model

![](/img/1.1.3.6.batch.png) 
- __Batch Gradient Descent__ => each step of the gradient descent uses _all the training examples_
- DeepLearning.AI newsletter: _The Batch_

##### Lab: Gradient descent

- [Coursera: Optional lab: Gradient descent | Coursera](https://www.coursera.org/learn/machine-learning/ungradedLab/lE1al/optional-lab-gradient-descent/lab)
- [Lab: Cost Function](/code/C1_W1_Lab04_Gradient_Descent_Soln.ipynb)

##### Quiz: Train the Model with Gradient Descent

1. Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J.
```math
\begin{aligned}
\text{repeat until convergence \{} \\ 
  &w = w - \alpha \frac{\partial}{\partial w} J_{(w,b)}\\
  &b = b - \alpha \frac{\partial}{\partial b} J_{(w,b)}\\
\}
\end{aligned}
```
When $`\frac{\partial}{\partial w} J_{(w,b)}`$ is a negative number, what happens to `w` after one update step?
- [ ] It is not possible to tell is `w` will increase or decrease
- [ ] w increases
- [ ] w stays the same
- [ ] w decreases 

2. For linear regression, what is the update step for parameter b?
- [ ] $`b = b - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} ((f_{w,b}(x^{(i)}) - y^{(i)})`$
- [ ] $`b = b - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} ((f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)}`$

<details><summary>Ans</summary>2, 2</details>

### C1_W2: Regression with Multiple Input Variables

#### C1_W2_M1 Multiple Linear Regression

##### C1_W2_M1_1 Multiple features

##### C1_W2_M1_2 Vectorization part 1
##### C1_W2_M1_3 Vectorization part 2
##### Lab: Python, NumPy and vectorization
##### C1_W2_M1_4 Gradient descent for multiple linear regression
##### Lab: Muliple linear regression
#### C1_W2_M2 Gradient Descent In Practice
<!-- vim: set textwidth=0: -->
<!-- vim: set wrapmargin=0: -->
<!-- vim: set nowrap: -->
<!-- vim: set foldlevel=9: -->
