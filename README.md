# ML Bootcamp Project

## Anant Upadhyay, IIT (ISM) Dhanbad

This repository contains implementations of various machine learning algorithms as part of the ML Bootcamp at IIT (ISM) Dhanbad.

## Project Details

This project is my submission for WOC 5.0 in the Machine Learning division. The training data used to train the models are stored in the `Train_data` folder, and the data on which predictions are made is stored in the `Test_data` folder. The final trained parameters for all the models are stored in `.npy` and `.npz` formats under the `Trained_Model_Parameters` folder to save time during model loading.

## Algorithms Implemented

### Linear Regression
- **Model**: `f(w,b,x(i))= wx(i)+b`
- **Cost Function**: Mean Squared Error
  $`
  J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2
  `$
- **Training**: Gradient Descent
  $`
  w=w-\alpha \frac{\partial J}{\partial w}, \quad b=b-\alpha \frac{\partial J}{\partial b}
  `$

### Polynomial Regression
- **Model**: 
  $`
  f(w,b,x(i))= w_1x(i)+w_2(x(i))^2+w_3(x(i))^3+\cdots+b
  `$
- **Cost Function**: Similar to Linear Regression
- **Training**: Gradient Descent
  $`
  w=w-\alpha \frac{\partial J}{\partial w}, \quad b=b-\alpha \frac{\partial J}{\partial b}
  `$

### Logistic Regression
- **Model**: Sigmoid function for binary classification, Softmax for multi-class
  $`
f_{w,b}(x^{(i)}) = \frac{e^{w_i(x) + b_i}}{\sum_{k=1}^{n} e^{w_k(x) + b_k}}
  `$
- **Loss Function**: Cross-Entropy
  $`
  L(f_{w,b}(x^{(i)}), y^{(i)}) = -\ln(f_{w,b}(x^{(i)})_j)  \quad \text{where} \quad j=Y[i]
  `$
- **Training**: Gradient Descent
  $`
  w=w-\alpha \frac{\partial J}{\partial w}, \quad b=b-\alpha \frac{\partial J}{\partial b}
  `$

### Neural Networks
- **Architecture**: Multi-layer perceptron with ReLU activation for hidden layers and problem-specific activation for output layer.
- **Forward Propagation**:
  $`
  a_j^{[L]}=g(w_j^{[L]}a^{[L-1]}+b_j^{[L]})
  `$
- **Training**: Backpropagation
  $`
  w=w-\alpha \frac{\partial J}{\partial w}, \quad b=b-\alpha \frac{\partial J}{\partial b}
  `$

### K-Nearest Neighbors (KNN)
- **Distance Metric**: Euclidean distance (squared)
  $`
  d(x,y)=\sum_{i=1}^{m}(x_i-y_i)^2
  `$
- **Classification**: Majority vote among k-nearest neighbors
- **Regression**: Average of k-nearest neighbors

## Technology Stack
- Python
- Numpy
- Pandas
- Matplotlib
- Jupyter/Google Colab Notebooks

## Development Process
1. **Data Splitting**: Training data split into `J_train` and `J_cv` to address underfitting and overfitting.
2. **Feature Engineering**: Normalization using Z-score
   $`
   z=\frac{x-\mu}{\sigma}
   `$
3. **Regularization**: Added to cost function to prevent overfitting
   $` 
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 
  `$

## Results
- **Linear Regression**: 84.29%
- **Polynomial Regression**: 99.99999999984027%
- **Logistic Regression**: 87.47%
- **KNN**: 83.33%
- **Neural Network (Linear)**: 84.29%
- **Neural Network (Polynomial)**: 99.995%
- **Neural Network (Classification)**: 85.6%

## References
- [Coursera](https://www.coursera.org)
- [Numpy Documentation](https://numpy.org/doc/stable/reference/)
- [Pandas Documentation](https://pandas.pydata.org/docs/reference/api/)
- [GeeksforGeeks](https://www.geeksforgeeks.org/)
- [Analytics Vidhya](https://www.analyticsvidhya.com/)
- [Medium](https://medium.com/)
