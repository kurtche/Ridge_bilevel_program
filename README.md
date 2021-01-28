# Ridge regression as a bilevel-optimization problem.
This code implements a ridge regression in a bilevel programming framework in which the hyper-parameter lambda minimizes the sum of square residuals (SSR) on a validation-set within the constraint that the regression parameters minimize the regularized SSR on a training-set. <br>
The solution is achieved through reverse-mode hyper-gradient computation described in https://arxiv.org/abs/1703.01785 followed by hyper-gradient descent. <br>
This implentation is intended as a toy example for hyper-gradient descent optimization and contains many information verbosity and methods to check the correctness of the computations. <br>
The code is written in python 3.7 and relies primarily on TensorFlow 2.4.

BLPPRidge is the core class of the code. You need first to initiate the BLPPRidge object whose arguments are the number of weights in the regression (excluding the intercept that is automatically included) and an optional string describing the initialization method for the variables, this can be one of ['zeros', 'ones', 'random']. Then, to start the estimation procedure, call the fit method of BLPPRidge whose arguments are the training-set, the validation-set, as tuples of the type (X, y), a dictionary of keras optimizers with 'inner' and 'outer' keys, the number of epochs (the number of hyper-gradient descent steps to perform) and the number of inner steps T. <br>
In this implementation, inner optimization must be performed through gradient descent, while the outer problem can be solved with any keras optimizer. <br>
Once the model is fitted, prediction can be done easily by calling the predict method with the new data as argument.

Be sure that the inner optimization reaches the convergence to correctly procede with the hyper-optimization!

Example of use:


```python
from tensorflow.keras.datasets import boston_housing
import scikit.preprocessing as sk_pp
import scikit.model_selection as sk_ms

from BLPP_ridge import BLPPRidge

# Import data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Validation split
x_train, x_val, y_train, y_val = sk_ms.train_test_split(
    x_train, y_train, test_size=0.2, random_state=46
)

# Standardize
x_scaler = sk_pp.StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
x_val = x_scaler.transform(x_val)

# BLPP Ridge
inner_opt = tf.keras.optimizers.SGD(learning_rate=4e-4)
outer_opt = tf.keras.optimizers.SGD(
    learning_rate=8e-3, momentum=0.95, nesterov=True
    )
optim = {"inner": inner_opt, "outer": outer_opt}

train_set = (x_train, y_train)
val_set = (x_val, y_val)

blpp_ridge = BLPPRidge(initializer="zeros", n_var=x_train.shape[1])
blpp_ridge.fit(train_set, val_set, optimizers=optim, epochs=300, T=800)

print()
print("BLPP regularizer:", blpp_ridge.hyper.numpy())
print("BLPP weights:", blpp_ridge.w.numpy())
print("BLPP intercept:", blpp_ridge.b.numpy())
mse = np.mean((y_train - blpp_ridge.predict(x_train)) ** 2)
print("BLPP train mse:", mse)
mse = np.mean((y_val - blpp_ridge.predict(x_val)) ** 2)
print("BLPP validation mse:", mse)
mse = np.mean((y_test - blpp_ridge.predict(x_test)) ** 2)
print("BLPP test mse:", mse)
```

You can also run this example, with an added comparison with the scikit-learn CV random search, by running the BLPP_ridge.py file from your terminal:

```bash
python BLPP_ridge.py
```
