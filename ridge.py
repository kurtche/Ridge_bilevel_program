import tensorflow as tf
import numpy as np

ALPHA = 1.50001


def exact_solution(X, y):
    ridge_weights = np.dot(
        np.linalg.inv(np.dot(X.T, X) + ALPHA * np.eye(X.shape[-1])),
        np.dot(X.T, y - np.mean(y)),
    )
    return ridge_weights


W_STAR = 0


class BLPPRidge:
    def __init__(self, n_var, initializer="random"):

        if initializer == "ones":
            weights_init = hyper_init = tf.ones_initializer()
        elif initializer == "zeros":
            weights_init = hyper_init = tf.zeros_initializer()
        elif initializer == "random":
            weights_init = tf.random_normal_initializer(mean=0, stddev=2)
            hyper_init = tf.random_uniform_initializer(minval=1e-2, maxval=5.0)

        self.w = tf.Variable(weights_init(shape=[n_var]))
        self.b = tf.Variable([0.0], trainable=True)
        self.sigma = tf.Variable([1.0], trainable=False)

    def predict(self, X):
        return tf.reduce_sum(X * self.w, 1) + self.b

    # Base losses
    def ssr(self, y, y_hat):
        return tf.reduce_sum((y - y_hat) ** 2)

    def mse(self, y, y_hat):
        return tf.reduce_mean((y - y_hat) ** 2)

    # Regularization
    def ridge_reg(self):
        return ALPHA * tf.reduce_sum(self.w ** 2)

    # Regularized loss
    def ridge_loss(self, X, y):
        y_hat = self.predict(X)
        return self.ssr(y, y_hat) + self.ridge_reg()

    def val_loss(self, X_train, y_train, X_val, y_val):
        weights = self.exact_weights(X_train, y_train)
        y_hat = tf.reduce_sum(X_val * weights, 1) + self.b
        loss = tf.reduce_sum((y_val - y_hat) ** 2)
        return loss

    def exact_weights(self, X, y):
        weights = tf.tensordot(
            tf.linalg.inv(tf.tensordot(X, X, [[0], [0]]) + ALPHA * tf.eye(X.shape[-1])),
            tf.tensordot(X, y - self.b, [[0], [0]]),
            [[1], [0]],
        )
        return weights

    # Parameters optimization
    def _inner_optim(self, X, y, T, inner_opt):
        all_vars = []
        for t in range(T):
            with tf.GradientTape() as tape:
                train_loss = self.ridge_loss(X, y)
            train_grads = tape.gradient(train_loss, [self.w, self.b])
            inner_opt.apply_gradients(zip(train_grads, [self.w, self.b]))
            v1 = np.linalg.norm(self.w.read_value() - W_STAR)
            print(f"{t}, Train loss: {train_loss}, {v1}")
        return all_vars

    def fit(self, X_train, y_train, optimizers, T=200):
        # self.b.assign([tf.reduce_mean(y_train.astype("float32"))])
        X_train = X_train.astype("float32")
        y_train = y_train.astype("float32")

        inner_opt = optimizers["inner"]

        self._inner_optim(X_train, y_train, T, inner_opt)


if __name__ == "__main__":
    from tensorflow.keras.datasets import boston_housing
    import sklearn.preprocessing as sk_pp
    import sklearn.linear_model as sk_lm
    import sklearn.model_selection as sk_ms
    import numpy as np

    # Import data
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # Validation split
    x_train, x_val, y_train, y_val = sk_ms.train_test_split(
        x_train, y_train, test_size=0.3, random_state=46
    )

    # Standardize
    x_scaler = sk_pp.StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    x_val = x_scaler.transform(x_val)

    W_STAR = exact_solution(x_train, y_train)
    # BLPP Ridge
    inner_opt = tf.keras.optimizers.SGD(
        learning_rate=1e-4  # , momentum=0.95, nesterov=True
    )
    optim = {"inner": inner_opt}

    blpp_ridge = BLPPRidge(initializer="zeros", n_var=x_train.shape[1])
    blpp_ridge.fit(x_train, y_train, optimizers=optim, T=4000)
    print()
    print("BLPP weights:", blpp_ridge.w.numpy())
    print("BLPP intercept:", blpp_ridge.b.numpy()[0])
    mse = np.mean((y_train - blpp_ridge.predict(x_train)) ** 2)
    print("BLPP train mse:", mse)
    mse = np.mean((y_val - blpp_ridge.predict(x_val)) ** 2)
    print("BLPP validation mse:", mse)
    mse = np.mean((y_test - blpp_ridge.predict(x_test)) ** 2)
    print("BLPP test mse:", mse)

    print("")
    print("Exact weights:", W_STAR)
    print("Y mean:", y_train.mean())
