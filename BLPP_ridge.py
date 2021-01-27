import tensorflow as tf
import numpy as np
import time

from utils import ColorPrint


def true_weights(X, y, alpha):
    X = X.astype("float32")
    y = y.astype("float32")
    weights = tf.tensordot(
        tf.linalg.inv(tf.tensordot(X, X, [[0], [0]]) + alpha * tf.eye(X.shape[-1])),
        tf.tensordot(X, y - y.mean(), [[0], [0]]),
        [[1], [0]],
    )
    return weights


# Ridge Regression as Bilevel Programming Problem for hyperparameter optimization
class BLPPRidge:
    def __init__(self, n_var, initializer="zeros"):

        if initializer == "ones":
            weights_init = hyper_init = tf.ones_initializer()
        elif initializer == "zeros":
            weights_init = hyper_init = tf.zeros_initializer()
        elif initializer == "random":
            weights_init = tf.random_normal_initializer(mean=0.0, stddev=1.0)
            hyper_init = tf.random_uniform_initializer(minval=0.0, maxval=1.0)

        self.hyper = tf.Variable(
            hyper_init(shape=[1]), constraint=tf.keras.constraints.NonNeg()
        )
        self.w = tf.Variable(weights_init(shape=[n_var]))
        self.b = tf.Variable(weights_init(shape=[1]), trainable=False)

    def predict(self, X):
        return tf.reduce_sum(X * self.w, 1) + self.b

    # Base losses
    def ssr(self, y, y_hat):
        return tf.reduce_sum((y - y_hat) ** 2)

    def mse(self, y, y_hat):
        return tf.reduce_mean((y - y_hat) ** 2)

    # Regularization
    @property
    def ridge_reg(self):
        return self.hyper * tf.reduce_sum(self.w ** 2)

    # Complete loss
    def ridge_loss(self, X, y):
        y_hat = self.predict(X)
        return self.ssr(y, y_hat) + self.ridge_reg

    # Loss with known optimal weights from training-set
    def true_val_loss(self, X_train, y_train, X_val, y_val):
        weights = true_weights(X_train, y_train, self.hyper)
        y_hat = tf.reduce_sum(X_val * weights, 1) + tf.reduce_mean(y_train)
        loss = tf.reduce_sum((y_val - y_hat) ** 2)
        return loss

    # Hyper-gradient as definition
    def true_hyper_grad(self, X_train, y_train, X_val, y_val, h=1e-2):
        self.hyper.assign(self.hyper.read_value() + h)
        fxh = self.true_val_loss(X_train, y_train, X_val, y_val)

        self.hyper.assign(self.hyper.read_value() - h)
        fx = self.true_val_loss(X_train, y_train, X_val, y_val)

        return (fxh - fx) / h

    # Hyper-gradient from manually solved static BLPP
    def true_hyper_grad2(self, X_train, y_train, X_val, y_val):
        a = tf.tensordot(
            X_val, y_val - tf.reduce_mean(y_train), [[0], [0]]
        ) - tf.tensordot(
            tf.tensordot(X_val, X_val, [[0], [0]]),
            true_weights(X_train, y_train, self.hyper),
            [[1], [0]],
        )
        b = tf.linalg.inv(
            tf.tensordot(X_train, X_train, [[0], [0]])
            + self.hyper * tf.eye(X_train.shape[1])
        )
        alpha = tf.tensordot(a, b, [[0], [0]])
        return 2 * tf.tensordot(
            true_weights(X_train, y_train, self.hyper), alpha, [[0], [0]]
        )

    # Dynamic BLPP
    # Inner optimization loop
    def _inner_optim(self, X, y, T, inner_opt):
        all_vars = [[self.w.read_value()]]
        all_grads = []
        all_tapes = []
        for t in range(T):
            # Two tf.GradientTape() are needed to record operations of the inner-gradient
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    train_loss = self.ridge_loss(X, y)
                train_grads = inner_tape.gradient(train_loss, [self.w])
            inner_opt.apply_gradients(zip(train_grads, [self.w]))

            all_vars.append([self.w.read_value()])
            all_grads.append(train_grads)
            all_tapes.append(outer_tape)
            del outer_tape

        # Verify convergence of inner-loop
        v1 = np.linalg.norm(self.w - true_weights(X, y, self.hyper))
        print(f"Train loss: {train_loss[0]}, Diff. from exact weights: {v1} ")
        return [all_vars, all_grads, all_tapes]

    # Outer step with reverse loop
    def _outer_optim(
        self, inner_dyn, X_train, y_train, X_val, y_val, outer_opt, ground_lr
    ):
        all_vars, all_grads, all_tapes = inner_dyn
        T = len(all_vars)
        # Initialize alpha_t and hyper-gradient (reverse-step T)
        with tf.GradientTape() as tape:
            y_hat = self.predict(X_val)
            val_loss = self.ssr(y_val, y_hat)
        alpha_t = tape.gradient(val_loss, [self.w])
        hyper_grad = [0.0]
        print(f"Val loss: {val_loss}, Lambda: {self.hyper.numpy()}")

        for t in range(T - 2, -1, -1):
            self.w.assign(all_vars[t][0])
            # self.b.assign(all_vars[t][1])

            # Hessian vector product
            hvp = all_tapes[t].gradient(
                all_grads[t], [self.w, self.hyper], output_gradients=alpha_t
            )
            # Hyper-gradient update (reverse-step t)
            hyper_grad = [
                hyper_grad[i - len(alpha_t)] - ground_lr * hvp[i]
                for i in range(len(alpha_t), len(hvp))
            ]
            # alpha_t update
            alpha_t = [alpha_t[i] - ground_lr * hvp[i] for i in range(len(alpha_t))]

        # Verify hyper-gradient computation
        true_hg = self.true_hyper_grad(X_train, y_train, X_val, y_val)
        true_hg2 = self.true_hyper_grad2(X_train, y_train, X_val, y_val)
        print("Hyper grad:", hyper_grad[0].numpy())
        print(f"True hyper grad: {true_hg}, {true_hg2}")

        # Hyper-parameter update
        outer_opt.apply_gradients(zip(hyper_grad, [self.hyper]))

    # Exact hyper-parameter optimization with known optimal inner weights
    def _exact_optim(self, X_train, y_train, X_val, y_val, outer_opt):
        with tf.GradientTape() as tape:
            val_loss = self.true_val_loss(X_train, y_train, X_val, y_val)
        hyper_grad = tape.gradient(val_loss, [self.hyper])
        print(f"Val loss: {val_loss}")
        print(f"Lambda: {self.hyper.numpy()}, hypergrad: {hyper_grad[0].numpy()}")
        true_hg = self.true_hyper_grad(X_train, y_train, X_val, y_val)
        true_hg2 = self.true_hyper_grad2(X_train, y_train, X_val, y_val)
        print(f"True hyper grad: {true_hg}, {true_hg2}")
        outer_opt.apply_gradients(zip(hyper_grad, [self.hyper]))

    # BLPP optimization with hand-computed gradients
    # derivative of inner optimization step wrt the weights
    def _exact_B(self, inner_lr):
        return 2 * inner_lr * self.w

    # derivative of inner optimization step wrt the hyperparameter
    def _exact_A(self, X, inner_lr):
        return tf.eye(X.shape[1]) - 2 * inner_lr * (
            tf.tensordot(X, X, [[0], [0]]) + self.hyper * tf.eye(X.shape[1])
        )

    def _exact_BLPP_outer(
        self, inner_dyn, X_train, y_train, X_val, y_val, outer_opt, ground_lr
    ):
        all_vars, all_grads, all_tapes = inner_dyn
        T = len(all_vars)

        y_hat = self.predict(X_val)
        val_loss = self.ssr(y_val, y_hat)
        # derivative of the validation loss wrt the weights
        alpha_t = 2 * (
            tf.tensordot(tf.tensordot(X_val, X_val, [[0], [0]]), self.w, [[1], [0]])
            - tf.tensordot(X_val, y_val - tf.reduce_mean(y_train), [[0], [0]])
        )
        hyper_grad = [0.0]
        print(f"Val loss: {val_loss}, lambda: {self.hyper.numpy()}")

        for t in range(T - 2, -1, -1):
            self.w.assign(all_vars[t][0])

            hyper_grad = hyper_grad - tf.tensordot(
                alpha_t, self._exact_B(ground_lr), [[0], [0]]
            )
            alpha_t = tf.tensordot(
                alpha_t, self._exact_A(X_train, ground_lr), [[0], [0]]
            )

        # Verify hyper-gradient computation
        true_hg = self.true_hyper_grad(X_train, y_train, X_val, y_val)
        true_hg2 = self.true_hyper_grad2(X_train, y_train, X_val, y_val)
        print(f"Hyper grad: {hyper_grad}")
        print(f"True hyper grad: {true_hg}, {true_hg2}")

        outer_opt.apply_gradients(zip([hyper_grad], [self.hyper]))

    # Generic fit function for hyper-parameter optimization
    def fit(self, train_set, val_set, optimizers=None, epochs=300, T=800):
        X_train, y_train = train_set
        X_val, y_val = val_set
        X_train = X_train.astype("float32")
        X_val = X_val.astype("float32")
        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")

        self.b.assign([tf.reduce_mean(y_train)])

        # Check shape of validation and training sets
        assert (
            X_train.shape[-1] == X_val.shape[-1]
        ), "Number of variables in trainining \
                and validation sets must be equal"
        # if only one independent variable, reshape data
        if len(X_train.shape) == 1:
            X_train = tf.reshape(X_train, [X_train.shape[0], 1])
            X_val = tf.reshape(X_val, [X_val.shape[0], 1])

        # Check optimizers, if None assign defaults
        if optimizers is not None:
            try:
                inner_opt = optimizers["inner"]
                outer_opt = optimizers["outer"]
            except KeyError:
                raise KeyError(
                    "optimizers must be a dictionary of tf.keras.Optimizer\
                                objects with 'inner' and 'outer' keys"
                )
        else:
            inner_opt = tf.keras.optimizers.SGD(learning_rate=4e-4)
            outer_opt = tf.keras.optimizers.SGD(learning_rate=5e-3)

        gr_lr = inner_opt.learning_rate
        for i in range(epochs):
            t0 = time.time()
            print(f"{ColorPrint.BLUE}Epoch: {i + 1}{ColorPrint.END}")
            # self.w.assign(self.true_weights(X_train, y_train))
            inner_dyn = self._inner_optim(X_train, y_train, T, inner_opt)
            # self._exact_optim(X_train, y_train, X_val, y_val, outer_opt)
            if epochs > 1:
                self._outer_optim(
                    inner_dyn, X_train, y_train, X_val, y_val, outer_opt, gr_lr
                )
            d_t = round(time.time() - t0, 4)
            print(f"Time elapsed: {d_t}")
        # self.w.assign(self.true_weights(X_train, y_train))
        _ = self._inner_optim(X_train, y_train, T, inner_opt)


if __name__ == "__main__":
    from tensorflow.keras.datasets import boston_housing
    import sklearn.preprocessing as sk_pp
    import sklearn.linear_model as sk_lm
    import sklearn.model_selection as sk_ms

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
    print("BLPP regularizer:", blpp_ridge.hyper.numpy()[0])
    print("BLPP weights:", blpp_ridge.w.numpy())
    print("BLPP intercept:", blpp_ridge.b.numpy()[0])
    mse = np.mean((y_train - blpp_ridge.predict(x_train)) ** 2)
    print("BLPP train mse:", mse)
    mse = np.mean((y_val - blpp_ridge.predict(x_val)) ** 2)
    print("BLPP validation mse:", mse)
    mse = np.mean((y_test - blpp_ridge.predict(x_test)) ** 2)
    print("BLPP test mse:", mse)

    # Sklearn Regression
    alphas = np.random.uniform(0, 150, 500)
    sk_fit = sk_lm.RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        cv=4,
    )
    sk_fit.fit(x_train, y_train)
    print()
    print("Sklearn CV regularizer:", sk_fit.alpha_)
    print("Sklearn CV weights:", sk_fit.coef_)
    print("Sklearn CV intercept:", sk_fit.intercept_)
    mse = np.mean((y_train - sk_fit.predict(x_train)) ** 2)
    print("Sklearn CV train mse:", mse)
    mse = np.mean((y_val - sk_fit.predict(x_val)) ** 2)
    print("Sklearn CV validation mse:", mse)
    mse = np.mean((y_test - sk_fit.predict(x_test)) ** 2)
    print("Sklearn CV test mse:", mse)

    # Exact weights from closed ridge formula
    exact_weights = true_weights(x_train, y_train, blpp_ridge.hyper)
    print()
    print("Exact weights:", exact_weights.numpy())
