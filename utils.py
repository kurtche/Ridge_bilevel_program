import tensorflow as tf


# Hessian vector products
def forward_over_back_hvp(params, X, y, loss_fun, vector):
    inner_vars, outer_vars = params
    with tf.autodiff.ForwardAccumulator(outer_vars, vector) as acc:
        with tf.GradientTape() as grad_tape:
            loss = loss_fun(X, y)
        grads = grad_tape.gradient(loss, inner_vars)
        print(grads)
    hvp = acc.jvp(grads)
    return hvp


def back_over_forward_hvp(X, y, loss_fun, inner_vars, outer_vars, vector):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(outer_vars)
        with tf.autodiff.ForwardAccumulator(inner_vars, vector) as acc:
            loss = loss_fun(X, y)
    hvp = grad_tape.gradient(acc.jvp(loss), outer_vars)
    return hvp


def back_over_back_hvp(X, y, loss_fun, inner_vars, outer_vars, vector):
    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = loss_fun(X, y)
        inner_grads = inner_tape.gradient(loss, inner_vars)
    hvp = outer_tape.gradient(inner_grads, outer_vars, output_gradients=vector)
    del outer_tape
    return hvp


class ColorPrint:
    """ ANSI color codes """

    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    YELLOW = "\033[1;33m"
    BOLD = "\033[1m"
    END = "\033[0m"
