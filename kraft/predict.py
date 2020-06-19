from sklearn.svm import SVC, SVR

from .CONSTANT import RANDOM_SEED


def train_and_classify(
    training_sample_x_feature,
    training_sample_class,
    testing_sample_x_feature,
    c=1.0,
    kernel="rbf",
    degree=3,
    gamma="auto",
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=int(2e2),
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape="ovr",
    random_seed=RANDOM_SEED,
):

    model = SVC(
        C=c,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        random_state=random_seed,
    )

    model.fit(training_sample_x_feature, training_sample_class)

    return model.predict(testing_sample_x_feature)


def train_and_regress(
    training_sample_x_feature,
    training_sample_class,
    testing_sample_x_feature,
    kernel="rbf",
    degree=3,
    gamma="auto",
    coef0=0.0,
    tol=1e-3,
    c=1.0,
    epsilon=0.1,
    shrinking=True,
    cache_size=int(2e2),
    verbose=False,
    max_iter=-1,
):

    model = SVR(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        tol=tol,
        C=c,
        epsilon=epsilon,
        shrinking=shrinking,
        cache_size=cache_size,
        verbose=verbose,
        max_iter=max_iter,
    )

    model.fit(training_sample_x_feature, training_sample_class)

    return model.predict(testing_sample_x_feature)
