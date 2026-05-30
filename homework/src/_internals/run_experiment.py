"""Ejecuta un experimento completo de entrenamiento para un estimador dado."""

from .calculate_metrics import calculate_metrics
from .prepare_data import prepare_data
from .print_metrics import print_metrics
from .save_model import save_model


def run_experiment(estimator):
    x_train, x_test, y_train, y_test = prepare_data()

    # entrenar el modelo
    estimator.fit(x_train, y_train)
    save_model(estimator)

    print()
    print(estimator, ":", sep="")

    mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
    print_metrics(mse, mae, r2, title="Metricas de entrenamiento:")

    mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
    print_metrics(mse, mae, r2, title="Metricas de testing:")
