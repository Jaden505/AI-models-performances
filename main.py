import models as m
import data_format as df
import model_performances as mp


def main():
    # Load data
    data = df.DataFormat()
    data.load_data()
    data.standard_scaler()
    data.one_hot_encoder()
    data.split_data()
    X_train, X_test, y_train, y_test = data.get_data()

    models = m.Model(X_train, X_test, y_train, y_test)

    evaluations = []
    for model in models.get_models():
        # Train models
        models.train_model(model)

        # Evaluate model
        models.evaluate(model)

        evaluations.append(models.evaluate_model_matrices(model))

    performances = mp.ModelPerformances(*evaluations)
    fig, ax1, ax2 = performances.create_fig()
    performances.bar_plot(ax1)
    performances.roc_plot(ax2)
    performances.display()

if __name__ == '__main__':
    main()
