from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # models
        self.dtc = DecisionTreeClassifier()
        self.rfc = GaussianNB()
        self.xgb = GradientBoostingClassifier()
        self.knn = KNeighborsClassifier()

    def get_models(self):
        return [self.dtc, self.rfc, self.xgb, self.knn]

    def train_model(self, model):
        hist = model.fit(self.X_train, self.y_train)
        return hist

    def evaluate(self, model):
        model_eval = self.evaluate_model_matrices(model)

        # Print result
        print('Accuracy:', model_eval['acc'])
        print('Precision:', model_eval['prec'])
        print('Recall:', model_eval['rec'])
        print('F1 Score:', model_eval['f1'])
        print('Cohens Kappa Score:', model_eval['kappa'])
        print('Area Under Curve:', model_eval['auc'])
        print('Confusion Matrix:\n', model_eval['cm'])

    def evaluate_model_matrices(self, model):
        # Predict Test Data
        y_pred = model.predict(self.X_test)

        # Calculate accuracy, precision, recall, f1-score, and kappa score
        acc = metrics.accuracy_score(self.y_test, y_pred)
        prec = metrics.precision_score(self.y_test, y_pred)
        rec = metrics.recall_score(self.y_test, y_pred)
        f1 = metrics.f1_score(self.y_test, y_pred)
        kappa = metrics.cohen_kappa_score(self.y_test, y_pred)

        # Calculate area under curve (AUC)
        y_pred_proba = model.predict_proba(self.X_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test, y_pred_proba)
        auc = metrics.roc_auc_score(self.y_test, y_pred_proba)

        # Display confussion matrix
        cm = metrics.confusion_matrix(self.y_test, y_pred)

        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
                'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}