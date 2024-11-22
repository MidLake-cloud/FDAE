from sklearn import svm


class SVMModel:
    def __init__(self, C_value, gamma_value, use_probability, class_weights, multi_mode, kernel='rbf') -> None:
        self.model = svm.SVC(gamma='auto', decision_function_shape='ovo', probability=use_probability)
    
    def fit(self, featuers, labels):
        self.model.fit(X=featuers, y=labels)

    def predict(self, features):
        return self.model.predict(features)

    def predict_prob(self, features):
        return self.model.predict_proba(features)
    
    def decision_function(self, test_X):
        return self.model.decision_function(test_X)