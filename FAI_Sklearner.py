import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import validation_curve, cross_val_score
from sklearn.metrics import mean_squared_error


class FAI_Sklearner:
    """docstring for FAI_Sklearner"""

    def __init__(self,alpha=0.0001, learning_rate=1e-4, iterations=1000, loss='squared_error', model='sgdregressor'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.zeros((1, 4))
        self.bias = np.zeros((1, 4))
        self.val_model = self.getModel(alpha, learning_rate, iterations, loss, model)
        self.ar_model = self.getModel(alpha, learning_rate, iterations, loss, model)

    def getModel(self, alpha, learning_rate, iterations, loss, model):
        models = {
            'Sgdregressor': SGDRegressor(alpha=alpha, max_iter=iterations),
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=alpha),
            'Lasso': Lasso(alpha=alpha),
            'ElasticNet': ElasticNet(alpha=alpha),
        }
        return models[model]

    def forward_pass(self, X, Y):
        pass
    def backward_pass(self, X, y):
        pass


    def train(self, X_valence, X_arousal, Y_valence, Y_arousal):
        self.val_model.fit(X_valence, Y_valence)
        self.ar_model.fit(X_arousal, Y_arousal)
        self.weights = np.array((self.val_model.coef_, self.ar_model.coef_))
        self.bias = np.array((self.val_model.intercept_, self.ar_model.intercept_))
        print("final weights: ", self.weights, 'biases:', self.bias)


    def validate(self, X_valence, X_arousal, Y_valence, Y_arousal):
        result = np.zeros((X_valence.shape[0],2))
        result[:,0] = self.val_model.predict(X_valence)
        result[:,1] = self.ar_model.predict(X_arousal)
        total_loss_valence = mean_squared_error( Y_valence, result[:,0]) # np.fabs((result[:,0] - Y_valence)).mean()
        total_loss_arousal = mean_squared_error( Y_arousal, result[:,1]) # np.fabs((result[:,1] - Y_arousal)).mean()
        print("total losses: ", total_loss_valence, total_loss_arousal)
        return result, total_loss_valence, total_loss_arousal

    def predict(self, X_valence, X_arousal):
        result = np.zeros((X_valence.shape[0], 2))
        result[:, 0] = self.val_model.predict(X_valence)
        result[:, 1] = self.ar_model.predict(X_arousal)
        return result

    def cross_validate(self, X_valence, X_arousal, Y_valence, Y_arousal):
        # 5-fold cross-validation
        v_cv_scores = cross_val_score(
            self.val_model, X_valence, Y_valence,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        a_cv_scores = cross_val_score(
            self.ar_model, X_arousal, Y_arousal,
            cv=5,
            scoring='neg_mean_squared_error'
        )

        print(f"Valence CV MSE: {-v_cv_scores.mean():.6f} (+/- {v_cv_scores.std():.6f})")
        print(f"Arousal CV MSE: {-a_cv_scores.mean():.6f} (+/- {a_cv_scores.std():.6f})")