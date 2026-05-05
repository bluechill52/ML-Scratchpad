import numpy as np


class LogisticRegression:
    """
    Binary logistic regression trained via full-batch gradient descent.
 
    Math recap:
        z       = X @ w + b                   (linear combination)
        p       = sigmoid(z)                  (predicted probability)
        loss    = BCE(y, p) + L2 penalty
        BCE     = -mean( y*log(p) + (1-y)*log(1-p) )
        L2      = (l2 / 2) * ||w||^2          (note: bias NOT regularized)
        dw      = (1/N) * X.T @ (p - y) + l2 * w
        db      = (1/N) * sum(p - y)
    """
 
    def __init__(self, lr: float = 0.1, epochs: int = 1000, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray = None
        self.b: float = 0.0
        self.loss_history: list[float] = []
 
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Clamp z to [-500, 500] to prevent overflow in exp()
        # sigmoid(-500) ≈ 0,  sigmoid(500) ≈ 1 — no meaningful loss of precision
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
 
    def _loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        # 1e-9 inside log prevents log(0) = -inf
        bce = -np.mean(
            y_true * np.log(y_prob + 1e-9) +
            (1 - y_true) * np.log(1 - y_prob + 1e-9)
        )
        # L2 penalty on weights only — bias is not penalized (standard practice)
        l2_penalty = (self.l2 / 2.0) * np.sum(self.w ** 2)
        return bce + l2_penalty
 
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        N, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        self.loss_history = []
 
        for iter in range(self.epochs):
            # Forward pass
            z = X @ self.w + self.b
            p = self.sigmoid(z)             # shape (N,)
 
            # Gradients
            error = p - y                   # shape (N,)
            dw = (1 / N) * (X.T @ error) + self.l2 * self.w
            db = (1 / N) * np.sum(error)
 
            # Gradient descent update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = self._loss(y, p)
            print(f'Iteration {iter} loss -> {loss}')
 
            self.loss_history.append(loss)
 
        return self
 
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of class 1. Shape: (N,)"""
        return self.sigmoid(X @ self.w + self.b)
 
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns hard binary labels {0, 1}. Shape: (N,)"""
        return (self.predict_proba(X) >= threshold).astype(int)