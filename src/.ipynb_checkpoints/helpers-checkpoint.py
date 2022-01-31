import matplotlib
import numpy as np
from scipy.optimize import minimize
np.random.seed(42)

heatmap_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("mycolormap",
    [
        (0 ,"#fdf094"), 
        (0.2, "#f4cc00"),
        (0.3, "#f06c04"),
        (0.5 , "#9e0000"),
        (0.9 , "#470000"), 
        (1   , "black")
    ]
)

season_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("mycolormap",
    [
        (0/4 ,"#3399ff"), 
        (3/12, "#66ff66"),
        (7/12, "#336600"),
        (10/12 , "#ff9900"),
        (12/12 , "#3399ff")
    ]
)

class ExpSigModel:
    
    def __init__(self, initial_parameters = [0.15, -0.004,  -0.2, -10], startvalue=0):
        self.guess = initial_parameters
        self.startvalue = startvalue
        self.params = []
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def _f(self, X, new_params):
        time = X[:, 0]
        temp = X[:, 1]

        a = new_params[0]
        b = new_params[1]
        c = new_params[2]
        d = new_params[3]

        return a * np.exp(b * (time - self.startvalue)) * self.sigmoid(c * (temp + d))

    def fit(self, X, y, verbose = True):
        
        res = minimize(
            lambda params: np.square(y[self.startvalue:] - self._f(X[self.startvalue:], params)).sum(),
            self.guess,
            method="Nelder-Mead",
            options={
                "maxiter": 10000
            }
        )
        params = res.x
        
        if verbose:
            print(res)
            
        self.params = params
        
    def predict(self, X):
        if len(self.params) == 0:
            raise RuntimeError("You should fit the model first")
        
        return self._f(X, self.params)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def evaluate(self, X, y):
        return np.square(y - self.predict(X)).sum()
    
    def to_latex(self):
        a = f"{self.params[0]:0.4f}"
        b = f"{self.params[1]:0.4f}"
        c = f"{self.params[2]:0.3f}"
        d = f"{self.params[3]:0.2f}"

        return f"{a} \\cdot e^{{ \\frac{{{b}}}{{d}} \\cdot (t - {self.startvalue}d) }} \\cdot  \\sigma \\left(\\frac{{{c}}}{{^\\circ C}} \cdot (\\theta {'+' if float(d) >= 0 else ''} {d}^\\circ C) \\right) \\cdot \\frac{{deaths}}{{cases}}"