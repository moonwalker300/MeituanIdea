import numpy as np
from scipy import stats

class Parameters:
    def __init__(self, p, ifnew, name):
        if (ifnew):
            self.v = np.random.uniform(0, 1, size=[p, 2])
            for i in range(2):
                self.v[:, i] /= np.linalg.norm(self.v[:, i], ord=2)
            np.save(name + 'v.npy', self.v)
        else:
            self.v = np.load(name + 'v.npy')
    def GetV(self, idx):
        return self.v[:, idx - 1 : idx]

class Uniform_Policy:
    def __init__(self):
        pass
    def GetTreatment(self, x):
        n = x.shape[0]
        t = np.random.rand(n, 1)
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.ones([n, 1])
        return pb

def Sigmoid(x):
    a = x.copy()
    b = -np.abs(x)
    a[a > 0] = 0 
    return np.exp(a) / (1 + np.exp(b))

class SCIGAN_Policy:
    def __init__(self, alpha, param):
        self.alpha = alpha
        self.param = param
    def compute_beta(self, d):
        beta = (self.alpha - 1) / d + 2 - self.alpha
        return beta
    def GetTreatment(self, x):
        n = x.shape[0]
        t = np.zeros([n, 1])
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        d_star = Sigmoid(np.abs(x.dot(v2)))
        beta = self.compute_beta(d_star)
        for i in range(n):
            t[i][0] = np.random.beta(self.alpha, beta[i][0])
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        d_star = Sigmoid(np.abs(x.dot(v2)))
        for i in range(n):
            beta = self.compute_beta(d_star[i][0])
            beta_prob = stats.beta(self.alpha, beta)
            pb[i][0] = beta_prob.pdf(t[i][0])
        return pb

class SCIGAN_Outcome:
    def __init__(self, param):
        self.param = param
    def BestTreatmentOutcome(self, x):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        d_star = Sigmoid(np.abs(x.dot(v2)))
        y = self.GetOutcome(x, d_star)
        return d_star, y
    def GetOutcome(self, x, t):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        a = -np.abs(x.dot(v1)) * 2 - 4
        b = Sigmoid(np.abs(x.dot(v2)))
        c = x.dot(v3)

class Linear_Policy:
    def __init__(self, alpha, param):
        self.alpha = alpha
        self.param = param
    def compute_beta(self, d):
        beta = (self.alpha - 1) / d + 2 - self.alpha
        return beta
    def GetTreatment(self, x):
        n = x.shape[0]
        t = np.zeros([n, 1])
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1) * 2
        b = np.min(-x.dot(v2) * 2 + 1, -0.01)
        d_star = (a / (-2 * b)) / 2
        beta = self.compute_beta(d_star)
        for i in range(n):
            t[i][0] = np.random.beta(self.alpha, beta[i][0])
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1) * 2
        b = np.min(-x.dot(v2) * 2 + 1, -0.01)
        d_star = (a / (-2 * b)) / 2
        for i in range(n):
            beta = self.compute_beta(d_star[i][0])
            beta_prob = stats.beta(self.alpha, beta)
            pb[i][0] = beta_prob.pdf(t[i][0])
        return pb

class Linear_Outcome:
    def __init__(self, param):
        self.param = param
    def BestTreatmentOutcome(self, x):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1) * 2
        b = np.min(-x.dot(v2) * 2 + 1, -0.01)
        d_star = a / (-2 * b)
        y = self.GetOutcome(x, d_star)
        return d_star, y
    def GetOutcome(self, x, t):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1) * 2
        b = np.min(-x.dot(v2) * 2 + 1, -0.01)
        return a * t + b * t * t


class Exp_Policy:
    def __init__(self, alpha, param):
        self.alpha = alpha
        self.param = param
    def compute_beta(self, d):
        beta = (self.alpha - 1) / d + 2 - self.alpha
        return beta
    def GetTreatment(self, x):
        n = x.shape[0]
        t = np.zeros([n, 1])
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1)
        b = x.dot(v2)
        d_star = (1 / b) / 2
        beta = self.compute_beta(d_star)
        for i in range(n):
            t[i][0] = np.random.beta(self.alpha, beta[i][0])
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1)
        b = x.dot(v2)
        d_star = (1 / b) / 2
        for i in range(n):
            beta = self.compute_beta(d_star[i][0])
            beta_prob = stats.beta(self.alpha, beta)
            pb[i][0] = beta_prob.pdf(t[i][0])
        return pb

class Exp_Outcome:
    def __init__(self, param):
        self.param = param
    def BestTreatmentOutcome(self, x):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1)
        b = x.dot(v2)
        d_star = 1 / b
        y = self.GetOutcome(x, d_star)
        return d_star, y
    def GetOutcome(self, x, t):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        a = x.dot(v1)
        b = x.dot(v2)
        return np.exp(a - b * t) * t

class MT_Policy:
    def __init__(self, alpha, param):
        self.alpha = alpha
        self.param = param
    def compute_beta(self, d):
        if ((d <= 0.001) or (d >= 1.0)):
            beta = 1.0
        else:
            beta = (self.alpha - 1) / d + 2 - self.alpha
        return beta
    def GetTreatment(self, x):
        n = x.shape[0]
        t = np.zeros([n, 1])
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        for i in range(n):
            phase = Sigmoid(x[i:i + 1].dot(v2))[0, 0] / 2
            d_star = 7 / 8 - phase
            beta = self.compute_beta(d_star)
            t[i][0] = np.random.beta(self.alpha, beta)
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        for i in range(n):
            phase = Sigmoid(x[i:i + 1].dot(v2))[0, 0] / 2
            d_star = 7 / 8 - phase
            beta = self.compute_beta(d_star)
            beta_prob = stats.beta(self.alpha, beta)
            pb[i][0] = beta_prob.pdf(t[i][0])
        return pb

class MT_Outcome:
    def __init__(self, param):
        self.param = param
    def BestTreatmentOutcome(self, x):
        y = np.zeros([x.shape[0], 0])
        resolution = 1000
        for i in range(resolution + 1):
            d_tmp = np.ones([x.shape[0], 1]) / resolution * i
            y_tmp = self.GetOutcome(x, d_tmp)
            y = np.concatenate([y, y_tmp], axis = 1)
        y_star = np.max(y, axis = 1, keepdims = True)
        d_star = np.argmax(y, axis = 1) * 1.0 / resolution
        d_star = np.expand_dims(d_star, axis = 1)
        return d_star, y_star
    def GetOutcome(self, x, t):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        phase = Sigmoid(x.dot(v2)) / 2
        y = (np.abs(x.dot(v1)) + 0.5) * t + np.sin(4 * (t + phase) * np.pi)
        return y

class Dataset:
    def __init__(self, n, p, param, ifnew, name):
        if (ifnew):
            np.random.seed(0)
            x = np.random.uniform(0, 1, size=[n, p]) 
            alpha = 3.0
            behavior_policy = Linear_Policy(alpha, param)
            outcome_model = Linear_Outcome(param)
            t = behavior_policy.GetTreatment(x)
            print(((t > 0.2) & (t < 0.3)).sum(), (t < 0.1).sum())
            y = outcome_model.GetOutcome(x, t)
            y += np.random.normal(0, 0.2, size = y.shape)
            ps = behavior_policy.GetProb(x, t)
            np.save(name + 'x.npy', x)
            np.save(name + 't.npy', t)
            np.save(name + 'y.npy', y)
            np.save(name + 'ps.npy', ps)
        else:
            x = np.load(name + 'x.npy')
            t = np.load(name + 't.npy')
            y = np.load(name + 'y.npy')
            ps = np.load(name + 'ps.npy')

        self.x = x
        self.t = t
        self.y = y
        self.ps = ps
    def GetData(self):
        return self.x, self.t, self.y, self.ps
