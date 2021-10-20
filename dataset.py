import numpy as np
from scipy import stats

class Parameters:
    def __init__(self, p, ifnew, name):
        if (ifnew):
            self.v = np.random.normal(0, 1, size=[p, 3])
            for i in range(3):
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
        d_star = Sigmoid(np.abs(x.dot(v2))) / 2
        beta = self.compute_beta(d_star)
        for i in range(n):
            t[i][0] = np.random.beta(self.alpha, beta[i][0])
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        d_star = Sigmoid(np.abs(x.dot(v2))) / 2
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
        a = -np.abs(x.dot(v1))
        b = Sigmoid(np.abs(x.dot(v2)))
        c = x.dot(v3)
        return a * (t - b) * (t - b) + c

class SCIGAN3_Policy:
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
        for i in range(n):
            b = 0.75 * ((x[i:i + 1].dot(self.param.GetV(2))) / (x[i:i + 1].dot(self.param.GetV(3))))[0, 0]
            if (b >= 0.75):
                d_star = b / 3
            else:
                d_star = 1.0
            beta = self.compute_beta(d_star)
            t[i][0] = np.random.beta(self.alpha, beta)
            if (d_star <= 0.001):
                t[i][0] = 1 - t[i][0]
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        for i in range(n):
            b = 0.75 * ((x[i:i + 1].dot(self.param.GetV(2))) / (x[i:i + 1].dot(self.param.GetV(3))))[0, 0]
            if (b >= 0.75):
                d_star = b / 3
            else:
                d_star = 1.0
            beta = self.compute_beta(d_star)
            beta_prob = stats.beta(self.alpha, beta)
            if (d_star <= 0.001):
                pb[i][0] = beta_prob.pdf(1 - t[i][0])
            else:
                pb[i][0] = beta_prob.pdf(t[i][0])
        return pb

class SCIGAN3_Outcome:
    def __init__(self, param):
        self.param = param
    def BestTreatmentOutcome(self, x):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        b = 0.75 * (x.dot(v2)) / (x.dot(v3))
        d_star = b.copy()
        d_star[b >= 0.75] /= 3
        d_star[b < 0.75] = 1.0
        d_star[d_star > 1.0] = 1.0
        y = (x.dot(v1) + d_star * (d_star - b) * (d_star - b))
        return d_star, y
    def GetOutcome(self, x, t):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        b = 0.75 * (x.dot(v2)) / (x.dot(v3))
        y = (x.dot(v1) + t * (t - b) * (t - b))
        return y

class SCIGAN2_Policy:
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
            cof = (x[i:i + 1].dot(v2) / x[i:i + 1].dot(v3))[0, 0] * np.pi
            if (cof > 0):
                d_star = min(np.pi / (cof * 2), 1.0)
            else:
                if (cof < -np.pi * 3 / 2):
                    d_star = -np.pi * 3 / (cof * 2)
                elif (cof > -np.pi):
                    d_star = 0.0
                else:
                    d_star = 1.0
            beta = self.compute_beta(d_star)
            t[i][0] = np.random.beta(self.alpha, beta)
            if (d_star <= 0.001):
                t[i][0] = 1 - t[i][0]
        return t
    def GetProb(self, x, t):
        n = x.shape[0]
        pb = np.zeros([n, 1])
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        for i in range(n):
            cof = (x[i:i + 1].dot(v2) / x[i:i + 1].dot(v3))[0, 0] * np.pi
            if (cof > 0):
                d_star = min(np.pi / (cof * 2), 1.0)
            else:
                if (cof < -np.pi * 3 / 2):
                    d_star = -np.pi * 3 / (cof * 2)
                elif (cof > -np.pi):
                    d_star = 0.0
                else:
                    d_star = 1.0
            beta = self.compute_beta(d_star)
            beta_prob = stats.beta(self.alpha, beta)
            if (d_star <= 0.001):
                pb[i][0] = beta_prob.pdf(1 - t[i][0])
            else:
                pb[i][0] = beta_prob.pdf(t[i][0])
        return pb

class SCIGAN2_Outcome:
    def __init__(self, param):
        self.param = param
    def BestTreatmentOutcome(self, x):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        d_star = np.zeros([x.shape[0], 1])
        for i in range(x.shape[0]):
            cof = (x[i:i+1].dot(v2) / x[i:i+1].dot(v3))[0, 0] * np.pi
            if (cof > 0):
                d_star[i][0] = min(np.pi / (cof * 2), 1.0)
            else:
                if (cof < -np.pi * 3 / 2):
                    d_star[i][0] = -np.pi * 3 / (cof * 2)
                elif (cof > -np.pi):
                    d_star[i][0] = 0.0
                else:
                    d_star[i][0] = 1.0
        y = (x.dot(v1) + np.sin((x.dot(v2)) * d_star * np.pi / (x.dot(v3))))
        return d_star, y
    def GetOutcome(self, x, t):
        v1 = self.param.GetV(1)
        v2 = self.param.GetV(2)
        v3 = self.param.GetV(3)
        y = (x.dot(v1) + np.sin((x.dot(v2)) * t * np.pi / (x.dot(v3))))
        return y

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
            x = np.random.normal(0, 1, size=[n, p])
            alpha = 1.6
            behavior_policy = SCIGAN_Policy(alpha, param)
            outcome_model = SCIGAN_Outcome(param)
            t = behavior_policy.GetTreatment(x)
            print((t > 0.9).sum())
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
