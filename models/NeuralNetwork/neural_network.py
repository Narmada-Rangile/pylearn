import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time

#region: Utilities

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.astype(int)]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def macro_f1_score(y_true, y_pred):
    unique_classes = np.unique(y_true)
    f1_scores = []
    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0.0

def plot_history(history, title="Training History", filename="training_plot.png"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1')
    plt.plot(history['val_r2'], label='Val R2')
    plt.title(f'{title} - Metrics')
    plt.xlabel('Epochs'); plt.ylabel('Score'); plt.legend()
    plt.tight_layout(); plt.savefig(filename); plt.close()

#region: Initializers

class Initializer:
    @staticmethod
    def get(name, shape):
        fan_in, fan_out = shape
        if name == 'he_normal': return np.random.randn(*shape) * np.sqrt(2. / fan_in)
        elif name == 'he_uniform': return np.random.uniform(-np.sqrt(6./fan_in), np.sqrt(6./fan_in), shape)
        elif name == 'xavier_normal': return np.random.randn(*shape) * np.sqrt(2. / (fan_in + fan_out))
        elif name == 'xavier_uniform': return np.random.uniform(-np.sqrt(6./(fan_in+fan_out)), np.sqrt(6./(fan_in+fan_out)), shape)
        return np.random.randn(*shape) * 0.01

#region: Layers

class Layer:
    def forward(self, input_data, training=True): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, l1=0.0, l2=0.0, initializer='he_normal'):
        self.weights = Initializer.get(initializer, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        self.l1, self.l2 = l1, l2
        self.dweights, self.dbiases = None, None
        self.m_w, self.v_w, self.m_b, self.v_b = 0, 0, 0, 0 # Optimizer states

    def forward(self, input_data, training=True):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, output_gradient):
        l2_grad = 2 * self.l2 * self.weights
        l1_grad = self.l1 * np.sign(self.weights)
        self.dweights = np.dot(self.input.T, output_gradient) + l2_grad + l1_grad
        self.dbiases = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)

class BatchNormalization(Layer):
    def __init__(self, input_dim, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        self.m_g, self.v_g, self.m_b, self.v_b = 0, 0, 0, 0 # Optimizer states
        self.dgamma, self.dbeta = None, None

    def forward(self, input_data, training=True):
        if training:
            self.batch_mean = np.mean(input_data, axis=0, keepdims=True)
            self.batch_var = np.var(input_data, axis=0, keepdims=True)
            self.normalized = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            return self.gamma * self.normalized + self.beta
        else:
            normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * normalized + self.beta

    def backward(self, output_gradient):
        N = output_gradient.shape[0]
        self.dgamma = np.sum(output_gradient * self.normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(output_gradient, axis=0, keepdims=True)
        
        dx_norm = output_gradient * self.gamma
        dvar = np.sum(dx_norm * (self.input - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (self.input - self.batch_mean), axis=0, keepdims=True) / N
        return dx_norm / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.input - self.batch_mean) / N + dmean / N
    
    # needs to capture input in forward for backward
    def forward(self, input_data, training=True):
        self.input = input_data # Capture input
        if training:
            self.batch_mean = np.mean(input_data, axis=0, keepdims=True)
            self.batch_var = np.var(input_data, axis=0, keepdims=True)
            self.normalized = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            return self.gamma * self.normalized + self.beta
        else:
            normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * normalized + self.beta

class ReLU(Layer):
    def forward(self, input_data, training=True):
        self.input = input_data
        return np.maximum(0, input_data)
    def backward(self, output_gradient): return output_gradient * (self.input > 0)

class Sigmoid(Layer):
    def forward(self, input_data, training=True):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    def backward(self, output_gradient): return output_gradient * self.output * (1 - self.output)

class Softmax(Layer):
    def forward(self, input_data, training=True):
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    def backward(self, output_gradient): return output_gradient # Assuming Combined Loss

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
    def forward(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            return input_data * self.mask
        return input_data
    def backward(self, output_gradient): return output_gradient * self.mask

#region: Optimizers

class HybridOptimizer:
    def __init__(self, opt='adam', lr=0.001, mo=0.9, b1=0.9, b2=0.999, eps=1e-8):
        self.opt, self.lr, self.mo, self.b1, self.b2, self.eps = opt, lr, mo, b1, b2, eps
        self.t = 0
    
    def update(self, layer):
        if isinstance(layer, Dense): params, grads, states = [layer.weights, layer.biases], [layer.dweights, layer.dbiases], [[layer.m_w, layer.v_w], [layer.m_b, layer.v_b]]
        elif isinstance(layer, BatchNormalization): params, grads, states = [layer.gamma, layer.beta], [layer.dgamma, layer.dbeta], [[layer.m_g, layer.v_g], [layer.m_b, layer.v_b]]
        else: return

        self.t += 1
        new_states = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m, v = states[i]
            if self.opt == 'sgd': p -= self.lr * g
            elif self.opt == 'momentum': m = self.mo * m - self.lr * g; p += m
            elif self.opt == 'adagrad': v += g**2; p -= self.lr * g / (np.sqrt(v) + self.eps)
            elif self.opt == 'rmsprop': v = self.b1 * v + (1 - self.b1) * g**2; p -= self.lr * g / (np.sqrt(v) + self.eps)
            elif self.opt == 'adam':
                m = self.b1 * m + (1 - self.b1) * g
                v = self.b2 * v + (1 - self.b2) * g**2
                m_h, v_h = m / (1 - self.b1**self.t), v / (1 - self.b2**self.t)
                p -= self.lr * m_h / (np.sqrt(v_h) + self.eps)
            new_states.append([m, v])
        
        if isinstance(layer, Dense): layer.m_w, layer.v_w, layer.m_b, layer.v_b = new_states[0][0], new_states[0][1], new_states[1][0], new_states[1][1]
        elif isinstance(layer, BatchNormalization): layer.m_g, layer.v_g, layer.m_b, layer.v_b = new_states[0][0], new_states[0][1], new_states[1][0], new_states[1][1]

#region: Loss Functions

def categorical_crossentropy(pred, true): return -np.mean(np.sum(true * np.log(np.clip(pred, 1e-15, 1)), axis=1))
def sparse_categorical_crossentropy(pred, true_idx): 
    # true_idx is 1D array of class indices
    log_p = np.log(np.clip(pred[np.arange(len(pred)), true_idx.astype(int)], 1e-15, 1))
    return -np.mean(log_p)

def binary_crossentropy(pred, true):
    pred = np.clip(pred, 1e-15, 1-1e-15)
    return -np.mean(true * np.log(pred) + (1-true) * np.log(1-pred))

def mse_loss(pred, true): return np.mean((pred - true)**2)
def huber_loss(pred, true, delta=1.0):
    error = true - pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)

#region: Model

class NeuralNetwork:
    def __init__(self, input_dim, n_classes, config):
        self.layers = []
        hidden = config.get('hidden_units', [256, 128])
        init = config.get('initializer', 'he_normal')
        use_bn = config.get('batch_norm', False)
        drop = config.get('dropout_rate', 0.0)
        l1, l2 = config.get('l1', 0.0), config.get('l2', 0.0)
        
        # trunk
        d_in = input_dim
        for h in hidden:
            self.layers.append(Dense(d_in, h, l1, l2, init))
            if use_bn: self.layers.append(BatchNormalization(h))
            self.layers.append(ReLU())
            if drop > 0: self.layers.append(Dropout(drop))
            d_in = h
            
        # heads
        self.cls_binary = config.get('binary_cls', False)
        # class Head
        cls_out = 1 if self.cls_binary else n_classes
        self.head_cls = [Dense(d_in, cls_out, l1, l2, init)]
        self.head_cls.append(Sigmoid() if self.cls_binary else Softmax())
        
        # reg Head
        self.head_reg = [Dense(d_in, 1, l1, l2, init)] # Linear default

    def forward(self, X, training=True):
        out = X
        for l in self.layers: out = l.forward(out, training)
        trunk_out = out
        
        c = trunk_out
        for l in self.head_cls: c = l.forward(c, training)
        
        r = trunk_out
        for l in self.head_reg: r = l.forward(r, training)
        return c, r

    def backward(self, g_cls, g_reg, opt):
        # backprop Heads
        gc = g_cls
        for l in reversed(self.head_cls):
            if not isinstance(l, (Softmax, Sigmoid)): gc = l.backward(gc); opt.update(l)
        
        gr = g_reg
        for l in reversed(self.head_reg): gr = l.backward(gr); opt.update(l)
        
        # backprop Trunk
        gt = gc + gr
        for l in reversed(self.layers): gt = l.backward(gt); opt.update(l)

#region: Bayesian Optimization (GP & TPE)
class GaussianProcess:
    def __init__(self, length_scale=1.0, noise=1e-5):
        self.ls, self.noise = length_scale, noise
        self.X_train, self.y_train = None, None
        self.K_inv = None

    def rbf_kernel(self, X1, X2):
        sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / self.ls**2 * sq_dist)

    def fit(self, X, y):
        self.X_train, self.y_train = X, y.reshape(-1, 1)
        K = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_new):
        K_trans = self.rbf_kernel(self.X_train, X_new)
        mean = np.dot(K_trans.T, np.dot(self.K_inv, self.y_train)).flatten()
        K_new = self.rbf_kernel(X_new, X_new)
        cov = K_new - np.dot(K_trans.T, np.dot(self.K_inv, K_trans))
        return mean, np.diag(cov)

class TPESampler:
    def __init__(self, gamma=0.25):
        self.gamma = gamma
        self.good_samples = []
        self.bad_samples = []
    
    def fit(self, X, y):
        # quantile split
        threshold = np.quantile(y, 1 - self.gamma) # Higher score is better
        # if we are maximizing metric. If minimizing loss, use quantile(y, gamma)
        # here assuming maximize metric (F1+R2)
        
        mask_good = y >= threshold
        self.good_samples = X[mask_good]
        self.bad_samples = X[~mask_good]

    def _parzen_window(self, x, samples, bandwidth=0.2):
        if len(samples) == 0: return 1.0
        # check type for categorical vs continuous
        # assuming normalized continuous for now
        sq_dist = np.sum((samples - x)**2, axis=1)
        kernel = np.exp(-0.5 * sq_dist / bandwidth**2) / (bandwidth * np.sqrt(2*np.pi))
        return np.mean(kernel)

    def score(self, X_candidates):
        scores = []
        for x in X_candidates:
            l_x = self._parzen_window(x, self.good_samples)
            g_x = self._parzen_window(x, self.bad_samples)
            scores.append(l_x / (g_x + 1e-9))
        return np.array(scores)

class BayesianOptimizer:
    def __init__(self, param_space, sampler='gp'):
        self.space = param_space # Dict of ranges
        self.sampler_type = sampler
        self.history_X = []
        self.history_y = []
        
    def _random_sample(self):
        config = {}
        vec = []
        for k, v in self.space.items():
            if isinstance(v, list): val = np.random.choice(v); idx = v.index(val) / len(v) # encode cat
            else: val = np.random.uniform(v[0], v[1]); idx = (val - v[0])/(v[1]-v[0]) # norm
            config[k] = val
            vec.append(idx)
        return config, np.array(vec)

    def _vector_to_config(self, vec):
        config = {}
        for i, (k, v) in enumerate(self.space.items()):
            if isinstance(v, list): 
                idx = int(vec[i] * len(v))
                idx = min(idx, len(v)-1)
                config[k] = v[idx]
            else:
                config[k] = vec[i] * (v[1]-v[0]) + v[0]
        return config

    def suggest(self):
        if len(self.history_X) < 5: return self._random_sample()
        
        X_train = np.array(self.history_X)
        y_train = np.array(self.history_y)
        
        # acquisition Optimization: Random sampling candidates and picking best EI/Ratio
        candidates = np.random.uniform(0, 1, (1000, len(self.space)))
        
        if self.sampler_type == 'gp':
            gp = GaussianProcess()
            gp.fit(X_train, y_train)
            mu, sigma = gp.predict(candidates)
            # EI
            best_y = np.max(y_train)
            with np.errstate(divide='ignore'):
                z = (mu - best_y) / (np.sqrt(sigma) + 1e-9)
                # approximation of CDF/PDF not avail in pure numpy without scipy
                # using Relu(mu - best) as proxy for exploitation + sigma for exploration
                # or just simple UCB: mu + kappa * sigma
                scores = mu + 1.0 * np.sqrt(sigma)
            best_idx = np.argmax(scores)
            
        elif self.sampler_type == 'tpe':
            tpe = TPESampler()
            tpe.fit(X_train, y_train)
            scores = tpe.score(candidates)
            best_idx = np.argmax(scores)
            
        return self._vector_to_config(candidates[best_idx]), candidates[best_idx]

    def register(self, vec, score):
        self.history_X.append(vec)
        self.history_y.append(score)

#endregion: Training Loop

def train_model(model, X_train, y_cls, y_reg, X_val, y_val_cls, y_val_reg, optimizer, config, epochs=30):
    bs = config.get('batch_size', 32)
    use_sparse = config.get('sparse_cls', False)
    use_huber = config.get('huber_reg', False)
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_r2': []}
    n_samples = len(X_train)
    
    # process Targets
    if not use_sparse and not model.cls_binary:
        y_cls_enc = one_hot_encode(y_cls, model.head_cls[0].weights.shape[1])
    else:
        y_cls_enc = y_cls # Keep indices for sparse or binary
        
    for ep in range(epochs):
        perm = np.random.permutation(n_samples)
        X_t, y_c_t, y_r_t = X_train[perm], y_cls_enc[perm], y_reg[perm]
        
        ep_loss = 0
        for i in range(0, n_samples, bs):
            X_b, yc_b, yr_b = X_t[i:i+bs], y_c_t[i:i+bs], y_r_t[i:i+bs].reshape(-1, 1)
            
            # forward
            pc, pr = model.forward(X_b, training=True)
            
            # loss & Grad
            if model.cls_binary:
                lc = binary_crossentropy(pc, yc_b.reshape(-1, 1))
                gc = (pc - yc_b.reshape(-1, 1)) / bs
            elif use_sparse:
                lc = sparse_categorical_crossentropy(pc, yc_b)
                # gradient for Sparse Softmax: p - 1 at correct index
                # need to construct full grad array
                gc = pc.copy()
                gc[np.arange(len(gc)), yc_b.astype(int)] -= 1
                gc /= bs
            else:
                lc = categorical_crossentropy(pc, yc_b)
                gc = (pc - yc_b) / bs
                
            if use_huber:
                # huber Gradient needs implementation specific logic
                # for simplicity here using MSE gradient or approx
                # implementing Huber Grad:
                delta = 1.0
                err = yr_b - pr # Note: my huber func defined true - pred
                # but gradient needed is dL/dPred. L = 0.5 * (y-p)^2 -> dL/dp = p-y
                # huber: if |y-p| < d: 0.5(y-p)^2 -> p-y. else d*|y-p| -> d*sign(p-y)
                diff = pr - yr_b
                is_small = np.abs(diff) <= delta
                mask_large = ~is_small
                gr = np.zeros_like(diff)
                gr[is_small] = diff[is_small]
                gr[mask_large] = delta * np.sign(diff[mask_large])
                lr_loss = huber_loss(pr, yr_b)
                gr /= bs
            else:
                lr_loss = mse_loss(pr, yr_b)
                gr = 2 * (pr - yr_b) / bs
            
            ep_loss += lc + lr_loss
            model.backward(gc, gr, optimizer)
            
        # validation
        pvc, pvr = model.forward(X_val, training=False)
        
        if model.cls_binary: pv_idx = (pvc > 0.5).astype(int).flatten()
        else: pv_idx = np.argmax(pvc, axis=1)
        
        f1 = macro_f1_score(y_val_cls, pv_idx)
        r2 = r2_score(y_val_reg, pvr)
        loss_val = 0 # placeholder for exact calc
        
        history['train_loss'].append(ep_loss / (n_samples/bs))
        history['val_loss'].append(loss_val) # Lazy for perf
        history['val_f1'].append(f1)
        history['val_r2'].append(r2)
        
        # print(f"Ep {ep}: F1 {f1:.3f} R2 {r2:.3f}")

    return history

#region: Main

def main():
    np.random.seed(42)
    # load
    t_df, te_df = pd.read_csv('c:/Users/nprra/Desktop/HackTech/CP/Python/ML/Data/train_nn.csv'), pd.read_csv('c:/Users/nprra/Desktop/HackTech/CP/Python/ML/Data/test_nn.csv')
    
    # preprocess
    cols = [f'F_{i}' for i in range(1, 21)]
    for c in cols:
        t_df[c] = t_df[c].fillna(t_df[c].mean())
        te_df[c] = te_df[c].fillna(t_df[c].mean())
    X = t_df[cols].values
    Xt = te_df[cols].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)
    
    yc = t_df['target_cls'].values
    yr = t_df['target_reg'].values
    N_cls = int(yc.max() + 1)
    
    # split
    idx = np.random.permutation(len(X))
    tr_idx, va_idx = idx[:int(0.8*len(X))], idx[int(0.8*len(X)):]
    X_tr, y_c_tr, y_r_tr = X[tr_idx], yc[tr_idx], yr[tr_idx]
    X_va, y_c_va, y_r_va = X[va_idx], yc[va_idx], yr[va_idx]
    
    # bayes Opt
    space = {
        'learning_rate': [1e-4, 1e-2], 'batch_size': [32, 64, 128],
        'dropout_rate': [0.1, 0.5], 'optimizer': ['adam', 'rmsprop', 'momentum'],
        'batch_norm': [0, 1], # 0: False, 1: True
        'sparse_cls': [1], # Force sparse for demo
        'huber_reg': [1] # Force huber for demo
    }
    
    bo = BayesianOptimizer(space, sampler='tpe') # Using TPE as requested
    
    best_score = -np.inf
    best_cfg = None
    best_hist = None
    best_model = None
    
    print("Starting Bayesian Optimization (TPE)...")
    for i in range(10): # 10 Iterations
        cfg, vec = bo.suggest()
        cfg['batch_norm'] = bool(cfg['batch_norm'] > 0.5)
        cfg['sparse_cls'] = True
        cfg['huber_reg'] = True
        cfg['hidden_units'] = [256, 128, 64]
        
        print(f"Iter {i+1}: {cfg}")
        mdl = NeuralNetwork(20, N_cls, cfg)
        opt = HybridOptimizer(cfg['optimizer'], cfg['learning_rate'])
        
        hist = train_model(mdl, X_tr, y_c_tr, y_r_tr, X_va, y_c_va, y_r_va, opt, cfg, epochs=15)
        
        score = max(hist['val_f1']) + max(hist['val_r2'])
        bo.register(vec, score)
        
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_model = mdl
            best_hist = hist
            print(f"  New Best! Score: {score:.4f}")

    print("\nBest Config:", best_cfg)
    plot_history(best_hist, title="Best Model (BayesOpt-TPE)", filename="best_bayesopt_plot.png")
    
    # Final Predict
    pc, pr = best_model.forward(Xt, training=False)
    pred_c = np.argmax(pc, axis=1)
    
    sub = pd.DataFrame({'target_cls': pred_c, 'target_reg': pr.flatten()})
    sub.to_csv('submission_nn.csv', index=False)
    print("Saved submission_nn.csv")

if __name__ == '__main__':
    main()

