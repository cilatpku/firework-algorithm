import time
import numpy as np

class BBFWA(object):

    def  __init__(self):
        # Parameters

        # params of method
        self.sp_size = None       # total spark size
        self.init_amp = None      # initial dynamic amplitude
        
        # params of problem
        self.evaluator = None
        self.dim = None
        self.upper_bound = None
        self.lower_bound = None

        self.max_iter = None
        self.max_eval = None


        # States

        # private states
        self._num_iter = None
        self._num_eval = None
        self._dyn_amp = None

        # public states
        self.best_idv = None    # best individual found
        self.best_fit = None    # best fitness found
        self.trace = None       # trace of best individual in each generation

        # for inspection
        self.time = None

    def load_prob(self, 
                  # params for prob
                  evaluator = None,
                  dim = 2,
                  upper_bound = 100,
                  lower_bound = -100,
                  max_iter = 10000,
                  max_eval = 20000,
                  # params for method
                  sp_size = 200,
                  init_amp = 200,
                  ):

        # load params
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.max_iter = max_iter
        self.max_eval = max_eval

        self.sp_size = sp_size
        self.init_amp = init_amp
        
        # init states
        self._num_iter = 0
        self._num_eval = 0
        self._dyn_amp = init_amp
        self.best_idv = None
        self.best_fit = None
        self.trace = []

        self.time = 0

    def run(self):
        begin_time = time.time()

        fireworks, fits = self._init_fireworks()
        for idx in range(self.max_iter):
            
            if self._terminate():
                break

            fireworks, fits = self.iter(fireworks, fits)
        
        self.time = time.time() - begin_time

        return self.best_fit

    def iter(self, fireworks, fits):
        
        e_sparks, e_fits = self._explode(fireworks, fits)
         
        n_fireworks, n_fits = self._select(fireworks, fits, e_sparks, e_fits)    

        # update states
        if n_fits[0] < fits[0]:
            self._dyn_amp *= 1.2
        else:
            self._dyn_amp *= 0.9

        self._num_iter += 1
        self._num_eval += len(e_sparks)
            
        self.best_idv = n_fireworks[0]
        self.best_fit = n_fits[0]
        self.trace.append([n_fireworks[0], n_fits[0], self._dyn_amp])

        fireworks = n_fireworks
        fits = n_fits
        
        return fireworks, fits

    def _init_fireworks(self):
    
        fireworks = np.random.uniform(self.lower_bound, 
                                      self.upper_bound, 
                                      [1, self.dim])
        fireworks = fireworks.tolist()
        fits = self.evaluator(fireworks)

        return fireworks, fits

    def _terminate(self):
        if self._num_iter >= self.max_iter:
            return True
        if self._num_eval >= self.max_eval:
            return True
        return False

    def _explode(self, fireworks, fits):
        
        bias = np.random.uniform(-self._dyn_amp, self._dyn_amp, [self.sp_size, self.dim])
        rand_samples = np.random.uniform(self.lower_bound, self.upper_bound, [self.sp_size, self.dim])
        e_sparks = fireworks + bias
        in_bound = (e_sparks > self.lower_bound) * (e_sparks < self.upper_bound)
        e_sparks = in_bound * e_sparks + (1 - in_bound) * rand_samples
        e_sparks = e_sparks.tolist()
        e_fits = self.evaluator(e_sparks)
        return e_sparks, e_fits    

    def _select(self, fireworks, fits, e_sparks, e_fits):
        idvs = fireworks + e_sparks
        fits = fits + e_fits
        idx = np.argmin(fits)
        return [idvs[idx]], [fits[idx]]
