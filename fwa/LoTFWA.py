import os
import time
import numpy as np

EPS = 1e-8

class LoTFWA(object):

    def  __init__(self):
        # Parameters

        # params of method
        self.fw_size = None       # num of fireworks
        self.sp_size = None       # total spark size
        self.init_amp = None      # initial dynamic amplitude
        self.gm_ratio = None      # ratio for top sparks in guided mutation

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
        self._dyn_amps = None
        self._num_spark = None

        # public states
        self.best_idv = None    # best individual found
        self.best_fit = None    # best fitness found

        # for inspection
        self.time = None
        self.info = None

    def load_prob(self, 
                  # params for prob
                  evaluator = None,
                  dim = 2,
                  upper_bound = 100,
                  lower_bound = -100,
                  max_iter = 10000,
                  max_eval = 20000,
                  # params for method
                  fw_size = 5,
                  sp_size = 300,
                  init_amp = 200,
                  gm_ratio = 0.2,):

        # load params
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.max_iter = min(max_iter, int(max_eval / (sp_size + fw_size)))
        self.max_eval = max_eval

        self.fw_size = fw_size
        self.sp_size = sp_size
        self.gm_ratio = gm_ratio
        self.init_amp = init_amp
        
        # init states
        self._num_iter = 0
        self._num_eval = 0
        self._dyn_amps = np.ones((self.fw_size)) * init_amp
        self._num_spark = int(self.sp_size / self.fw_size)
        
        self.best_idv = None
        self.best_fit = None
        
        # init inspection info
        self.time = 0
        self.info = {}

        # init random seed
        np.random.seed(int(os.getpid()*time.clock()))

    def run(self):
        begin_time = time.clock()

        fireworks, fits = self._init_fireworks()
        for idx in range(self.max_iter):
            
            if self._terminate():
                break

            fireworks, fits = self.iter(fireworks, fits)
        
        self.time = time.clock() - begin_time

        return self.best_fit, self.time

    def iter(self, fireworks, fits):
    
        e_sparks, e_fits = self._explode(fireworks, fits)
        
        m_sparks, m_fits = self._mutate(fireworks, 
                                        fits, 
                                        e_sparks, 
                                        e_fits)
        n_fireworks, n_fits = self._select(fireworks, 
                                           fits, 
                                           e_sparks, 
                                           e_fits, 
                                           m_sparks, 
                                           m_fits)    
        
        n_fireworks, n_fits, restart_num  = self._restart(fireworks, 
                                                          fits, 
                                                          n_fireworks, 
                                                          n_fits)

        # update states

        # dynamic amps
        for idx in range(self.fw_size):
            if n_fits[idx] < fits[idx] - EPS:
                self._dyn_amps[idx] *= 1.2
            else:
                self._dyn_amps[idx] *= 0.9
        
        # iter and eval num 
        self._num_iter += 1
        self._num_eval += self.sp_size + self.fw_size + restart_num

        # record best results
        min_idx = np.argmin(n_fits)
        self.best_idv = n_fireworks[min_idx, :]
        self.best_fit = n_fits[min_idx]
        
        # new fireworks
        fireworks = n_fireworks
        fits = n_fits
        
        return fireworks, fits

    def _init_fireworks(self):
    
        fireworks = np.random.uniform(self.lower_bound, 
                                      self.upper_bound, 
                                      [self.fw_size, self.dim])
        fits = self.evaluator(fireworks)

        return fireworks, fits

    def _terminate(self):
        if self._num_iter >= self.max_iter:
            return True
        if self._num_eval >= self.max_eval:
            return True
        return False

    def _explode(self, fireworks, fits):

        # alocate sparks(even for LoTFWA)
        sum_spark = self._num_spark * self.fw_size

        # compute amplitude
        amps = self._dyn_amps
        
        # explode
        bias = np.random.uniform(-1, 1, [self.fw_size, self._num_spark, self.dim])
        bias = bias * np.tile(amps[:, np.newaxis, np.newaxis], (1, self._num_spark, 1))
        e_sparks = np.tile(fireworks[:,np.newaxis, :], (1, self._num_spark, 1)) + bias

        # mapping
        rand_samples = np.random.uniform(self.lower_bound, 
                                         self.upper_bound, 
                                         [self.fw_size, self._num_spark, self.dim])
        in_bound = (e_sparks > self.lower_bound) * (e_sparks < self.upper_bound)
        e_sparks = in_bound * e_sparks + (1 - in_bound) * rand_samples

        e_fits = self.evaluator(e_sparks.reshape(sum_spark, self.dim))
        e_fits = e_fits.reshape((self.fw_size, self._num_spark))

        return e_sparks, e_fits

    def _mutate(self, fireworks, fits, e_sparks, e_fits):
       
        top_num = int(self._num_spark * self.gm_ratio)
        sort_idx = np.argsort(e_fits, axis=1)
        fw_idx = np.tile(np.array(range(self.fw_size))[np.newaxis, :].transpose(), top_num)
        top_idx = sort_idx[:, -top_num:]
        btm_idx = sort_idx[:, :top_num]

        delta = np.mean(e_sparks[fw_idx, top_idx, :], axis=1) - np.mean(e_sparks[fw_idx, btm_idx, :], axis=1)

        m_sparks = fireworks + delta
        m_fits = self.evaluator(m_sparks)
        m_sparks = m_sparks.reshape((self.fw_size, 1, self.dim))
        m_fits = m_fits.reshape((self.fw_size, 1)) 
        return m_sparks, m_fits
             
    def _select(self, fireworks, fits, e_sparks, e_fits, m_sparks, m_fits):
        
        all_sparks = np.concatenate((fireworks[:,np.newaxis,:],
                                     e_sparks,
                                     m_sparks), axis=1)
        all_fits = np.concatenate((fits[:, np.newaxis],
                                   e_fits,
                                   m_fits), axis=1)

        min_idx = np.argmin(all_fits, axis=1)
        fw_idx = np.array(range(self.fw_size))

        n_fireworks = all_sparks[fw_idx, min_idx, :]
        n_fits = all_fits[fw_idx, min_idx]

        return n_fireworks, n_fits

    def _restart(self, fireworks, fits, n_fireworks, n_fits):
        improves = fits - n_fits
        min_fit = min(n_fits)
        restart = (improves > 0) * (improves*(self.max_iter-self._num_iter) < (n_fits-min_fit))
        replace = restart[:, np.newaxis].astype(np.int32)
        restart_num = sum(replace)

        if restart_num > 0:
            rand_sample = np.random.uniform(self.lower_bound,
                                            self.upper_bound,
                                            (self.fw_size, self.dim))
            n_fireworks = (1-replace)*n_fireworks + replace*rand_sample
            n_fits[restart] = self.evaluator(n_fireworks[restart, :])
            self._dyn_amps[restart] = self.init_amp

        return n_fireworks, n_fits, restart_num

