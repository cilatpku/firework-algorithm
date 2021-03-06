### Firework Algorithm

Introduction
---
Firework algorithm (FWA) is a novel derivation-free optimization algorithm, based on the interaction among the swarm.

The FWA mimics the procedure of the firework explosion, where stronger firework explodes into more sparks.
To get the details of the FWA, we refer you to the [original paper](http://www.cil.pku.edu.cn/publications/papers/ICSI2010zhuyuanchun.pdf).

This package provides easy acess to the advanced variants of FWA, maintained by the Computational Intelligence Laboratory (CIL), Peking University.

Install
---

To install the package, run the following commands:

```
git clone git@github.com:wead-hsu/firework-algorithm.git
cd firework-algorithm
python3 setup.py install
```
If you do not have the sudo authority, try `python3 setup.py install --user` instead.

Usage
---
After installing the package, you can include the package anywhere on your machine, i.e.,

```
import fwa.BBFWA as BBFWA

algo = BBFWA()
obj_func = lambda x: [sum([_ * _ for _ in xi]) for xi in x]
algo.load_prob(evaluator=obj_func,
	dim=30,
	max_eval=30*10000,
	)
result = algo.run()
```

The algorithm in this package minimizes the objective function. Therefore, if you want to maximize instead, convert the objectve function by changing the sign of the fitness.
And note that the evalutor takes a batch of samples as input and returns a list of scalars.

To run the optimizer, we first should set the range (if there is a restriction) and the number of evaluation.

There are several examples of previous proposed algorithms in the example directory, tested with CEC benchmarks.
For easilly reproducing the results on the standard benchmarks, we also provides the benchmarks package at [https://github.com/wead-hsu/ec-benchmarks](https://github.com/wead-hsu/ec-benchmarks), where the uasage of the benchmark package is illustrated.
For simplicity, we also provide a simple script in the example dir to install the benchmark pacakge.

Contact
----
[Computational Intelligence Laboratory (CIL), Peking University](www.cil.pku.edu.cn)

- Maintainer: 
	- Weidi Xu (wead_hsu@pku.edu.cn)
	- Yifeng Li (yfli@pku.edu.cn)

