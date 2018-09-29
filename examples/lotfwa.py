import sys
import numpy as np
import pickle as pkl

sys.path.append("../")
import benchmarks.cec2013.cec13 as cec13
import benchmarks.cec2017.cec17 as cec17
import fwa.LotFWA as LotFWA

if __name__ == "__main__": 

    exp_info = {
                "algorihtm": "LoTFWA",
                "date": "2018.4.23",
                "description": "Basic run of re-writed LoTFWA on cec2017 without LOT",
                "data":{
                        "res": "results of each run.",
                        "cost": "time cost of each run.",
                       },
               }

    func_num = 30
    repetition = 50

    res = np.empty((func_num, repetition))
    cost = np.empty((func_num, repetition))

    for i in range(func_num):
        for j in range(repetition):
        
            # wrap the fitness function
            def evaluate(x):
                if type(x) is np.ndarray:
                    x = x.tolist()
                return np.array(cec17.eval(x, i+1))
            
            model = LoTFWA.LoTFWA()
            model.load_prob(evaluator=evaluate, 
                            dim=30,
                            max_eval=30*10000)
            result = model.run()
            time_cost = model.time
            print("prob.{}\t, rep. {}\t, res: {}\t, time: {}".format(i+1, j+1, result, time_cost))
            res[i, j] = result
            cost[i, j] = model.time
    
    with open("logs/LoTFWA_CEC17_30D.pkl", "wb") as f:
        pkl.dump([exp_info, res, cost], f)
