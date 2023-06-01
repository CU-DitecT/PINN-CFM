from .GA import *
import math
import numpy as np

class GA_OneStep(GA):
    def __init__(self, args_GA):
        self.args = args_GA
    
    def MSE(self, para, simfunc_oneStep, states, labels, delta_t = 0.1, mse = 'position'):
        """
        states: all the initial states
        labels: [a, v_f_new, x_f_new]
        """
        X_f_new = []
        V_f_new = []
        A = []
        for state in states:
            x_f_new, v_f_new, a = simfunc_oneStep(para, state)
            X_f_new.append(x_f_new)
            V_f_new.append(v_f_new)
            A.append(a)
            # update the MSE
        if (mse == 'speed') | (mse == 'all'):
            MSE_V = [(x-y)**2 for x,y in zip(V_f_new, labels[:,1])]
        if (mse == 'position') | (mse == 'all'):
            MSE_X = [(x-y)**2 for x,y in zip(X_f_new, labels[:,0])]
        if (mse == 'acceleration') | (mse == 'all'):
            MSE_A = [(x-y)**2 for x,y in zip(A, labels[:,2])]

        if mse == 'all':
            return (math.sqrt(sum(MSE_V)/len(MSE_V)), 
                    math.sqrt(sum(MSE_X)/len(MSE_X)),
                    math.sqrt(sum(MSE_A)/len(MSE_A)))

        if (mse == 'speed'):
            return math.sqrt(sum(MSE_V)/len(MSE_V))
        if (mse == 'position'): 
            return math.sqrt(sum(MSE_X)/len(MSE_X))
        if (mse == 'acceleration'): 
            return math.sqrt(sum(MSE_A)/len(MSE_A))
        
    def executeGA(self, sim_info, states, labels):
        """
        sim_info: ( simfunc, lb, ub )
        data: dim of n*4
        
        return: best para, best mse, duration
        """
        num_generations = self.args.num_generations
        delta_t = self.args.delta_t
        mse = self.args.mse
        
        simfunc, lb, ub = sim_info
        
        best_outputs = []
        n = 0
        
        new_population = np.random.uniform(lb,
                                           ub,
                                           size = (self.args.sol_per_pop, len(lb)))
        # start time
        start = timeit.default_timer()
        for generation in range(num_generations):
            n += 1
            #fitness = [-1*x for x in list(map(MSE,new_population))]
            #fitness = list(map(MSE,new_population))
            #fitness = [ self.MSE(para, simfunc, data) for para in new_population ]
            fitness = [self.MSE(para, simfunc,
                                   states, labels,
                                   mse = 'position') 
                       for para in new_population]
            #print("{}/{} Best Output: {}".format( n, num_generations,np.min(fitness)))
            


            best_outputs.append(np.min(fitness))

            # Selecting parents
            parents = self.select_mating_pool(new_population, fitness)
            #print("{}/{} selected parents".format( n, num_generations))

            # generating next generation
            offspring_crossover = self.crossover(parents, len(lb))
            #print("{}/{} generated offspring".format( n, num_generations))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover,lb, ub)
            #print("{}/{} added variation".format( n, num_generations))

            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
            #print("{}/{} done".format( n, num_generations))


        #fitness = [-1*x for x in list(map(MSE,new_population))]
        fitness = [self.MSE(para, simfunc,
                                   states, labels,
                                   mse = 'position') 
                       for para in new_population]

        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.argmin(fitness)

        print("Best solution : ", new_population[best_match_idx, :])
        print("Best solution fitness : ", fitness[best_match_idx])
        
        stop = timeit.default_timer()
        return new_population[best_match_idx, :], fitness[best_match_idx], (stop-start)