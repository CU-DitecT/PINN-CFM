import timeit
import numpy as np
import math

def simIDM(para, ini_state, traj_l, delta_t = 0.1):
    '''
    para: v0, T, s0, a, b
    '''
    v_f_sim = ini_state[1]
    x_f = ini_state[0]
    X_F_SIM = [x_f]
    V_F_SIM = [v_f_sim]
    A_F_SIM = []
    delta_v = traj_l[0,1] - ini_state[1]
    delta_s = traj_l[0,0] - ini_state[0]

    for i in range(1, len(traj_l)):
        root = math.sqrt(para[3] * para[4])
        temp = v_f_sim * para[1]-0.5 * v_f_sim * delta_v / root

        #if temp<0:
        #    temp = 0

        desired_s_n = para[2] + temp
        An = para[3] * (1 - (v_f_sim / para[0])**4 - (desired_s_n / delta_s)**2)
        #An = max(An, para[3])
        #if An < -10:
        #    An = -10
        v_f_sim_new = v_f_sim +  delta_t * An

        x_f_new = x_f + (v_f_sim_new + v_f_sim)/2*delta_t
        x_f = x_f_new
        v_f_sim = v_f_sim_new

        delta_v = traj_l[i,1] - v_f_sim_new
        delta_s = traj_l[i,0] - x_f
        X_F_SIM.append(x_f_new)
        V_F_SIM.append(v_f_sim)
        A_F_SIM.append(An)
    return X_F_SIM, V_F_SIM, A_F_SIM

class GA():
    #NB!: Following bug should be fixed for implementation
    #(1) some 'self.' are missing
    #(2) the lb and ub are given from Modeler, not by args_GA
    def __init__(self, args_GA):
        self.args = args_GA
        
        
    
    def select_mating_pool(self, pop, fitness):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((self.args['num_parents_mating'], pop.shape[1]))
        for parent_num in range(self.args['num_parents_mating']):
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
            parents[parent_num, :] = pop[min_fitness_idx, :]
            fitness[min_fitness_idx] = 99999999999
        return parents

    def crossover(self, parents, num_para):
        offspring_size = (self.args['sol_per_pop']-self.args['num_parents_mating'], num_para)
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover, lb, ub):
        num_mutations = self.args['num_mutations']
        mutations_extend = self.args['mutations_extend']
        
        mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
        # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
        for idx in range(offspring_crossover.shape[0]):
            gene_idx = mutations_counter - 1
            for mutation_num in range(num_mutations):
                # The random value to be added to the gene.
                #random_value = np.random.uniform(-1.0, 1.0, 1)
                #offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value

                # The random value to be added to the gene.
                random_value = np.random.uniform(1-mutations_extend, 1+mutations_extend, 1)
                x_to_lb = offspring_crossover[idx, gene_idx] - lb[gene_idx]
                x_to_ub = ub[gene_idx] - offspring_crossover[idx, gene_idx]
                ub_ratio_lb = x_to_ub / x_to_lb

                new_ub_ratio_lb = random_value * ub_ratio_lb
                new_x_to_lb = 1/(1+new_ub_ratio_lb) * (ub[gene_idx] - lb[gene_idx])
                offspring_crossover[idx, gene_idx] = new_x_to_lb + lb[gene_idx]

                gene_idx = gene_idx + mutations_counter
        return offspring_crossover
    
    def RMSPE(self, para, simfunc, data):
        global Obs_X, Obs_V, Obs_A, x_f, v_f, a_f
        delta_t = self.args['delta_t']
        mse = self.args['mse']
        X_train = data
        Obs_V = []
        Obs_X = []
        Obs_A = []
        Sim_V = []
        Sim_X = []
        Sim_A = []
        for pair in X_train:
            ini_state = pair[0,[0,1]]
            traj_l = pair[:, [2,3]]
            traj_f = pair[:, [0,1]]
            v_f = pair[:, 1]
            x_f = pair[:, 0]
            a_f = np.diff(v_f) / delta_t
            X_F_SIM, V_F_SIM, A_F_SIM = simfunc(para, ini_state, traj_l)
            Sim_X+=X_F_SIM ; Sim_V+=V_F_SIM; Sim_A+=A_F_SIM
            Obs_X+=list(x_f); Obs_V += list(v_f); Obs_A += list(a_f)
        Obs_X = np.array(Obs_X); Obs_V = np.array(Obs_V); Obs_A = np.array(Obs_A)
        Sim_X = np.array(Sim_X); Sim_V = np.array(Sim_V); Sim_A = np.array(Sim_A)
        alpha_x = self.args['RMSPE_alpha_X']
        alpha_v = self.args['RMSPE_alpha_V']
        RMSPE_X = np.sqrt( sum(np.square(Obs_X-Sim_X))/sum(np.square(Obs_X)) )
        RMSPE_V = np.sqrt( sum(np.square(Obs_V-Sim_V))/sum(np.square(Obs_V)) )
        RMSPE = alpha_x*RMSPE_X + alpha_v*RMSPE_V
        return RMSPE
	
    def MSE(self, para, simfunc, data):
        delta_t = self.args['delta_t']
        mse = self.args['mse']
        X_train = data
        MSE_V = []
        MSE_X = []
        MSE_A = []

        for pair in X_train:
            ini_state = pair[0,[0,1]]
            traj_l = pair[:, [2,3]]
            traj_f = pair[:, [0,1]]
            v_f = pair[:, 1]
            x_f = pair[:, 0]
            a_f = np.diff(v_f) / delta_t
            X_F_SIM, V_F_SIM, A_F_SIM = simfunc(para, ini_state, traj_l)

            # update the MSE
            if ('speed' in mse) | (mse == 'all'):
                MSE_V += [(x-y)**2 for x,y in zip(V_F_SIM, v_f)]
            if ('position' in mse) | (mse == 'all'):
                MSE_X += [(x-y)**2 for x,y in zip(X_F_SIM, x_f)]
            if ('acceleration' in mse) | (mse == 'all'):
                MSE_A += [(x-y)**2 for x,y in zip(A_F_SIM, a_f)]
        if (mse == 'speed') | (mse == 'all'):
            return math.sqrt(sum(MSE_V)/len(MSE_V))
        if (mse == 'position') | (mse == 'all'):
            return math.sqrt(sum(MSE_X)/len(MSE_X))
        if (mse == 'acceleration') | (mse == 'all'):
            return math.sqrt(sum(MSE_A)/len(MSE_A))

    
    def executeGA(self, data):
        """
        sim_info: ( simfunc, lb, ub )
        data: dim of n*4
        
        return: best para, best mse, duration
        """
        num_generations = self.args['num_generations']
        delta_t = self.args['delta_t']
        mse = self.args['mse']
        
        lb = self.args['lb']
        ub = self.args['ub']
        simfunc = simIDM # defined at the top of this file
        
        best_outputs = []
        n = 0
        self.fitness = []
        
        # start time
        # initial population
        #print("lb:{}".format(lb))
        #print("ub:{}".format(ub))
        a = self.args['sol_per_pop']
        #print(self.args.sol_per_pop)
        a = self.args['sol_per_pop']
        new_population = np.random.uniform(low = lb,
                                               high = ub,
                                               size = (self.args['sol_per_pop'], len(lb)))
        start = timeit.default_timer()
        for generation in range(num_generations):
            n += 1
            #fitness = [-1*x for x in list(map(MSE,new_population))]
            #fitness = list(map(MSE,new_population))
            self.new_population = new_population
            self.data = data
            self.simfunc = simfunc
            fitness = [ self.RMSPE(para, simfunc, data) for para in new_population ]
            fitness = np.array(fitness)
            fitness[np.where(np.isnan(fitness))] = 99999999999
            self.fitness.append(fitness)
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
        #fitness = list(map(self.MSE,new_population))
        fitness = [ self.RMSPE(para, simfunc, data) for para in new_population ]
        

        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.argmin(fitness)

        #print("Best solution : ", new_population[best_match_idx, :])
        #print("Best solution fitness : ", fitness[best_match_idx])
        
        stop = timeit.default_timer()
        return new_population[best_match_idx, :], fitness[best_match_idx], (stop-start)
            