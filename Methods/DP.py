import numpy as np 
import random
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from copy import deepcopy


class DP_agent(object):     
        
    def policy_evaluation(self, env, policy, threshold = 0.0001):
        """ Policy evaluation on Maze environment.

        Args: 
            policy (np.array): policy to evaluate
            threshold (float): threshold value used to stop the policy evaluation algorithm
            gamma (float): discount factor

        Returns:           
            V (np.array): value function corresponding to the policy 
            epochs (int): number of epochs to find this value function
        """
        #Get environment information
        state_size = env.get_state_size()
        action_size = env.get_action_size()
        T = env.get_T()
        R = env.get_R()
        gamma = env.get_gamma()
        absorbing = env.get_absorbing()

        #Ensure inputs are valid
        assert (policy.shape[0] == state_size) and (policy.shape[1] == action_size), "The dimensions of the policy are not valid."
        assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

        #Initialisation
        delta= 2*threshold  #ensure delta is bigger than the threshold to start the loop
        V = np.zeros(state_size)  #initialise value function to 0
        Vnew = np.copy(V)  #make a deep copy of the value array to hold the update during the evaluation    
        epoch = 0


        #Loop until convergence
        while delta > threshold:    
          epoch += 1

          #Perform a full back-up of states
          for prior_state in range(state_size):
            if not absorbing[0, prior_state]:  #ensure state are non-absorbing

              tmpV = 0  #temporary variable for V
              for action in range(action_size):

                tmpQ = 0  #temporary variable for Q
                for post_state in range(state_size):
                  tmpQ = tmpQ + T[prior_state, post_state, action] * (R[prior_state, post_state, action] + gamma * V[post_state])
                tmpV += policy[prior_state, action] * tmpQ

              #Update the value of the state
              Vnew[prior_state] = tmpV

          #Compute new delta (other stopping criteria may be used)
          delta =  max(abs(Vnew - V))

          #Update V
          V = np.copy(Vnew)

        return V, epoch

    def policy_iteration(self, env, threshold = 0.0001):
        """ Policy iteration on Maze environment.

        Args: 
            threshold (float): threshold value used to stop the policy iteration algorithm
            gamma (float): discount factor

        Returns:
            policy (np.array): policy found using the policy iteration algorithm
            V (np.array): value function corresponding to the policy 
            epochs (int): number of epochs to find this policy
        """

        #Get environment information
        state_size = env.get_state_size()
        action_size = env.get_action_size()
        T = env.get_T()
        R = env.get_R()
        gamma = env.get_gamma()
        absorbing = env.get_absorbing()

        #Ensure gamma value is valid
        assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

        #Initialisation
        policy = np.zeros((state_size, action_size))  #matrix of 0
        policy[:, 0] = 1  #initialise policy to choose action 1 systematically
        epochs = 0
        policy_stable = False  #condition to stop the main loop

        while not policy_stable: 

          #Policy evaluation
          V, epochs_eval = self.policy_evaluation(env, policy, threshold = threshold)
          epochs += epochs_eval  #increment epoch

          #Set the boolean to True, it will be set to False later if the policy prove unstable
          policy_stable = True

          #Policy iteration
          for prior_state in range(state_size):
            if not absorbing[0, prior_state]:  #ensure state are non-absorbing

              #Store the old action (used for convergence criterion)
              old_action = np.argmax(policy[prior_state, :])

              #Compute Q value
              Q = np.zeros(action_size)  #initialise with value 0
              for post_state in range(state_size):
                Q += T[prior_state, post_state, :] * (R[prior_state, post_state, :] + gamma * V[post_state])

              #Compute optimal policy for prior state
              new_policy = np.zeros(action_size)
              new_policy[np.argmax(Q)] = 1  #The action that maximises the Q value gets probability 1

              #Add to general policy
              policy[prior_state, :] = new_policy

              #Check if the policy has converged
              if old_action != np.argmax(policy[prior_state, :]):
                policy_stable = False

        return policy, V, epochs

    def value_iteration(self, env, threshold=0.0001):
        """ Value iteration on Maze environment.

        Args:
            threshold (float): threshold value used to stop the value iteration algorithm        
            env (Maze): environment

        Returns:
            policy (np.array): policy found using the value iteration algorithm
            V (np.array): value function corresponding to the policy
            epochs (int): number of epochs to find this policy
        """
        #Get environment information
        state_size = env.get_state_size()
        action_size = env.get_action_size()
        T = env.get_T()
        R = env.get_R()
        gamma = env.get_gamma()
        absorbing = env.get_absorbing()
        
        #Initialisation
        epochs = 0
        delta = 2*threshold #Setting value of delta to go through the first breaking condition
        V = np.zeros(state_size) #Initialise values at 0 for each state
        
        while delta > threshold:
            
            epochs += 1
            delta = 0

            #For each state
            for prior_state in range(state_size):

                #If not an absorbing state
                if not absorbing[0, prior_state]:

                  #Store the previous value for that state
                  v = V[prior_state] 

                  #Compute Q value
                  Q = np.zeros(action_size)
                  for post_state in range(state_size):
                    Q += T[prior_state, post_state,:] * (R[prior_state, post_state, :] + gamma * V[post_state])

                  #Set the new value to the maximum of Q
                  V[prior_state]= np.max(Q) 

                  #Compute the new delta                  
                  delta = max(delta, np.abs(v - V[prior_state]))
                
        #When the loop is finished, fill in the optimal policy
        policy = np.zeros((state_size, action_size))

        for prior_state in range(state_size):
            #Compute the Q value
            Q = np.zeros(action_size)
            for post_state in range(state_size):
                Q += T[prior_state, post_state,:] * (R[prior_state, post_state, :] + gamma * V[post_state])

            #The action that maximises the Q value gets probability 1
            policy[prior_state, np.argmax(Q)] = 1 

        return policy, V, epochs                            
        
    def solve(self, env, method="V"):
        """
        Solve a given Maze environment using either value iteration or policy iteration.

        Args:
            env (Maze): environment
            method (str): method to use to solve the environment. Either "V" for value iteration or "P" for policy iteration.

        Returns:
            policy (np.array): policy found using the value iteration algorithm
            V (np.array): value function corresponding to the policy
        """

        #Solve the environment using the chosen method
        if method == "P":
            policy, V, epochs = self.policy_iteration(env)
        elif method == "V":
            policy, V, epochs = self.value_iteration(env)
        else:
            raise ValueError("Method should be either 'V' or 'P'.")
        
        return policy, V