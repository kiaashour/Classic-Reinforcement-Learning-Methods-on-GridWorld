class TD_agent(object):

  def epsilon_greedy_policy(self, epsilon, Q, nA):
        """ Create an epislon greedy policy for a state s.
        
        Args:            
            epsilon (float): Epsilon value to use
            Q (np.ndarray): Q values for state s
            nA (int): Number of actions possible                        
            
        Returns:
            np.ndarray: NumPy arrayof shape nA with epislocy greedy policy
        """        
        best_action = np.argmax(Q) 
        policy = np.ones(nA)*epsilon/nA
        policy[best_action] += (1-epsilon)
        return policy
    
  def solve(self, env):
    """
    Solve a given Maze environment using Q-learning

    Args:
        env (Maze): Maze environment to solve
        
    Returns:
        np.ndarray: Optimal policy
        list: List of value functions for each iteration
        list: List of total rewards for each episode        
    """
    #Variables
    epsilon = 0.85  #epsilon for policies
    alpha = 0.1
    gamma = env.get_gamma()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    #State and episode information
    Q = np.random.rand(state_size, action_size) 
    V = np.zeros(state_size)
    num_episodes = 3500
    returns = {(s,a): [] for s in range(state_size) for a in range(action_size)}
    values = [V]
    total_rewards = []
    
    #Policy
    policy = np.zeros((state_size, action_size))
    for state in range(state_size):
        policy[state,:] = self.epsilon_greedy_policy(epsilon, Q[state,:], action_size)
        
    
    #Q learning Algorithm:
    
    #For each episode:
    n_iter = 500
    for i in range(num_episodes):
                
        #Initialise S
        timestamp, state, reward, absorbing = env.reset()
        episode = [state]   #states in episode
        total_reward = 0    #total rewards of episode

        #Choosing action by sampling probability of actions
        action_probs = policy[state,:] 
        action = np.random.choice(a=np.arange(action_size), size=1, p=action_probs)[0]
        
        #Updating Q(s,a) for each step in the episode
        for j in range(n_iter):
            
            #Choosing action by sampling probability of actions
            action_probs = policy[state,:] 
            action = np.random.choice(a=np.arange(action_size), size=1, p=action_probs)[0]
            
            #Move one step
            timestamp, next_state, reward, absorbing = env.step(action)            
                        
            #Update Q with TD update
            next_best_action = np.argmax(Q[next_state, :])
            target = reward + gamma*Q[next_state, next_best_action]
            delta = target - Q[state, action]
            Q[state, action] += alpha*delta
            
            #Update total reward
            total_reward += reward
            
            #Break if have reached terminal state
            if absorbing:
                break
            
            #Update states and actions    
            episode.append(next_state)
            state = next_state
            
            
        #Update total rewards
        total_rewards.append(total_reward)
        
        #Update the policy and value function
        V = deepcopy(V)
        for state in episode:

            #Update policy for all actions given s
            policy[state,:] = self.epsilon_greedy_policy(epsilon, Q[state,:], action_size)

            #Update the values by maximum action-value
            best_value = np.max(Q[state,:])
            V[state] = best_value

        #Adding new values to values
        values.append(V)

    return policy, values[1:], total_rewards
