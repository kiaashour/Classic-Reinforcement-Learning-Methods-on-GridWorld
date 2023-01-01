class MC_agent(object):

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
    Solve a given Maze environment using Monte Carlo first-visit learning
    
    Args:
        env (Maze): Maze environment to solve

    Returns:
        np.ndarray: Optimal policy
        list: List of value functions for each iteration
        list: List of total rewards for each episode        
    """
    #Variables
    epsilon = 0.85  #epsilon for policies
    gamma = env.get_gamma()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    #State and episode information
    Q = np.random.rand(state_size, action_size) 
    V = np.zeros(state_size)
    num_episodes = 3000
    returns = {(s,a): [] for s in range(state_size) for a in range(action_size)}
    values = [V]
    total_rewards = []
    
    #Policy
    policy = np.zeros((state_size, action_size))
    for state in range(state_size):
        policy[state,:] = self.epsilon_greedy_policy(epsilon, Q[state,:], action_size)
    
    for i in range(num_episodes):    
        timestamp, state, reward, absorbing = env.reset()
        episode = []
        rewards = [] 
        no_iter = 500
        absorbing = False
        
        #Generate episode
        for i in range(no_iter):

            #Choosing action by sampling probability of actions
            action_probs = policy[state,:] 
            action = np.random.choice(a=np.arange(action_size), size=1, p=action_probs)[0]

            #Getting state information for either start state or next state
            timestamp, next_state, reward, absorbing = env.step(action)                
            episode.append((state, action))
            rewards.append(reward)

            #If the state is absorbing, break the loop
            if absorbing:
                break

            state = next_state

        #Update total_rewards
        total_rewards.append(sum(rewards))


        #Update Q(s,a) for each s, a
        visited = []
        for s,a in episode:

            #Update for first occurrence:
            if (s,a) not in visited:        
                #Find first index of occurrence
                idx = episode.index((s,a))

                #Calculate return following (s,a)
                discounted_return = np.sum([x*(gamma**i) for i, x in enumerate(rewards[idx:])])
                returns[(s,a)].append(discounted_return)

                #Update Q(s,a)
                Q[s,a] = np.mean(returns[s,a])
                visited.append((s,a))


        #Update the policy and value function
        episode_states = np.unique([s for s,a in episode])
        V = deepcopy(V)
        for state in episode_states:

            #Update policy for all actions given s
            policy[state,:] = self.epsilon_greedy_policy(epsilon, Q[state,:], action_size)

            #Update the values by maximum action-value
            best_value = np.max(Q[state,:])
            V[state] = best_value

        #Adding new values to values
        values.append(V)
            
                
    return policy, values[1:], total_rewards