from .utils import *
from MC import MC_agent
from TD import TD_agent
from DP import DP_agent

############################################################################################################
#This is the main file for the project. It contains the code for several experiments.
############################################################################################################

#DP agent comparisons
maze = Maze()

#Create agent with policy iteration
dp_agent = DP_agent()
dp_policy, dp_value, epochs = dp_agent.solve(maze, method="P")

#Printing results
print("Results of the DP agent:\n")
print("Policy:\n")
maze.get_graphics().draw_policy(dp_policy)
print("Values:\n")
maze.get_graphics().draw_value(dp_value)

#Create agent with value iteration
dp_agent = DP_agent()
dp_policy_final, dp_value_final, epohcs = dp_agent.solve(maze, method="V")

#Printing results
print("Results of the DP agent:\n")
print("Policy:\n")
maze.get_graphics().draw_policy(dp_policy_final)
print("Values:\n")
maze.get_graphics().draw_value(dp_value_final)


#Time comparisons of the two methods
no_runs = 30
times_V = []
times_P = []
for i in range(no_runs):

    #Time for V DP method
    t0 = time.time()

    dp_agent = DP_agent()    
    dp_agent.solve(maze, method="V")

    t1 = time.time()
    times_V.append(t1-t0)

    #Time for P DP method
    t0 = time.time()

    dp_agent = DP_agent()    
    dp_agent.solve(maze, method="P")

    t1 = time.time()
    times_P.append(t1-t0)

#Printing means:
print(f"Value iteration mean time: {np.mean(times_V)}")
print(f"Policy iteration mean time: {np.mean(times_P)}")


#Plotting
xs = np.arange(1, no_runs+1)
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, times_V, label = "Value iteration")
plt.plot(xs, times_P, label = "Policy iteration")
plt.ylabel("Time (s)")
plt.xlabel("Run number")
plt.legend()
plt.show()
plt.savefig("DP_PI_vs_VI_Time.png", dpi=1000)


#Solution with different parameters for
#p < 0.25, p = 0.25 or p > 0.25, and similarly γ < 0.5 or γ > 0.5. 
gamma = np.array([0.05, 0.9])
probs = np.array([0.05, 0.25,0.95])

#Values of final states for different gamma and probs
values_gamma = []
values_probs = []

#Gamma
for i in range(len(gamma)):

    new_maze = Maze()
    new_maze._gamma = gamma[i]

    #Create agent
    dp_agent = DP_agent()
    dp_policy, dp_value, epohcs = dp_agent.solve(new_maze, method="V")

    #Output
    print(f"Gamma {gamma[i]}")    
    new_maze.get_graphics().draw_policy(dp_policy)
    print("\n\n")
    print(f"Gamma {gamma[i]}")    
    new_maze.get_graphics().draw_value(dp_value)
    print("\n\n")


#Probs
for i in range(len(probs)):

    new_maze = Maze(probs[i])
    
    #Create agent
    dp_agent = DP_agent()
    dp_policy, dp_value, epohcs = dp_agent.solve(new_maze, method="V")    

    #Output
    print(f"p {probs[i]}")
    new_maze.get_graphics().draw_policy(dp_policy)
    print("\n\n")
    print(f"p {probs[i]}")
    new_maze.get_graphics().draw_value(dp_value)
    print("\n\n")    


############################################################################################################
############################################################################################################

#Defining helper functions
def replicate(n, m, agent, env):
    """ Function that replicates results of an agent for Maze environment.

    Parameters:
        n: number of replications
        m: number of episodes
        agent: RL agent to use
        env: environment to use

    Returns:
        tuple: (V, rewards)
        V: a matrix of size nxm, where m is the number of episodes of our agent
        where element i,j contains the squared sum of the states values for episode j in replication i
        total_rewards: a matrix of size nxm, where element i,j is the total non-discounted reward for episode j in replication i
    """
    V = np.zeros((n,m))
    total_rewards = np.zeros((n,m))

    #Solve the agent for each replication
    for i in range(n):
        policy, values, rewards = agent.solve(env)
        values = values[1:]  #removing zero initial state

        #Transform values to vector of size mx1 by square summing
        for j in range(len(values)):
            squared_sum = np.sum(values[j]**2)
            values[j] = squared_sum

        #Update V and total_rewards        
        V[i,:] = values
        total_rewards[i,:] = rewards

    return V, total_rewards


def gradient(point_1, point_2):
    """ Function that finds the gradient between two points.

    Parameters:
        point_1: first point
        point_2: second point
        
    Returns:
        gradient: gradient between the two points
    """
    dy = point_1[1] - point_2[1]
    dx = point_1[0] - point_2[0]
    return dy/dx


def smoothness(points):
    """ Function that finds the smoothness metric of a curve.

    Parameters:
        points: array of (x,y) points (can be tuple or array)

    Returns:
        smoothness_metric: smoothness metric of the curve
    """
    smoothness_metric = 0    
    #Find smoothness metric by summing adjacent gradients
    for i in range(len(points)-1):
        grad = gradient(points[i], points[i+1])
        smoothness_metric += grad

    return smoothness_metric


def avg_difference(matrix, n):
    """ Function that finds the difference of the row average up to nth row and (n-1)th row.

    Parameters:
        matrix: matrix to find the difference of the row average
        n: row index to find the difference of the row average

    Returns:
        avg_difference: difference of the row average up to nth row and (n-1)th row
    """
    avg_1 = np.mean(matrix[:n,:], axis=0)
    avg_2 = np.mean(matrix[:(n-1),:], axis=0)
    return avg_1 - avg_2



#MC agent comparisons

#Choosing solution by viewing different solutions for epsilon
epsilons = [0.1, 0.20, 0.5]
for epsilon in epsilons:

    agent = MC_agent(epsilon=epsilon, episode_no=3000)
    policy, values, tot_rewards = agent.solve(maze)

    #Plot solution
    print(f"Solution for epsilon {epsilon}:\n\n")
    maze.get_graphics().draw_policy(policy)
    maze.get_graphics().draw_value(values[-1])
    print("\n\n")

agent = MC_agent(epsilon=0.85, episode_no=3000)
policy, values, tot_rewards = agent.solve(maze)

#Plot solution
print(f"Solution for epsilon {0.85}:\n\n")
maze.get_graphics().draw_policy(policy)
maze.get_graphics().draw_value(values[-1])
print("\n\n")



#Create replications and obtain V and total rewards matrices for the replications
MC = MC_agent(epsilon=0.85, episode_no=3000)
n = 40
m = 3000
results = replicate(n, m, MC, maze)

#Get values and rewards
values = results[0]
rewards = results[1]

#Renaming for later use
optimal_rewards = rewards
optimal_values = values

#Saving
np.save(f'MC_sum_square_values_repls_m{m}_n{n}.npy', values, allow_pickle=True)
np.save(f'MC_total_rewards_repls_m{m}_n{n}.npy', rewards, allow_pickle=True)


#Plot values against episode number as example
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(np.arange(1,m+1), values[0])
plt.ylabel("Sum of values squared")
plt.xlabel("Episode number")
plt.show()
plt.savefig("Q2.3_Sum_Values_Squared_vs_Episode.png", dpi=1000)


#For i=2 to n+1, finding avg_difference(values, i), where we have added the 0 initial state to the values matrix
#Which will results in a list with (n) elements of size mx1
avg_differences = []
values = np.vstack([np.zeros(m), values])
for i in range(2, n+2):
    diff = avg_difference(values, i)
    avg_differences.append(diff)    

#Saving
np.save('MC_avg_diffs_values_repls.npy', avg_differences, allow_pickle=True)

#Finding the smoothness for each plot in avg_differences
xs = np.arange(1,m+1)
smooths = []
for i, diff in enumerate(avg_differences):
    coordinates = list(zip(xs, diff))
    smooth = smoothness(coordinates)
    smooths.append(smooth)

np.save("MC_smooths.npy", smooths, allow_pickle=True)    


#Plot smooths
xs = np.arange(1, len(smooths)+1)
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, smooths)
plt.ylabel("Smoothness metric")
plt.xlabel("Episode number")
plt.show()
plt.savefig("Q2.3_Smoothness_vs_Episode.png", dpi=1000)


#Plotting mean and std of total discounted rewards against episode number
#For 25 replications
reward_mean = np.mean(rewards[:25,:], axis=0)
optimal_reward_mean = reward_mean  #saving for later
reward_std = np.std(rewards[:25,:], axis=0)
xs = np.arange(1, rewards.shape[1]+1)

#Above and below mean
above_mean = reward_mean + reward_std
below_mean = reward_mean - reward_std

plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, reward_mean, label="mean")
plt.fill_between(xs, reward_mean, above_mean, color="grey", alpha=0.2, label = "+/- std")
plt.fill_between(xs, below_mean, reward_mean, color="grey", alpha=0.2)
plt.ylabel("Total non-discounted reward")
plt.xlabel("Episode number")
plt.legend()
plt.show()
plt.savefig("Q2.4_Total_Reward_vs_Episode.png", dpi=1000)


#Learning curve for different epsilons
epsilons = [0.2, 0.5, 0.85]
rewards_list = []
for epsilon in epsilons:

    if epsilon==0.85:
        rewards_list.append(optimal_reward_mean)
    else:
        MC = MC_agent(epsilon=epsilon, episode_no=3000)
        n = 25
        m = 3000
        values, rewards = replicate(n, m, MC, maze)
        rewards_list.append(np.mean(rewards, axis=0))

        #Saving
        filename_vals = f"MC_Q2.5_vals_e{epsilon}_n{n}_m{m}.npy"
        filename_rews = f"MC_Q2.5_rews_e{epsilon}_n{n}_m{m}.npy"

        np.save(filename_vals, values, allow_pickle=True)
        np.save(filename_rews, rewards, allow_pickle=True)


#Plotting
#Getting x coordinates
xs = np.arange(1, rewards_list[0].shape[0]+1)

#Plots
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, rewards_list[0], label=f"epsilon={epsilons[0]}")
plt.plot(xs, rewards_list[1], label=f"epsilon={epsilons[1]}")
plt.plot(xs, rewards_list[2], label=f"epsilon={epsilons[2]}")
plt.ylabel("Total non-discounted reward")
plt.xlabel("Episode number")
plt.legend()
plt.show()
plt.savefig("Q2.5_Total_Reward_vs_Episode.png", dpi=1000)

############################################################################################################
############################################################################################################

#TD agent comparislons

#Choosing solution by viewing different solutions for epsilon
epsilons = [0.20, 0.5, 0.70, 0.85]
for epsilon in epsilons:

    agent = TD_agent(epsilon=epsilon, alpha=0.1, episode_no=3500)
    td_policy, td_values, td_tot_rewards = agent.solve(maze)

    #Plot solution
    print(f"Solution for epsilon {epsilon}:\n\n")
    maze.get_graphics().draw_policy(td_policy)
    maze.get_graphics().draw_value(td_values[-1])
    print("\n\n")


#Create replications and obtain V and total rewards matrices for the replications
TD = TD_agent(epsilon=0.85, alpha=0.1, episode_no=3500)
n = 25
m = 3500
results = replicate(n, m, TD, maze)

#Obtaining results
values = results[0]
rewards = results[1]

#Saving
np.save(f'TD_sum_square_values_repls_m{m}_n{n}.npy', values, allow_pickle=True)
np.save(f'TD_total_rewards_repls_m{m}_n{n}.npy', rewards, allow_pickle=True)


#Plotting mean and std of total discounted rewards against episode number
reward_mean = np.mean(rewards, axis=0)
optimal_reward_mean = reward_mean  #saving for later use
reward_std = np.std(rewards, axis=0)
xs = np.arange(1, rewards.shape[1]+1)

#Above and below mean
above_mean = reward_mean + reward_std
below_mean = reward_mean - reward_std

plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, reward_mean, label="mean")
plt.fill_between(xs, reward_mean, above_mean, color="grey", alpha=0.2, label = "+/- std")
plt.fill_between(xs, below_mean, reward_mean, color="grey", alpha=0.2)
plt.ylabel("Total non-discounted reward")
plt.xlabel("Episode number")
plt.legend()
plt.show()
plt.savefig("Q3.3_Total_Reward_vs_Episode.png", dpi=1000)


#Learning curve for different epsilons
epsilons = [0.2, 0.5, 0.85]
rewards_list = []
for epsilon in epsilons:

    if epsilon==0.85:
        rewards_list.append(optimal_reward_mean)
    else:
        TD = TD_agent(epsilon=epsilon, alpha=0.1, episode_no=3500)
        n = 25
        m = 3500
        values, rewards = replicate(n, m, TD, maze)
        rewards_list.append(np.mean(rewards, axis=0))    

        #Saving
        filename_vals = f"TD_Q3.4_vals_e{epsilon}_n{n}_m{m}.npy"
        filename_rews = f"TD_Q3.4_rews_e{epsilon}_n{n}_m{m}.npy"

        np.save(filename_vals, values, allow_pickle=True)
        np.save(filename_rews, rewards, allow_pickle=True)


#Plotting
#Getting x coordinates
xs = np.arange(1, rewards_list[0].shape[0]+1)

#Plots
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, rewards_list[0], label=f"epsilon={epsilons[0]}")
plt.plot(xs, rewards_list[1], label=f"epsilon={epsilons[1]}")
plt.plot(xs, rewards_list[2], label=f"epsilon={epsilons[2]}")
plt.ylabel("Total non-discounted reward")
plt.xlabel("Episode number")
plt.legend()
plt.show()
plt.savefig("Q3.4_Total_Reward-vs_Episode.png", dpi=1000)


#Learning curve for different alphas
alphas = [0.1, 0.5, 0.9]
rewards_list = []
for alpha in alphas:

    if alpha==0.1:
        rewards_list.append(optimal_reward_mean)
    else:
        TD = TD_agent(epsilon=0.85, alpha=alpha, episode_no=3500)
        n = 25
        m = 3500
        values, rewards = replicate(n, m, TD, maze)
        rewards_list.append(np.mean(rewards, axis=0))    

        #Saving
        filename_vals = f"TD_Q3.4_vals_a{alpha}_n{n}_m{m}.npy"
        filename_rews = f"TD_Q3.4_rews_a{alpha}_n{n}_m{m}.npy"

        np.save(filename_vals, values, allow_pickle=True)
        np.save(filename_rews, rewards, allow_pickle=True)


#Plotting
#Getting x coordinates
xs = np.arange(1, rewards_list[0].shape[0]+1)

#Plots
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(xs, rewards_list[0], label=f"alpha={alphas[0]}")
plt.plot(xs, rewards_list[1], label=f"alpha={alphas[1]}")
plt.plot(xs, rewards_list[2], label=f"alpha={alphas[2]}")
plt.ylabel("Total discounted reward")
plt.xlabel("Episode number")
plt.legend()
plt.show()
plt.savefig("Q3.4_Total_Reward_vs_Episode_Alpha.png", dpi=1000)

############################################################################################################
############################################################################################################

#Comparisons of MC and TD

#Finding MSE against episode for MC and TD agents
#For MC, TD:
MC = MC_agent(epsilon=0.85, episode_no=3000)
TD = TD_agent(epsilon=0.85, alpha=0.1, episode_no=3000)
n = 25
m = 3000


#Finding the MSEs
state_size = 98
MC_mses = np.zeros((n,m))
TD_mses = np.zeros((n,m))
for i in range(n):

    #Getting values and policies
    MC_policy, MC_values, MC_tot_rewards = MC.solve(maze)
    TD_policy, TD_values, TD_tot_rewards = TD.solve(maze)

    #Finding MSEs and adding them
    for j in range(m):
        MC_mses[i,j] = mean_squared_error(y_true=dp_value_final, y_pred=MC_values[j+1])  #j+1 because first is initial
        TD_mses[i,j] = mean_squared_error(y_true=dp_value_final, y_pred=TD_values[j+1])  #j+1 because first is initial    


#Getting meand and stds
#Means
MC_means = np.mean(MC_mses, axis=0)
TD_means = np.mean(TD_mses, axis=0)

#Stds
MC_stds = np.std(MC_mses, axis=0)
TD_stds = np.std(TD_mses, axis=0)


#Plotting
xs = np.arange(1, MC_means.shape[0]+1)

#Plots
plt.figure(figsize=(10, 6), dpi=80)

#MC
plt.plot(xs, MC_means, label="MC mean")
plt.fill_between(xs, MC_means, MC_means+MC_stds, color="grey", alpha=0.2, label = "+/- std")
plt.fill_between(xs, MC_means-MC_stds, MC_means, color="grey", alpha=0.2)

#TD
plt.plot(xs, TD_means, label="TD mean")
plt.fill_between(xs, TD_means, TD_means+TD_stds, color="grey", alpha=0.2)
plt.fill_between(xs, TD_means-TD_stds, TD_means, color="grey", alpha=0.2)
plt.ylabel("Estimation error")
plt.xlabel("Episode number")
plt.legend()
plt.show()
plt.savefig("Q4.1_MSE_vs_Episode.png", dpi=1000)



#Getting estimation error against total reward
#Storing
MC_mses_one_run = np.zeros(m)
TD_mses_one_run = np.zeros(m)

#Getting values and policies
MC_policy, MC_values, MC_tot_rewards = MC.solve(maze)
TD_policy, TD_values, TD_tot_rewards = TD.solve(maze)

#Updating mses
for i in range(m):
    MC_mses_one_run[i] = mean_squared_error(y_true=dp_value_final, y_pred=MC_values[i+1])  #j+1 because first is initial
    TD_mses_one_run[i] = mean_squared_error(y_true=dp_value_final, y_pred=TD_values[i+1])  #j+1 because first is initial    


#Creating plots
#MC plot
plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(MC_tot_rewards, MC_mses_one_run, label="MC", color="blue")
plt.ylabel("Estimation error")
plt.xlabel("Total non-discounted reward")
plt.legend()
plt.show()
plt.savefig("Q4.3_MC_Estimation_Err_vs_Total_Reward.png", dpi=1000)


#Creating plots
#TD plot
plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(TD_tot_rewards, TD_mses_one_run, label="TD", color="blue")
plt.ylabel("Estimation error")
plt.xlabel("Total non-discounted reward")
plt.legend()
plt.show()
plt.savefig("Q4.3_TD_Estimation_Err_vs_Total_Reward.png", dpi=1000)