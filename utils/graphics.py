import numpy as np 
import random
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error 


class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

    """ Class to draw the Maze and the policies on it

    Arguments:
        shape {tuple} -- shape of the Maze
        locations {np.array} -- locations of each state
        default_reward {float} -- default reward for each state
        obstacle_locs {np.array} -- locations of the obstacles
        absorbing_locs {np.array} -- locations of the absorbing states
        absorbing_rewards {np.array} -- rewards of the absorbing states
        absorbing {np.array} -- absorbing states
    """

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    #Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    #Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    #Print the map to show it
    self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) #Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): #If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] #List of arrows corresponding to each possible action
      action_arrow = arrows[action] #Take the corresponding action
      location = self.locations[state] #Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') #Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) #Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): #If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] #Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') #Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): #Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) #Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) #Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): #If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] #List of arrows corresponding to each possible action
        action_arrow = arrows[action] #Take the corresponding action
        location = self.locations[state] #Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') #Place it on graph
      ax.title.set_text(title[subplot]) #Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): #Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) #Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) #Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): #If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] #Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') #Place it on graph
      ax.title.set_text(title[subplot]) #Set the title for the graoh given as argument
    plt.show()