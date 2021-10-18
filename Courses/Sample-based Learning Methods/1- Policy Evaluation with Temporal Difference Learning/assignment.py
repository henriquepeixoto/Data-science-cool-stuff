#!/usr/bin/env python
# coding: utf-8

# # Assignment: Policy Evaluation in Cliff Walking Environment
# 
# Welcome to the Course 2 Module 2 Programming Assignment! In this assignment, you will implement one of the fundamental sample and bootstrapping based model free reinforcement learning agents for prediction. This is namely one that uses one-step temporal difference learning, also known as TD(0). The task is to design an agent for policy evaluation in the Cliff Walking environment. Recall that policy evaluation is the prediction problem where the goal is to accurately estimate the values of states given some policy.
# 
# ### Learning Objectives
# - Implement parts of the Cliff Walking environment, to get experience specifying MDPs [Section 1].
# - Implement an agent that uses bootstrapping and, particularly, TD(0) [Section 2].
# - Apply TD(0) to estimate value functions for different policies, i.e., run policy evaluation experiments [Section 3].

# ## The Cliff Walking Environment
# 
# The Cliff Walking environment is a gridworld with a discrete state space and discrete action space. The agent starts at grid cell S. The agent can move (deterministically) to the four neighboring cells by taking actions Up, Down, Left or Right. Trying to move out of the boundary results in staying in the same location. So, for example, trying to move left when at a cell on the leftmost column results in no movement at all and the agent remains in the same location. The agent receives -1 reward per step in most states, and -100 reward when falling off of the cliff. This is an episodic task; termination occurs when the agent reaches the goal grid cell G. Falling off of the cliff results in resetting to the start state, without termination.
# 
# The diagram below showcases the description above and also illustrates two of the policies we will be evaluating.
# 
# <img src="cliffwalk.png" style="height:400px">

# ## Packages.
# 
# We import the following libraries that are required for this assignment. We shall be using the following libraries:
# 1. jdc: Jupyter magic that allows defining classes over multiple jupyter notebook cells.
# 2. numpy: the fundamental package for scientific computing with Python.
# 3. matplotlib: the library for plotting graphs in Python.
# 4. RL-Glue: the library for reinforcement learning experiments.
# 5. BaseEnvironment, BaseAgent: the base classes from which we will inherit when creating the environment and agent classes in order for them to support the RL-Glue framework.
# 6. Manager: the file allowing for visualization and testing.
# 7. itertools.product: the function that can be used easily to compute permutations.
# 8. tqdm.tqdm: Provides progress bars for visualizing the status of loops.
# 
# **Please do not import other libraries** this will break the autograder.
# 
# **NOTE: For this notebook, there is no need to make any calls to methods of random number generators. Spurious or missing calls to random number generators may affect your results.**

# In[4]:


# Do not modify this cell!

import jdc
import numpy as np
from rl_glue import RLGlue
from Agent import BaseAgent 
from Environment import BaseEnvironment  
from manager import Manager
from itertools import product
from tqdm import tqdm


# ## Section 1. Environment
# 
# In the first part of this assignment, you will get to see how the Cliff Walking environment is implemented. You will also get to implement parts of it to aid your understanding of the environment and more generally how MDPs are specified. In particular, you will implement the logic for:
#  1. Converting 2-dimensional coordinates to a single index for the state,
#  2. One of the actions (action up), and,
#  3. Reward and termination.
#  
# Given below is an annotated diagram of the environment with more details that may help in completing the tasks of this part of the assignment. Note that we will be creating a more general environment where the height and width positions can be variable but the start, goal and cliff grid cells have the same relative positions (bottom left, bottom right and the cells between the start and goal grid cells respectively).
# 
# <img src="cliffwalk-annotated.png" style="height:400px">
# 
# Once you have gone through the code and begun implementing solutions, it may be a good idea to come back here and see if you can convince yourself that the diagram above is an accurate representation of the code given and the code you have written.

# In[5]:


# ---------------
# Discussion Cell
# ---------------

# Create empty CliffWalkEnvironment class.
# These methods will be filled in later cells.
class CliffWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        raise NotImplementedError

    def env_start(self):
        raise NotImplementedError

    def env_step(self, action):
        raise NotImplementedError

    def env_cleanup(self):
        raise NotImplementedError
    
    # helper method
    def state(self, loc):
        raise NotImplementedError


# ## env_init()
# 
# The first function we add to the environment is the initialization function which is called once when an environment object is created. In this function, the grid dimensions and special locations (start and goal locations and the cliff locations) are stored for easy use later.

# In[14]:


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef env_init(self, env_info={}):\n        """Setup for the environment called when the experiment first starts.\n        Note:\n            Initialize a tuple with the reward, first state, boolean\n            indicating if it\'s terminal.\n        """\n        \n        # Note, we can setup the following variables later, in env_start() as it is equivalent. \n        # Code is left here to adhere to the note above, but these variables are initialized once more\n        # in env_start() [See the env_start() function below.]\n        \n        reward = None\n        state = None # See Aside\n        termination = None\n        self.reward_state_term = (reward, state, termination)\n        \n        # AN ASIDE: Observation is a general term used in the RL-Glue files that can be interachangeably \n        # used with the term "state" for our purposes and for this assignment in particular. \n        # A difference arises in the use of the terms when we have what is called Partial Observability where \n        # the environment may return states that may not fully represent all the information needed to \n        # predict values or make decisions (i.e., the environment is non-Markovian.)\n        \n        # Set the default height to 4 and width to 12 (as in the diagram given above)\n        self.grid_h = env_info.get("grid_height", 4) \n        self.grid_w = env_info.get("grid_width", 12)\n        \n        # Now, we can define a frame of reference. Let positive x be towards the direction down and \n        # positive y be towards the direction right (following the row-major NumPy convention.)\n        # Then, keeping with the usual convention that arrays are 0-indexed, max x is then grid_h - 1 \n        # and max y is then grid_w - 1. So, we have:\n        # Starting location of agent is the bottom-left corner, (max x, min y). \n        self.start_loc = (self.grid_h - 1, 0)\n        # Goal location is the bottom-right corner. (max x, max y).\n        self.goal_loc = (self.grid_h - 1, self.grid_w - 1)\n        \n        # The cliff will contain all the cells between the start_loc and goal_loc.\n        self.cliff = [(self.grid_h - 1, i) for i in range(1, (self.grid_w - 1))]\n        \n        # Take a look at the annotated environment diagram given in the above Jupyter Notebook cell to \n        # verify that your understanding of the above code is correct for the default case, i.e., where \n        # height = 4 and width = 12.')


# ## *Implement* state()
#     
# The agent location can be described as a two-tuple or coordinate (x, y) describing the agentâ€™s position. 
# However, we can convert the (x, y) tuple into a single index and provide agents with just this integer.
# One reason for this choice is that the spatial aspect of the problem is secondary and there is no need 
# for the agent to know about the exact dimensions of the environment. 
# From the agentâ€™s viewpoint, it is just perceiving some states, accessing their corresponding values 
# in a table, and updating them. Both the coordinate (x, y) state representation and the converted coordinate representation are thus equivalent in this sense.
# 
# Given a grid cell location, the state() function should return the state; a single index corresponding to the location in the grid.
# 
# 
# ```
# Example: Suppose grid_h is 2 and grid_w is 2. Then, we can write the grid cell two-tuple or coordinate
# states as follows (following the usual 0-index convention):
# |(0, 0) (0, 1)| |0 1|
# |(1, 0) (1, 1)| |2 3|
# Assuming row-major order as NumPy does,  we can flatten the latter to get a vector [0 1 2 3].
# So, if loc = (0, 0) we return 0. While, for loc = (1, 1) we return 3.
# ```

# In[15]:


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# -----------\n# Graded Cell\n# -----------\n\n# Modify the return statement of this function to return a correct single index as \n# the state (see the logic for this in the previous cell.)\ndef state(self, loc):\n    # your code here\n    return loc[0] * self.grid_w + loc[1]')


# In[16]:


# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

env = CliffWalkEnvironment()
env.env_init({ "grid_height": 4, "grid_width": 12 })

coords = [(0, 0), (0, 11), (1, 5), (3, 0), (3, 9), (3, 11)]
correct_outputs = [0, 11, 17, 36, 45, 47]

got = [env.state(s) for s in coords]
assert got == correct_outputs


# In[17]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.

np.random.seed(0)

env = CliffWalkEnvironment()
for n in range(100):
    # make a gridworld of random size and shape
    height = np.random.randint(2, 100)
    width = np.random.randint(2, 100)
    env.env_init({ "grid_height": height, "grid_width": width })
    
    # generate some random coordinates within the grid
    idx_h = np.random.randint(height)
    idx_w = np.random.randint(width)
    
    # check that the state index is correct
    state = env.state((idx_h, idx_w))
    correct_state = width * idx_h + idx_w
    
    assert state == correct_state


# ## env_start()
# 
# In env_start(), we initialize the agent location to be the start location and return the state corresponding to it as the first state for the agent to act upon. Additionally, we also set the reward and termination terms to be 0 and False respectively as they are consistent with the notion that there is no reward nor termination before the first action is even taken.

# In[18]:


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef env_start(self):\n    """The first method called when the episode starts, called before the\n    agent starts.\n\n    Returns:\n        The first state from the environment.\n    """\n    reward = 0\n    # agent_loc will hold the current location of the agent\n    self.agent_loc = self.start_loc\n    # state is the one dimensional state representation of the agent location.\n    state = self.state(self.agent_loc)\n    termination = False\n    self.reward_state_term = (reward, state, termination)\n\n    return self.reward_state_term[1]')


# ## *Implement* env_step()
# 
# Once an action is taken by the agent, the environment must provide a new state, reward and termination signal. 
# 
# In the Cliff Walking environment, agents move around using a 4-cell neighborhood called the Von Neumann neighborhood (https://en.wikipedia.org/wiki/Von_Neumann_neighborhood). Thus, the agent has 4 available actions at each state. Three of the actions have been implemented for you and your first task is to implement the logic for the fourth action (Action UP).
# 
# Your second task for this function is to implement the reward logic. Look over the environment description given earlier in this notebook if you need a refresher for how the reward signal is defined.

# In[45]:


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# -----------\n# Graded Cell\n# -----------\n\ndef isInBounds(x, y, width, height):\n    a = False\n    if (0 <= x < height -1) and (0 <= y <  width -1):\n        a=True\n    elif (x < height +1) and (y <  width +1):\n        a=True\n    return a\n\n# Fill in the code for action UP and implement the logic for reward and termination.\ndef env_step(self, action):\n    """A step taken by the environment.\n\n    Args:\n        action: The action taken by the agent\n\n    Returns:\n        (float, state, Boolean): a tuple of the reward, state,\n            and boolean indicating if it\'s terminal.\n    """\n    \n    x, y = self.agent_loc\n\n    # UP\n    if action == 0:\n        x = x -1\n        \n    # LEFT\n    elif action == 1:\n        y = y - 1\n        \n    # DOWN\n    elif action == 2:\n        x = x + 1\n        \n    # RIGHT\n    elif action == 3:\n        y = y + 1\n        \n    # Uh-oh\n    else: \n        raise Exception(str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!")\n\n    # if the action takes the agent out-of-bounds\n    # then the agent stays in the same state\n    if not isInBounds(x, y, self.grid_w, self.grid_h):\n        x, y = self.agent_loc\n        \n    # assign the new location to the environment object\n    self.agent_loc = (x, y)\n    \n    # by default, assume -1 reward per step and that we did not terminate\n    reward = -1\n    terminal = False\n    \n    # assign the reward and terminal variables \n    # - if the agent falls off the cliff (don\'t forget to reset agent location!)\n    # - if the agent reaches the goal state\n    if self.agent_loc == self.goal_loc: # Reached Goal!\n        terminal = True\n    elif self.agent_loc in self.cliff: # Fell into the cliff!\n        reward = -100\n        self.agent_loc = self.start_loc    \n    \n    self.reward_state_term = (reward, self.state(self.agent_loc), terminal)\n    return self.reward_state_term')


# In[46]:


# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

def test_action_up():
    env = CliffWalkEnvironment()
    env.env_init({"grid_height": 4, "grid_width": 12})
    env.agent_loc = (0, 0)
    env.env_step(0)
    assert(env.agent_loc == (0, 0))
    
    env.agent_loc = (1, 0)
    env.env_step(0)
    assert(env.agent_loc == (0, 0))
    
def test_reward():
    env = CliffWalkEnvironment()
    env.env_init({"grid_height": 4, "grid_width": 12})
    env.agent_loc = (0, 0)
    reward_state_term = env.env_step(0)
    assert(reward_state_term[0] == -1 and reward_state_term[1] == env.state((0, 0)) and
           reward_state_term[2] == False)
    
    env.agent_loc = (2, 1)
    reward_state_term = env.env_step(2)
    assert(reward_state_term[0] == -100 and reward_state_term[1] == env.state((3, 0)) and
           reward_state_term[2] == False)
    
    env.agent_loc = (2, 11)
    reward_state_term = env.env_step(2)
    assert(reward_state_term[0] == -1 and reward_state_term[1] == env.state((3, 11)) and
           reward_state_term[2] == True)
    
test_action_up()
test_reward()


# In[47]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.
np.random.seed(0)

env = CliffWalkEnvironment()
for n in range(100):
    # create a cliff world of random size
    height = np.random.randint(2, 100)
    width = np.random.randint(2, 100)
    env.env_init({"grid_height": height, "grid_width": width})
    
    # start the agent in a random location
    idx_h = 0 if np.random.random() < 0.5 else np.random.randint(height)
    idx_w = np.random.randint(width)
    env.agent_loc = (idx_h, idx_w)
    
    env.env_step(0)
    assert(env.agent_loc == (0 if idx_h == 0 else idx_h - 1, idx_w))


# In[48]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.
np.random.seed(0)

env = CliffWalkEnvironment()
for n in range(100):
    # create a cliff world of random size
    height = np.random.randint(4, 10)
    width = np.random.randint(4, 10)
    env.env_init({"grid_height": height, "grid_width": width})
    env.env_start()
    
    # start the agent near the cliff
    idx_h = height - 2
    idx_w = np.random.randint(1, width - 2)
    env.agent_loc = (idx_h, idx_w)
    
    r, sp, term = env.env_step(2)
    assert(r == -100 and sp == (height - 1) * width and term == False)

for n in range(100):
    # create a cliff world of random size
    height = np.random.randint(4, 10)
    width = np.random.randint(4, 10)
    env.env_init({"grid_height": height, "grid_width": width})
    env.env_start()
    
    # start the agent near the goal
    idx_h = height - 2
    idx_w = width - 1
    env.agent_loc = (idx_h, idx_w)
    
    r, sp, term = env.env_step(2)
    assert(r == -1 and sp == (height - 1) * width + (width - 1) and term == True)

for n in range(100):
    # create a cliff world of random size
    height = np.random.randint(4, 10)
    width = np.random.randint(4, 10)
    env.env_init({"grid_height": height, "grid_width": width})
    env.env_start()
    
    # start the agent in a random location
    idx_h = np.random.randint(0, height - 3)
    idx_w = np.random.randint(0, width - 1)
    env.agent_loc = (idx_h, idx_w)
    
    r, sp, term = env.env_step(2)
    assert(r == -1 and term == False)


# ## env_cleanup()
# 
# There is not much cleanup to do for the Cliff Walking environment. Here, we simply reset the agent location to be the start location in this function.

# In[49]:


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef env_cleanup(self):\n    """Cleanup done after the environment ends"""\n    self.agent_loc = self.start_loc')


# ## Section 2. Agent
# 
# In this second part of the assignment, you will be implementing the key updates for Temporal Difference Learning. There are two cases to consider depending on whether an action leads to a terminal state or not.

# In[50]:


# ---------------
# Discussion Cell
# ---------------

# Create empty TDAgent class.
# These methods will be filled in later cells.

class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        raise NotImplementedError
        
    def agent_start(self, state):
        raise NotImplementedError

    def agent_step(self, reward, state):
        raise NotImplementedError

    def agent_end(self, reward):
        raise NotImplementedError

    def agent_cleanup(self):        
        raise NotImplementedError
        
    def agent_message(self, message):
        raise NotImplementedError


# ## agent_init()
# 
# As we did with the environment, we first initialize the agent once when a TDAgent object is created. In this function, we create a random number generator, seeded with the seed provided in the agent_info dictionary to get reproducible results. We also set the policy, discount and step size based on the agent_info dictionary. Finally, with a convention that the policy is always specified as a mapping from states to actions and so is an array of size (# States, # Actions), we initialize a values array of shape (# States,) to zeros.

# In[51]:


get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef agent_init(self, agent_info={}):\n    """Setup for the agent called when the experiment first starts."""\n\n    # Create a random number generator with the provided seed to seed the agent for reproducibility.\n    self.rand_generator = np.random.RandomState(agent_info.get("seed"))\n\n    # Policy will be given, recall that the goal is to accurately estimate its corresponding value function. \n    self.policy = agent_info.get("policy")\n    # Discount factor (gamma) to use in the updates.\n    self.discount = agent_info.get("discount")\n    # The learning rate or step size parameter (alpha) to use in updates.\n    self.step_size = agent_info.get("step_size")\n\n    # Initialize an array of zeros that will hold the values.\n    # Recall that the policy can be represented as a (# States, # Actions) array. With the \n    # assumption that this is the case, we can use the first dimension of the policy to\n    # initialize the array for values.\n    self.values = np.zeros((self.policy.shape[0],))')


# # agent_start()
# 
# In agent_start(), we choose an action based on the initial state and policy we are evaluating. We also cache the state so that we can later update its value when we perform a Temporal Difference update. Finally, we return the action chosen so that the RL loop can continue and the environment can execute this action.

# In[52]:


get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef agent_start(self, state):\n    """The first method called when the episode starts, called after\n    the environment starts.\n    Args:\n        state (Numpy array): the state from the environment\'s env_start function.\n    Returns:\n        The first action the agent takes.\n    """\n    # The policy can be represented as a (# States, # Actions) array. So, we can use \n    # the second dimension here when choosing an action.\n    action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])\n    self.last_state = state\n    return action')


# ## *Implement* agent_step()
# 
# In agent_step(), the agent must:
# 
# - Perform an update to improve the value estimate of the previously visited state, and
# - Act based on the state provided by the environment.
# 
# The latter of the two steps above has been implemented for you. Implement the former. Note that, unlike later in agent_end(), the episode has not yet ended in agent_step(). in other words, the previously observed state was not a terminal state.

# In[53]:


get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# -----------\n# Graded Cell\n# -----------\n\ndef agent_step(self, reward, state):\n    """A step taken by the agent.\n    Args:\n        reward (float): the reward received for taking the last action taken\n        state (Numpy array): the state from the\n            environment\'s step after the last action, i.e., where the agent ended up after the\n            last action\n    Returns:\n        The action the agent is taking.\n    """\n    \n    # Hint: We should perform an update with the last state given that we now have the reward and\n    # next state. We break this into two steps. Recall for example that the Monte-Carlo update \n    # had the form: V[S_t] = V[S_t] + alpha * (target - V[S_t]), where the target was the return, G_t.\n    \n    target = reward + self.discount * self.values[state]\n    self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])    \n\n    # Having updated the value for the last state, we now act based on the current \n    # state, and set the last state to be current one as we will next be making an \n    # update with it when agent_step is called next once the action we return from this function \n    # is executed in the environment.\n\n    action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])\n    self.last_state = state\n\n    return action')


# ## *Implement* agent_end() 
# 
# Implement the TD update for the case where an action leads to a terminal state.

# In[54]:


get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# -----------\n# Graded Cell\n# -----------\n\ndef agent_end(self, reward):\n    """Run when the agent terminates.\n    Args:\n        reward (float): the reward the agent received for entering the terminal state.\n    """\n    \n    # Hint: Here too, we should perform an update with the last state given that we now have the \n    # reward. Note that in this case, the action led to termination. Once more, we break this into \n    # two steps, computing the target and the update itself that uses the target and the \n    # current value estimate for the state whose value we are updating.\n    \n    target = reward\n    self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])    ')


# ## agent_cleanup()
# 
# In cleanup, we simply reset the last state to be None to ensure that we are not storing any states past an episode.

# In[55]:


get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef agent_cleanup(self):\n    """Cleanup done after the agent ends."""\n    self.last_state = None')


# ## agent_message()
# 
# agent_message() can generally be used to get different kinds of information about an RLGlue agent in the interaction loop of RLGlue. Here, we conditonally check for a message matching "get_values" and use it to retrieve the values table the agent has been updating over time.

# In[56]:


get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# ---------------\n# Discussion Cell\n# ---------------\n\ndef agent_message(self, message):\n    """A function used to pass information from the agent to the experiment.\n    Args:\n        message: The message passed to the agent.\n    Returns:\n        The response (or answer) to the message.\n    """\n    if message == "get_values":\n        return self.values\n    else:\n        raise Exception("TDAgent.agent_message(): Message not understood!")')


# In[57]:


# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

# The following test checks that the TD check works for a case where the transition 
# garners reward -1 and does not lead to a terminal state. This is in a simple two state setting 
# where there is only one action. The first state's current value estimate is 0 while the second is 1.
# Note the discount and step size if you are debugging this test.
agent = TDAgent()
policy_list = np.array([[1.], [1.]])
agent.agent_init({"policy": np.array(policy_list), "discount": 0.99, "step_size": 0.1})
agent.values = np.array([0., 1.])
agent.agent_start(0)

reward = -1
next_state = 1
agent.agent_step(reward, next_state)

assert(np.isclose(agent.values[0], -0.001) and np.isclose(agent.values[1], 1.))

# The following test checks that the TD check works for a case where the transition 
# garners reward -100 and lead to a terminal state. This is in a simple one state setting 
# where there is only one action. The state's current value estimate is 0.
# Note the discount and step size if you are debugging this test.
agent = TDAgent()
policy_list = np.array([[1.]])
agent.agent_init({"policy": np.array(policy_list), "discount": 0.99, "step_size": 0.1})
agent.values = np.array([0.])
agent.agent_start(0)

reward = -100
next_state = 0
agent.agent_end(reward)

assert(np.isclose(agent.values[0], -10))


# In[58]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.

agent = TDAgent()
policy_list = [np.random.dirichlet(np.ones(10), size=1).squeeze() for _ in range(100)]

for n in range(100):
    gamma = np.random.random()
    alpha = np.random.random()
    agent.agent_init({"policy": np.array(policy_list), "discount": gamma, "step_size": alpha})
    agent.values = np.random.randn(*agent.values.shape)
    state = np.random.randint(100)
    agent.agent_start(state)
    
    for _ in range(100):
        prev_agent_vals = agent.values.copy()
        reward = np.random.random()
        if np.random.random() > 0.1:
            next_state = np.random.randint(100)
            agent.agent_step(reward, next_state)
            prev_agent_vals[state] = prev_agent_vals[state] + alpha * (reward + gamma * prev_agent_vals[next_state] - prev_agent_vals[state])
            assert(np.allclose(prev_agent_vals, agent.values))
            state = next_state
        else:
            agent.agent_end(reward)
            prev_agent_vals[state] = prev_agent_vals[state] + alpha * (reward - prev_agent_vals[state])
            assert(np.allclose(prev_agent_vals, agent.values))
            break


# ## Section 3. Policy Evaluation Experiments
# 
# Finally, in this last part of the assignment, you will get to see the TD policy evaluation algorithm in action by looking at the estimated values, the per state value error and after the experiment is complete, the Mean Squared Value Error curve vs. episode number, summarizing how the value error changed over time.
# 
# The code below runs one run of an experiment given env_info and agent_info dictionaries. A "manager" object is created for visualizations and is used in part for the autograder. By default, the run will be for 5000 episodes. The true_values_file is specified to compare the learned value function with the values stored in the true_values_file. Plotting of the learned value  function occurs by default after every 100 episodes. In addition, when true_values_file is specified, the value error per state and the root mean square value error will also be plotted.

# In[59]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# ---------------
# Discussion Cell
# ---------------

def run_experiment(env_info, agent_info,num_episodes=5000, experiment_name=None, plot_freq=100, true_values_file=None, value_error_threshold=1e-8):
    env = CliffWalkEnvironment
    agent = TDAgent
    rl_glue = RLGlue(env, agent)

    rl_glue.rl_init(agent_info, env_info)

    manager = Manager(env_info, agent_info, true_values_file=true_values_file, experiment_name=experiment_name)
    for episode in range(1, num_episodes + 1):
        rl_glue.rl_episode(0) # no step limit
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    values = rl_glue.agent.agent_message("get_values")
    return values


# The cell below just runs a policy evaluation experiment with the determinstic optimal policy that strides just above the cliff. You should observe that the per state value error and RMSVE curve asymptotically go towards 0. The arrows in the four directions denote the probabilities of taking each action. This experiment is ungraded but should serve as a good test for the later experiments. The true values file provided for this experiment may help with debugging as well.

# In[60]:


# ---------------
# Discussion Cell
# ---------------

env_info = {"grid_height": 4, "grid_width": 12, "seed": 0}
agent_info = {"discount": 1, "step_size": 0.01, "seed": 0}

# The Optimal Policy that strides just along the cliff
policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [1, 0, 0, 0]
for i in range(24, 35):
    policy[i] = [0, 0, 0, 1]
policy[35] = [0, 0, 1, 0]

agent_info.update({"policy": policy})

true_values_file = "optimal_policy_value_fn.npy"
_ = run_experiment(env_info, agent_info, num_episodes=5000, experiment_name="Policy Evaluation on Optimal Policy",
                   plot_freq=500, true_values_file=true_values_file)


# In[61]:


# -----------
# Graded Cell
# -----------

# The Safe Policy
# Hint: Fill in the array below (as done in the previous cell) based on the safe policy illustration 
# in the environment diagram. This is the policy that strides as far as possible away from the cliff. 
# We call it a "safe" policy because if the environment has any stochasticity, this policy would do a good job in 
# keeping the agent from falling into the cliff (in contrast to the optimal policy shown before). 

# build a uniform random policy
policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25

# build an example environment
env = CliffWalkEnvironment()
env.env_init(env_info)

# modify the uniform random policy
for h in range(env_info['grid_height']):
    for w in range(env_info['grid_width']):
        state = h * env_info['grid_width'] + w
        action = [1, 0, 0, 0]
        if h == 0:
            action = [0, 0, 0, 1]
        if w == env_info['grid_width'] - 1:
            action = [0, 0, 1, 0]
        policy[state] = action


# In[62]:


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.

width = env_info['grid_width']
height = env_info['grid_height']

# left side of space
for x in range(1, height):
    s = env.state((x, 0))
    
    # go up
    assert np.all(policy[s] == [1, 0, 0, 0])

# top of space
for y in range(width - 1):
    s = env.state((0, y))

    # go right
    assert np.all(policy[s] == [0, 0, 0, 1])
    
# right side of space
for x in range(height - 1):
    s = env.state((x, width - 1))
    
    # go down
    assert np.all(policy[s] == [0, 0, 1, 0])


# In[63]:


# ---------------
# Discussion Cell
# ---------------

agent_info.update({"policy": policy})
v = run_experiment(env_info, agent_info, experiment_name="Policy Evaluation On Safe Policy", num_episodes=5000, plot_freq=500)


# In[66]:


# ---------------
# Discussion Cell
# ---------------

# A Near Optimal Stochastic Policy
# Now, we try a stochastic policy that deviates a little from the optimal policy seen above. 
# This means we can get different results due to randomness.
# We will thus average the value function estimates we get over multiple runs. 
# This can take some time, upto about 5 minutes from previous testing. 
# NOTE: The autograder will compare . Re-run this cell upon making any changes.

env_info = {"grid_height": 4, "grid_width": 12}
agent_info = {"discount": 1, "step_size": 0.01}

policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [0.9, 0.1/3., 0.1/3., 0.1/3.]
for i in range(24, 35):
    policy[i] = [0.1/3., 0.1/3., 0.1/3., 0.9]
policy[35] = [0.1/3., 0.1/3., 0.9, 0.1/3.]
agent_info.update({"policy": policy})
agent_info.update({"step_size": 0.01})


# In[67]:


# ---------------
# Discussion Cell
# ---------------
env_info['seed'] = 0
agent_info['seed'] = 0
v = run_experiment(env_info, agent_info,
               experiment_name="Policy Evaluation On Optimal Policy",
               num_episodes=5000, plot_freq=100)


# ## Wrapping Up
# Congratulations, you have completed assignment 2! In this assignment, we investigated a very useful concept for sample-based online learning: temporal difference. We particularly looked at the prediction problem where the goal is to find the value function corresponding to a given policy. In the next assignment, by learning the action-value function instead of the state-value function, you will get to see how temporal difference learning can be used in control as well.
