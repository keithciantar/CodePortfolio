Reinforcement Learning ReadMe

All code is found in rl2023

For each exercise - the main algorithm (agent) implementation can be found in agent.py


Exercise 2
Implementation of the Q-Learning and on-policy first-visit Monte Carlo algorithms to solve OpenAI Gym Taxi-v3 environment.
The environment is considered solved when the agent achieves return >= 7.
To run Q-Learning algo, execute the train_q_learning.py script.
To run on-policy first-visit Monte Carlo algo, execute the train_monte_carlo.py script.


Exercise 3 - Deep Reinforcement Learning 
Implementation of DQN and ReInforce to solve both OpenAI Gym CartPole and Acrobot environments. The default setup: REINFORCE agent is applied to the Acrobot environment and DQN agent is applied to the CartPole environment.
Environment Acrobat is considered solved when the agent achieves a return >= -400.
Environment CartPole is considered solved when the agent achieves a return >= 390.
To run ReInforce-algo, execute the train_reinforce.py script.
To run DQN-algo, execute the train_dqn.py script.


Exercise 4 Continuous Deep Reinforcement Learning 
Implementation of Deep Deterministic Policy Gradient (DDPG to solve Bipedal Walker control task) 
With standard (assignment dictated) parameters Environment is considered solved if the agent achieves a return >= -300. Our agent achieves >= -100.
Training the agent:  execute train_ddpg.py script in exercise 4. (it takes approx. 1 hr)
To visualise the trained agent interacting with the environment: execute evaluate_ddpg.py script in exercise 4.


Exercise 5 Hyperparameter tuning of Exercise 4
Implementation of various hyp-tuning methods to improve performance while keeping same neural network architecture. 
Performance improvement to 300 and agent solves the environment. 
Hyp-tuning the agent:  execute train_ddpg.py script in exercise 5. (it takes approx. 1 hr for each run for each config)
To see the hyp-tuned agent interacting with the environment, execute the evaluate_ddpg.py script in exercise 5.
