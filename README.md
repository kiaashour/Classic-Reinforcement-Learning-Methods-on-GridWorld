# Classic Reinforcement Learning Methods on GridWorld

This repository implements classic reinforcement learning (dynamic programming, Monte Carlo, and temporal difference methods) on GridWorld. The methods implemented include:

1. Dynamic programming - policy iteration, value iteration

2. Monte Carlo - Monte Carlo first-visit

3. Temporal difference - Q-learning

A discussions of these methods can be found in [Sutton and Barto, 2018](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf).

---

## Contents of repository

1. The `utils` directory contains plotting functions as well as the GridWorld environment.

2. The `Methods` directory contains scripts for:

  1. Dynamic programming methods on the GridWorld environment

  2. Monte Carlo methods on the GridWorld environment  

  3. Temporal difference methods on the GridWorld environment

  4. Analysis of the different methods

---

## Prerequisites

Before you begin, ensure that you have the following:

- Python 3.8 or higher
- Virtualenv (optional, but recommended)

---

## Setting up a virtual environment

It is recommended to use a virtual environment to keep the dependencies for this project separate from other projects on your system. To set up a virtual environment:

1. If you don't have virtualenv installed, run `pip install virtualenv`
2. Navigate to the project directory and create a virtual environment by running `virtualenv env`
3. Activate the virtual environment by running `source env/bin/activate`

---

## Installing dependencies

To install the dependencies for this project, run the following command:

'pip install -r requirements.txt'

This will install all of the packages listed in the `requirements.txt` file.

---

## Cloning the repository

To clone this repository, run the following command:

`git clone https://github.com/user/repo.git`
