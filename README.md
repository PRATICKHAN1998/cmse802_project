# cmse802_project
This is the main repository for the CMSE-802 final project

Comparative Analysis of Numerical Methods for Solving the Time-Dependent Schrödinger Equation in Quantum Tunneling Dynamics

## Project Description
This project implements numerical solutions to the Time-Dependent Schrödinger Equation (TDSE) using three different methods:
- Split-Operator Method
- Crank-Nicholson Method

These methods are applied to study quantum tunneling dynamics in various potentials:
- Double-Well Potential (with varying depth and width)
- Harmonic Potential Well
- Step Potential Well

Additionally, the project explores advanced cases, such as the effects of magnetic and electric fields and simple time-dependent fields on quantum wave packet dynamics.

---

## Project Objectives
1. Implementation of two different numerical methods (Split-Operator and Crank-Nicholson methods) to solve TDSE.
2. Application of these methods to study tunneling effects in different potential wells.
3. Comparision of these methods in terms of accuracy, stability, efficiency, and numerical convergence.
4. Permitting time extending these ideas for the study to include magnetic/electric field interactions and time-dependent fields.

---

## Folder Structure
cmse802_project
 
1.  methods  # Folder for different numerical methods

    1a. split_operator.py # Split-Operator Method Implementation

    1b. crank_nicholson.py # Crank-Nicholson Method Implementation

2.  examples/ # Folder for applications on different potentials

    2a. double_well.py # Double-Well Potential Example

    2b. harmonic_well.py # Harmonic Potential Example
    
    2c. step_potential.py # Step Potential Example

3.  advanced_cases/ # Folder for more complex scenarios

    3a. magnetic.py # Interaction with Magnetic Fields

    3b. electric.py # Interaction with Electric Fields

    3c. time_dependent.py # Time-Dependent Field Effect

4.  README.md # Project documentation

## Requirements
The required Python libraries are s follows:

numpy  
scipy  
matplotlib  
seaborn 

## Instructions for running the code

## Running Individual Methods:

python methods/split_operator.py

python methods/crank_nicholson.py


## Running Specific Potential Examples

python examples/double_well.py

python examples/harmonic_well.py

python examples/step_potential.py


## Running Advanced Cases

python advanced_cases/magnetic.py

python advanced_cases/electric.py

python advanced_cases/time_dependent.py


