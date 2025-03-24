# cmse802_project
This is the main repository for the CMSE-802 final project

Comparative Analysis of Numerical Methods for Solving the Time-Dependent Schrödinger Equation in Quantum Tunneling Dynamics

## Project Description
This project implements numerical solutions to the Time-Dependent Schrödinger Equation (TDSE) using two different methods:
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
 
1.  methods  # Folder for different numerical methods and different potential wells
    
    1a.crank_nicolson_square_well_1D.py.py # Crank-Nicolson Method for infinite square potential well 

    1b. crank_nicolson_harmonic_well_1D.py.py # Crank-Nicolson Method for harmonic potential well

    1c. crank_nicolson_step_well_1D.py # Crank-Nicolson Method for step potential well

    1d. crank_nicolson_double_well_1D.py # Crank-Nicolson Method for double well


2.  advanced_cases/ # Folder for more complex scenarios

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

python src/methods/crank_nicolson_square_well_1D.py

python src/methods/crank_nicolson_double_well_1D.py

python src/methods/crank_nicolson_harmonic_well_1D.py

python src/methods/crank_nicolson_step_well_1D.py

## Running Advanced Cases

python src/dvanced_cases/magnetic.py

python src/advanced_cases/electric.py

python src/advanced_cases/time_dependent.py


