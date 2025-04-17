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
    1. split_operator_square_well_1D.py.py # Split-Operator Method for infinite square potential well

    2. split_operator_harmonic_well_1D.py.py # Split-Operator Method for infinite square potential well

    3. split_operator_step_well_1D.py.py # Split-Operator Method for infinite square potential well

    4. split_operator_double_well_1D.py.py # Split-Operator Method for infinite square potential well

    5. crank_nicolson_square_well_1D.py.py # Crank-Nicolson Method for infinite square potential well 

    6. crank_nicolson_harmonic_well_1D.py.py # Crank-Nicolson Method for harmonic potential well

    7. crank_nicolson_step_well_1D.py # Crank-Nicolson Method for step potential well

    8. crank_nicolson_double_well_1D.py # Crank-Nicolson Method for double well


2.  advanced_cases/ # Folder for more complex scenarios

    2a. magnetic.py # Interaction with Magnetic Fields

    2b. electric.py # Interaction with Electric Fields


3.  README.md # Project documentation

## Requirements
The required Python libraries are s follows:

numpy  
scipy  
matplotlib  
seaborn
ffmpeg 

## Instructions for running the code

## Running Individual Methods:

python src/methods/split_operator_method/split_operatore_square_well.py

python src/methods/split_operator_method/split_operator_double_well.py

python src/methods/split_operator_method/split_operator_harmonic_well.py

python src/methods/split_operator_method/split_operator_step_well.py

python src/methods/crank_nicolson_method/crank_nicolson_square_well.py

python src/methods/crank_nicolson_method/crank_nicolson_double_well.py

python src/methods/crank_nicolson_method/crank_nicolson_harmonic_well.py

python src/methods/crank_nicolson_method/crank_nicolson_step_well.py

## Running Advanced Cases

python src/dvanced_cases/magnetic.py

python src/advanced_cases/electric.py


