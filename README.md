# Predicting Chaotic Dynamics of a Double Pendulum using LSTMs

This repository contains a quantitative pipeline for simulating the nonlinear dynamics of a double pendulum and evaluating the predictive capacity of Long Short-Term Memory (LSTM) networks under partial information constraints.

## Objective
The primary goal is to determine if a recurrent neural network can infer the hidden constraints of a chaotic system. Specifically, the project evaluates the model's ability to extrapolate the trajectory of the lower mass (Mass 2) when the kinetic state of the upper mass (Mass 1) is entirely blinded from the input features.

## Methodology
1.  **Physics Engine (`src/pendulum_env.py`)**: An object-oriented simulation utilizing `scipy.integrate.solve_ivp` to solve the Lagrangian equations of motion for the double pendulum.
2.  **Preprocessing (`src/lstm_pipeline.py`)**: Time-series phase space data is standardized to zero mean and unit variance. Sliding windows are generated using highly optimized, zero-copy C-level memory views (`numpy.lib.stride_tricks`).
3.  **Modeling**: A strictly causal LSTM architecture, trained with early stopping to prevent data leakage and overfitting. Extrapolation robustness is tested across both low-energy (periodic) and high-energy (chaotic) regimes.

## Repository Structure
* `src/`: Core object-oriented simulation and machine learning pipeline modules.
* `notebooks/`: Exploratory Data Analysis and generation of phase-space/extrapolation error plots.

## Installation & Usage
```bash
git clone [https://github.com/](https://github.com/)yasin21shah/double-pendulum-lstm.git
cd double-pendulum-lstm
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt