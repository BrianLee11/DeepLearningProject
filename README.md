# Objective of project
Using the same dataset, develop models to improve the prediction accuracy from the benchmark model (informer model). 


# Introduction
This project is for Deep Learning Spring 2024 Final Project. A novel deep model is developed and compared with other benchmark methods in time series forecasting. All these models are tested in wind power forecasting task.

Wind power is one of the most installed renewable energy resources in the world, and the accuracy of wind power forecasting method directly affects dispatching and operation safety of the power grid.




# User Guide
## Environment Setup   
1. Operating Platform

    The project has been tested on Google Colab.
2. Python version

    python >= 3.7

## Part 1: Benchmark model
Objective: This model and its prediction result would be the baseline to be outperformed. 
- **Informer model(a encoder + decoder model)** and result using this model are saved in the folder 1_benchmark_result. 

## Part 2: Model Training and Testing with the demo script
Objective: These models are developed to outperform prediction accuracy from the part 1's benchmark model. 
- Training models: **RNN (residual neural network), LSTM(long-short term memory), Modified GRU(encoder + decoder) models**
- These files are saved in the folder 2_improved_model

