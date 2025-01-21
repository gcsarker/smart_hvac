# Development of an IoT-Driven Deep Learning Framework for Smart Energy Prediction in Commercial Buildings

This repository provides the dataset and code for predicting the energy consumption of air conditioners in an office room environment. Additionally, it includes an energy optimization strategy designed to balance energy efficiency and occupant thermal comfort.

## Dataset 
The training and testing datasets are available in the 'dataset' directory in CSV format. The testing dataset is divided into five separate files, each representing one day. Each file contains minute-by-minute data on air conditioner operational and corresponding indoor-outdoor weather parameters.

## Using our system
To install the dependencies listed in the requirements.txt file, use the following command in your terminal or command prompt (Please make sure you have Python and pip installed on your system.):
```
pip install -r requirements.txt
```


- **Training.ipynb**  
  - Use this notebook to train the energy consumption prediction model.

- **Testing.ipynb**  
  - Use this notebook to evaluate the system's performance across the test days.

- **air_conditioner.py**  
  - Contains the thermal modeling of the air conditioner, calculating energy consumption at each timestep.

- **model_definition.py**  
  - Includes definitions for nine different models, including the proposed Attention-LSTM model.

