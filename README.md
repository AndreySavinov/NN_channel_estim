# NN_channel_estim
Diploma project - Neural Network for Channel Estimation in QuaDRiGa model.
It contains several folders:

### python_server
- `nn_experiments.py` is for training and testing NN without Matlab;
- `simple_algo_detector.py` contains network architecture and functions for features calculations using PyTorch;
- `data_load.py` is required for separating real and imaginary part of the signal;
- `algo_runner_train_test.py` is actually client parts providing interaction between NN in python with signal processing in Matlab;
- `helpers.py` contains some auxiliary functions.

### matlab_client 
- `CE_TTI_client.m` is a simple client allowing to generate channel in Matlab, send it to python server for channel estimation and get a response in json-format;
- `CE_TTI_det2_SRS.m` is advanced client performing beamforming;
- `tester_det_SRS.m` is a demo allowing to assess the performance of Neural Network in Matlab, i.e., with a bigger variety of channels, via EVM error.


### Quadriga_2_0
Main file is `GENERATE_CHANNEL.m`. It is obviously used to generate quadriga channels with some parameters that can be change and provide different channel models.
The forlder contains its own `read_me`.
