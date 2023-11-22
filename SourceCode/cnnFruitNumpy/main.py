import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.fruit import load_fruit
from convnet import ConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_fruit()

network = ConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

network.save_params("convnet_params.pkl")
