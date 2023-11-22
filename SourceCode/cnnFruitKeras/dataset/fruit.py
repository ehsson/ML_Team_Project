from PIL import Image as img
from numpy import asarray
import numpy as np
import os
np.set_printoptions(threshold=np.inf)

def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.

    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def load_fruit():

    x_train, t_train, x_test, t_test = [],[],[],[]
    base_dir = "./dataset/"
    fruit=['apple','banana','blueberry','cherry','coconut','grape','kiwi','lemon','pear','tomato']

    for i in range(len(fruit)):
        file_list = os.listdir(base_dir+fruit[i])
        for j in range(800):
            image=img.open(base_dir+fruit[i]+"/"+file_list[j])
            data=asarray(image)
            x_train.append(data)
            t_train.append(i)
        for j in range(800, 1000):
            image=img.open(base_dir+fruit[i]+"/"+file_list[j])
            data=asarray(image)
            x_test.append(data)
            t_test.append(i)

    x_train, t_train = np.array(x_train), np.array(t_train)
    x_train = x_train.transpose(0, 3, 2, 1)
    x_train, t_train = shuffle_dataset(x_train, t_train)

    x_test, t_test = np.array(x_test), np.array(t_test)
    x_test = x_test.transpose(0, 3, 2, 1)
    x_test, t_test = shuffle_dataset(x_test, t_test)

    return (x_train, t_train), (x_test, t_test)
