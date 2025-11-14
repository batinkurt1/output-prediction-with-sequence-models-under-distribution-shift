from data_collection.generate_training_data import generate_training_data
from training.train import train
from testing.test import test

generate_data_flag = False
train_flag = False
test_flag = True

if __name__ == "__main__":
    if generate_data_flag:
        generate_training_data()
    if train_flag:
        train()
    if test_flag:
        test()