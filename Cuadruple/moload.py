import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

model = load_model('modelo.h5')