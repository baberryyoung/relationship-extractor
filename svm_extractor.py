import tensorflow as tf
import numpy as np
import math
from matplotlib import pyplot as plt
from tensorflow import flags

flags.DEFINE_integer('epoch', 50000, "number of epoch")
flags.DEFINE_float('lr', 0.01, "learning rate")
flags.DEFINE_integer('padding', 0.1, "padding")
flags.DEFINE_integer('batch', 200, "batch size")
flags.DEFINE_integer('dim', 128,'dimension')
flags.DEFINE_integer('n_classes', 2,'number of classes')
FLAGS = flags.FLAGS

#svm.drawresult(x_data)
