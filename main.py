# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils import create_name_experiment
from GSN import GSN

parameters = dict()
parameters['dataset'] = 'celebA_128'
parameters['train_attribute'] = '65536'
parameters['test_attribute'] = '2048_after_65536'
parameters['dim'] = 512
parameters['embedding_attribute'] = 'ScatJ4_projected{}_1norm'.format(parameters['dim'])
parameters['nb_channels_first_layer'] = 32

parameters['name_experiment'] = create_name_experiment(parameters, 'NormL1')

gsn = GSN(parameters)
# gsn.train()
# gsn.save_originals()
gsn.generate_from_model(404)
# gsn.compute_errors(536)
# gsn.analyze_model(404)
