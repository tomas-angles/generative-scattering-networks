# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import create_name_experiment
from GSN import GSN

parameters = dict()
parameters['dataset'] = 'celeba'
parameters['dataset_attribute'] = '64_rgb_65536_8192'
parameters['dim'] = 2048
parameters['embedding_attribute'] = 'PSKOJ4_innEps10m8_dim{}'.format(parameters['dim'])
# parameters['embedding_attribute'] = 'PSKJ4_randproj_{0}_pca_norm_{0}'.format(parameters['dim'])
# parameters['embedding_attribute'] = 'SJ4_pca_norm_{0}'.format(parameters['dim'])
# parameters['embedding_attribute'] = 'SJ4_randproj_{0}_pca_norm_{0}'.format(parameters['dim'])
parameters['nb_channels_first_layer'] = 32

parameters['name_experiment'] = create_name_experiment(parameters, 'pilot_disentangling_scales_nw4')

gsn = GSN(parameters)
# gsn.train()
# gsn.save_originals()
# gsn.generate_from_model(591)
# gsn.compute_errors(60)
# gsn.analyze_model(404)
gsn.conditional_generation(591)
