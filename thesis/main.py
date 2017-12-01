import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from thesis.config import *
import thesis.img_gridworlds as img_gridworlds
import thesis.triangularfeature as triangularfeature
import thesis.exploration as exploration


parser = argparse.ArgumentParser()
parser.add_argument('-all', action='store_true')
parser.add_argument('-gridworlds', action='store_true')
parser.add_argument('-triangularfeature', action='store_true')
parser.add_argument('-exploration', action='store_true')

args = parser.parse_args()

# Create images gridworlds
if args.all or args.gridworlds:
    img_gridworlds.main(show_plot=SHOW_PLOT, save=SAVE, fp=IMG_DIR)

# Create image triangular feature
if args.all or args.triangularfeature:
    triangularfeature.main(show_plot=SHOW_PLOT, save=SAVE, fp=IMG_DIR)

# Create image results exploration
if args.all or args.exploration:
    exploration.main(show_plot=SHOW_PLOT, save=SAVE, fp=IMG_DIR)

