import argparse

from smartexploration.utilities.gcloud import run_on_gcloud

parser = argparse.ArgumentParser(description="Runs experiment on Google Compute Cloud. A new instance is created and deleted after completion. Results are saved on google cloud storage.")
parser.add_argument('experiment', type=str, help='Name of the experiment. The experiment must have its own folder with the provided name within the experiments folder. Within the provided experiment folder should be a python file experiment.py that is going to be executed on the instance.')
parser.add_argument('--instance_type', type=str, default='n1-highcpu-8', help='Google Compute Engine instance type, default is n1-highcpu-8. Available types can be found here: https://cloud.google.com/compute/pricing')
parser.add_argument('--keep_instance', action='store_true', default=False, help="Set to True to keep the instance on completion of the experiment.")

args = parser.parse_args()

run_on_gcloud(args.experiment, args.instance_type, args.keep_instance)
