import os
import time
import random

import googleapiclient.discovery

project_id = 'infra-rhino-169522'
zone = 'us-east1-c'
bucket_name = 'drl-data'
service_account = 'drl-service-account@infra-rhino-169522.iam.gserviceaccount.com'


def list_instances(compute, project, zone):
    result = compute.instances().list(project=project, zone=zone).execute()
    return result['items']


def create_instance(compute, project, zone, instance_type, name, experiment, keep_instance=False):
    image_response = compute.images().getFromFamily(family='ubuntu-1604-lts', project='ubuntu-os-cloud').execute()
    source_disk_image = image_response['selfLink']

    # Configure the machine
    machine_type = "zones/%s/machineTypes/%s" % (zone, instance_type)
    startup_script = open(
        os.path.join(
            os.path.dirname(__file__), 'startup-script.sh'), 'r').read()

    config = {
        'name': name,
        'machineType': machine_type,
        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': source_disk_image
                }
            }
        ],
        'networkInterfaces': [{
            'network': 'global/networks/default',
            'accessConfigs': [
                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
            ]
        }],
        'serviceAccounts': [
            {
                'email': service_account,
                'scopes': ['https://www.googleapis.com/auth/cloud-platform']
            }
        ],
        'metadata': {
            'items': [
                {
                    'key': 'startup-script',
                    'value': startup_script
                },
                {
                    'key': 'experiment',
                    'value': experiment
                },
                {
                    'key': 'zone',
                    'value': zone
                },
                {
                    'key': 'instance',
                    'value': name
                },
                {
                    'key': 'keep',
                    'value': keep_instance
                }
            ]
        }
    }

    return compute.instances().insert(
        project=project,
        zone=zone,
        body=config
    ).execute()


def delete_instance(compute, project, zone, name):
    return compute.instances().delete(
        project=project,
        zone=zone,
        instance=name).execute()


def wait_for_operation(compute, project, zone, operation):
    print('Starting up server...')
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print("Starting experiment...")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)


def run_on_gcloud(experiment, instance_type='n1-highcpu-8', keep_instance=False):
    instance_name = "drl-tmp-instance-" + str(random.randint(1000, 9999))
    compute = googleapiclient.discovery.build('compute', 'v1')

    print('Creating instance.')

    operation = create_instance(compute, project_id, zone, instance_type, instance_name, experiment, keep_instance)
    wait_for_operation(compute, project_id, zone, operation['name'])


if __name__ == "__main__":
    run_on_gcloud('test_experiment')