import os
import time

import googleapiclient.discovery


def list_instances(compute, project, zone):
    result = compute.instances().list(project=project, zone=zone).execute()
    return result['items']


def create_instance(compute, project, image, zone, name, experiment):
    image_response = compute.images().getFromFamily(family='ubuntu-1604-lts', project='ubuntu-os-cloud').execute()
    source_disk_image = image_response['selfLink']

    # Configure the machine
    machine_type = "zones/%s/machineTypes/n1-standard-1" % zone
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
                'email': 'drl-service-account@infra-rhino-169522.iam.gserviceaccount.com',
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
            print("Experiment started.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)


def main(project, zone, image, instance_name, experiment, wait=True):
    compute = googleapiclient.discovery.build('compute', 'v1')

    print('Creating instance.')

    operation = create_instance(compute, project, image, zone, instance_name, experiment)
    wait_for_operation(compute, project, zone, operation['name'])


if __name__ == "__main__":
    project_id = 'infra-rhino-169522'
    zone = 'us-east1-c'
    bucket_name = 'drl-data'
    image_name = 'drl-image'
    instances = ['drl-instance-2']

    main(project_id, zone, 'drl-image', 'test-instance-2', 'test_experiment')