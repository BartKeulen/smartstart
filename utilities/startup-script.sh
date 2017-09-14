apt-get update
apt-get -y install python3-pip python3-tk --fix-missing

#EXPERIMENT=$(curl http://metadata/computeMetadata/v1/instance/attributes/experiment -H "Metadata-Flavor: Google")
#ZONE=$(curl http://metadata/computeMetadata/v1/instance/attributes/zone -H "Metadata-Flavor: Google")
#INSTANCE=$(curl http://metadata/computeMetadata/v1/instance/attributes/instance -H "Metadata-Flavor: Google")
KEEP=$(curl http://metadata/computeMetadata/v1/instance/attributes/keep -H "Metadata-Flavor: Google")
#
#git clone https://github.com/BartKeulen/smartstart.git
#cd smartstart
#pip3 install . -r requirements.txt
#
#python3 ./experiments/$EXPERIMENT/experiment.py

echo "Begin of startup script"

echo $KEEP

#if [! $KEEP]
#then
#    gcloud compute instances delete $INSTANCE --zone=$ZONE --quiet
#fi