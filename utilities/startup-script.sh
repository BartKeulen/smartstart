apt-get update
apt-get -y install python3-pip python3-tk --fix-missing

EXPERIMENT=$(curl http://metadata/computeMetadata/v1/instance/attributes/experiment -H "Metadata-Flavor: Google")

git clone https://github.com/BartKeulen/smartstart.git
cd smartstart
pip3 install . -r requirements.txt

