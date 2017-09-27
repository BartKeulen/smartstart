:github_url: https://github.com/BartKeulen/smartstart

######
Setup
######

This page describes how to setup a computer with the smart exploration package.

==================
Installing Python
==================
The first step is to install python on your computer. Most Unix distributions (Debian, Ubuntu, Mac OS X) come with python pre-installed. Even if python comes pre-installed on your system I still recommend installing it through `Anaconda <https://www.anaconda.com/download/>`_. Anaconda is a package management tool for python that makes it easy to manage packages, dependencies and environments.

The recommended version of python is ``3.5`` or higher. Follow the instructions for you system on the `Anaconda <https://www.anaconda.com/download/>`_ site.

==================
Setting up Python
==================
It is good practice to make use of virtual environments in Python. A virtual environment is a sandbox with it's own python interpreter and package list, using virtual environments makes it easy to keep your python installation clean. Anaconda makes it really easy to create and manage virtual environments. First we create a virtual environment

``conda create --name smartstart python=3.5``

The environment will get it is own directory from Anaconda in which the interpreter and packages will be installed. To use the environment we simply have to activate it

``source activate smartstart``

Now our terminal will automatically use the interpreter and packages from this environment. You can see if your environment is activated by looking at your terminal, it will look like this:

``(smartstart) bartkeulen@ihmc:~$``

The environment can be deactivated by running

``source deactivate``

For more information on environments go `here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.

=====================================
Installing Smart Exploration Package
=====================================
To get started with the Smart Exploration package clone the `git repository
<https://github.com/BartKeulen/smartstart>`_ in a local directory on you file
system

``git clone https://github.com/BartKeulen/smartstart.git``

The package and its dependencies can now be installed by running

``pip install -r /path/to/smartstart/requirements.txt -e /path/to/smartstart/``

The ``-r **/requirements.txt`` option will install all dependencies needed by
the SmartStart package that are listed in the requirements.txt file. The ``-e
**/`` option installs the package in editable mode, all code changes have
immediate effect.

================
Updating Package
================

================
Generating Docs
================
Sphinx can be used to automatically generate documentation. This is done by
creating a link between the documentation and the source files. When source
files are created, deleted or the name is changed the link has to be updated.
First go into the docs directory

``cd /path/to/smartstart/docs/``

Execute the following command for generating/updating the necessary files

``sphinx-apidoc -f -o source/ ../smartstart/``

After the files have been generated/updated the documentation can be build with

``make clean && make html``