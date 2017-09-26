:github_url: https://github.com/BartKeulen/smartstart

######
Setup
######

This page describes how to setup a computer with the smart exploration
package.

==================
Installing Python
==================
The first step is to install python on your computer. Most Unix
distributions (Debian, Ubuntu, Mac OS X) come with python pre-installed.
Even if python comes pre-installed on your system I still recommend
installing it through `Anaconda <https://www.anaconda.com/download/>`_.
Anaconda is a package management tool for python that makes it easy to manage
packages, dependencies and environments.

The recommended version of python is ``3.5`` or higher. Follow the
instructions for you system on the `Anaconda <https://www.anaconda.com/download/>`_ site.

==================
Setting up Python
==================


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