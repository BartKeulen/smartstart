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
instructions for you system on the Anaconda site.

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

from the docs folder execute the following command for generating the
necessary files for the api

``sphinx-apidoc -o source/ ../smartexploration/``

After the files have been generated the documentation can be build with

``make html``