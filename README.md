# **HotPP: High order tensor Passing Potential**
[![Documentation Status](https://readthedocs.org/projects/hotpp/badge/?version=latest)](https://hotpp.readthedocs.io/en/latest/?badge=latest)

# Introduction
`HotPP` is an open-source package designed for constructing message passing network interatomic potentials. It facilitates the utilization of arbitrary order Cartesian tensors as messages while maintaining equivalence maintenance.

# Current Features
* Building machine learning potentials for molecular and periodic systems;
* Learning dipole moments and polarizability tensors;
* Interface to LAMMPS and ASE;


# Documentation
* An overview of code documentation and tutorials for getting started with `HotPP` can be found [here](https://hotpp.readthedocs.io/en/latest/) folder.

# Installation
<span id= "using_pip"> </span>

## Use pip 
You can use https:
```shell
$ pip install git+https://gitlab.com/bigd4/hotpp.git
```
or use [ssh](https://docs.gitlab.com/ee/user/ssh.html)
```shell
$ pip install git+ssh://git@gitlab.com/bigd4/hotpp.git
```
Your may need to add `--user` if you do not have the root permission. Or use `--force-reinstall` if you already  have `HotPP` (add `--no-dependencies` if you do not want to reinstall the dependencies).


## From Source
1. Use git clone to get the source code:
```shell
$ git clone https://gitlab.com/bigd4/hotpp.git
```
Alternatively, you can download the source code from website.

2. Go into the directory and install with pip:
```shell
$ pip install -e .
```
pip will read **setup.py** in your current directory and install. The `-e` option means python will directly import the module from the current path, but not copy the codes to the default lib path and import the module there, which is convenient for modifying in the future. If you do not have the need, you can remove the option.
<!-- 
## Offline package

We provide an offline package in the [release](https://gitlab.com/bigd4/magus/-/releases). You can also use [conda-build](https://docs.conda.io/projects/conda-build/en/latest/) and [constructor](https://conda.github.io/constructor/) to make it by yourself as described [here](https://gitlab.com/bigd4/magus/-/tree/master/conda).  
After get the package,
```shell
$ chmod +x magus-***-Linux-x86_64.sh
$ ./magus-***-Linux-x86_64.sh
```
and follow the guide. -->
## Check
You can use 
```shell
$ hotpp -v
```
to check if you have installed successfully

## Update
If you installed by pip, use:
```shell
$ hotpp update
```
If you installed from source, use:
```shell
$ cd <path-to-magus-package>
$ git pull origin master
```
<!-- 
# Environment variables
## Job management system
Add
```shell
$ export JOB_SYSTEM=LSF/SLURM/PBS
```
in your `~/.bashrc` according to your job management system (choose one of them).  

## Auto completion
Put [`auto_complete.sh`](https://gitlab.com/bigd4/magus/-/blob/master/magus/auto_complete.sh) in your `PATH` like:
```shell
source <your-path-to>/auto_complete.sh
``` -->

# Interface
`HotPP` now support [ASE](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators) and [lammps](https://www.lammps.org/). 
## ASE
## LAMMPS

# Contributors
HotPP is developed by Prof. Jian Sun's group at the School of Physics at Nanjing University.

The contributors are:
- Jian Sun
- Junjie Wang
- Yong Wang
- Haoting Zhang
- Ziyang Yang
- Zhixin Liang
- Jiuyang Shi

# Citations
| Reference | cite for what                         |
| --------- | ------------------------------------- |
| [1]    | for any work that used `HotPP`        |


# Reference

[1] 1. Wang, J. et al. E(n)-Equivariant Cartesian Tensor Passing Potential. Preprint at http://arxiv.org/abs/2402.15286 (2024).
 (https://arxiv.org/abs/2402.15286)
