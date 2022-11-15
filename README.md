# Campfires

## Installation

Prior to installing WOW, the following packages must be installed from their respective GitHub repositories: 

 * [watroo](https://github.com/frederic-auchere/wavelets) (à trous wavelets transforms)
 * [rectify](https://github.com/frederic-auchere/rectify) (geometric remapping)

Then, after cloning the present repository:

```shell
pip install .
```

will take care of the other dependencies. Or if you want to be able to edit & develop (requires reloading the package)

```shell
pip install -e .
```

The installation process will create a `wow` executable that can be run from the command line of the virtual environment (more below):

## Usage
