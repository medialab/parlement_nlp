# Dataset

This sub-folder contains the pipeline to build the datasets used in our projet.

Required non-python dependencies :
- uv
- xan
- curl

## Install python dependencies

Sync uv virtual environment using :

```bash
$ uv sync
```

## Usage

In the Makefile are all the commands of the pipeline. All of them can be runned using :

```bash
$ make all
```

Or independently with :

```bash
$ make download # download all the files needed for the pipeline from Assemblée Nationale servers
$ make build_open_data # build the CSV and JSON files based on downloaded raw data
$ make parse # parse the debates and extract their metadata 
$ make align # align the parsed debates with laws and votes
$ make dataset # turn parsed and aligned debates into single dataset
$ make paires-triplets # turn the single dataset into paires and triplets dataset for training
````

