# AnalyzeTrain

Repository for analyzing the train task.

[![Data](https://img.shields.io/badge/data-ConvertTrain-lightgrey)](https://github.com/HSUPipeline/ConvertTrain)
[![Template](https://img.shields.io/badge/template-HSUPipeline/AnalyzeTEMPLATE-yellow.svg)](https://github.com/HSUPipeline/AnalyzeTEMPLATE)

## Overview

This repository analyses data from the train task, a spatial navigation and memory task from the Jacobs lab, recorded with single-unit activity from human epilepsy patients.

## Requirements

This repository requires Python >= 3.7.

As well as typical scientific Python packages, dependencies include:
- [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)
- [convnwb](https://github.com/HSUPipeline/convnwb)
- [spiketools](https://github.com/spiketools/spiketools)

The full list of dependencies is listed in `requirements.txt`.

## Repository Layout

Add any details about repository layout here.

This repository is set up in the following way:
- `code/` contains custom code and utilities
- `notebooks/` contains notebooks for exploring analyses
- `scripts/` contains stand alone scripts

## Data

The datasets analyzed in this project are from human subjects with implanted microwires.

Data notes:
- Datasets for this project are organized into the [NWB](https://www.nwb.org/) format.
- Basic preprocessing and data conversion is done in the [ConvertTrain](https://github.com/HSUpipeline/ConvertTrain) repository.
- Spike sorting, to isolate putative single-neurons, has been performed on this data.
