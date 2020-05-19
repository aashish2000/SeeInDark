# See In The Dark
This project focuses on Low Light Photo Enhancement Using Simplified Fully Convolutional Neural Networks. This was attempted to produce a memory efficient model that can effectively run on edge devices without sacrificing image quality. 

## Table of contents
* [General Info](#general-info)
* [Packages Used](#packages-used)
* [Setup](#setup)
* [Usage](#usage)
* [Features](#features)
* [Screenshots](#screenshots)
* [References](#references)

## General Info
- This project is aimed at creating a solution to tackling the problem of Low Light Photography Using Simplified Fully Convolutional Neural Networks. 
- The Neural Network architecture is written in PyTorch and the application is deployed using Flask.

## Packages Used
- PyTorch
- Flask
- requests
- numpy

## Setup

### Requirements
- Python 3.3+

### Installation
- To install required packages:<br>
```bash
pip install -r requirements.txt
```

## Usage

### Linux
```bash
export FLASK_APP=app.py
flask run
```

### Windows
```bash
set FLASK_APP=app.py
flask run
```

## Screenshots

- Image Upload Page:
![](./assets/home.png)

- Result Page:
![](./assets/result.png)

## References
- <https://github.com/ninetf135246/pytorch-Learning-to-See-in-the-Dark>
- <https://github.com/cchen156/Learning-to-See-in-the-Dark>



