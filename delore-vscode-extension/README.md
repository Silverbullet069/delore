

<h1 align="center">
    DeLoRe - Detecting, Locating and Repairing C/C++ Software Vulnerabilities
</h1>

This is a **[Visual Studio Code](https://github.com/Microsoft/vscode)** extension, which can automatically:
1. Detect vulnerabilities at function-level.
2. Locate vulnerabilities specifically at line-level.
3. Repair vulnerabilities by providing suggestions to the user.

## Table of contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Demo](#demo)
4. [Requirements](#requirements)
5. [Development](#development)
6. [What's next?](#whats-next)
7. [Contributors](#contributors)
8. [References](#references)

## Introduction
In the modern digital age, the security of software applications is of paramount importance. Vulnerabilities in software can lead to serious consequences, including data breaches, system downtime, loss of user trust, ... Despite the best efforts of developers, vulnerabilities can still creep into software due to a variety of reasons such as coding errors, lack of understanding of security principles, or the complexity of modern software systems.

This is where a tool like DeLoRe comes into play. DeLoRe is a [Visual Studio Code](https://github.com/Microsoft/vscode) extension that is designed as an **AI wrapper** to automatically **detect, locate, and repair** vulnerabilities in **C/C++ software**. It operates at the function level, pinpointing vulnerabilities at the line level, and provides suggestions for repairing these vulnerabilities.

By integrating DeLoRe into the development process, developers can catch and fix vulnerabilities early, before they become a problem. This not only improves the security of the software but also saves time and resources that would otherwise be spent on dealing with the consequences of a security breach.

[Back to ToC](#table-of-contents)

## Installation

...insert marketplace URL here

[Back to ToC](#table-of-contents)

## Demo

<u>_Commands_</u>: **`Shift`** + **`Alt`** + **`D`**

![demo](./asset/delore.gif)

[Back to ToC](#table-of-contents)

## Requirements
- [**NodeJS**](https://nodejs.org/en/download/) (>= *v20*)
- [**Python**](https://www.python.org/downloads/) (tested in *v3.10.12*)
- [**Visual Studio Code**](https://code.visualstudio.com/download) (tested in *v1.82.0*)
- [**unzip**](https://linuxize.com/post/how-to-unzip-files-in-linux) (Linux only)

[Back to ToC](#table-of-contents)

## Development
1. Clone the repository and change directory
```sh
$ git clone --depth 1 git@github.com:Silverbullet069/delore.git && cd ./delore/delore-vscode-extension
```

2. Install NodeJS dependencies
```sh
$ npm install
```

3. Setup Python development environment
```sh
$ python -m venv ./python/virtual_envs/py-delore
$ source ./python/virtual_envs/py-delore/bin/activate[.fish] # if your terminal is Fish shell
$ pip install -r requirements.txt
```

4. Download AI Models from Google Drive

> <u>_NOTE:_</u>: make sure you save enough disk space for this step

You can remove `&& rm <model>.zip` to retain your archive files.

```sh
# Devign
$ gdown https://drive.google.com/file/d/1uhT71kvoJ87Eb4hCjPdP-Ekww8SaJ35W/view?usp=sharing && unzip devign.zip -d ./python/ai_models && rm devign.zip

# LineVul
$ gdown https://drive.google.com/file/d/1-A8WUw-4WnaeLRNsv3sUnfUXeJLmm1RG/view?usp=sharing && unzip linevul.zip -d ./python/ai_models && rm linevul.zip

# LineVD
$ gdown https://drive.google.com/file/d/1HQbCRMSixoKa_Y-nJK_bAvY9MSK2E72O/view?usp=sharing && unzip linevd.zip -d ./python/ai_models && rm linevd.zip
```

[Back to ToC](#table-of-contents)


## What's next?
DeLoRe has covered all basic functionalities and some nitty-gritty UI/UX features. However, there are some advanced features that contributors can consider:

- Let User toggle between **Local** or **Remote** mode. For now, DeLoRe is in **Local** mode, which means all resources are downloaded and stored in user machine. To develop **Remote** mode, contributors can:
    + Create a simple Python Back-end using Flask, FastAPI, Django, ... etc.
    + Read about the Standardized Input and Output that works with **EVERY** Python model and design REST APIs.
    
- Let User add their custom models. DeLoRe is designed so it can **merge the results of multiple models into one.** The more models it has, the better the result is. To develop this feature, contributors can:
    + Redefined the input and output of some unused AI models in the future to achieve compatibility with Standardized Input and Output.


[Back to ToC](#table-of-contents)

## Contributors

Special thanks to my third-year junior [Uyen Pham](https://github.com/21020419PhamTuUyen) for training, evaluating, refining and redefining Devign and LineVD model.

[Back to ToC](#table-of-contents)

## References 
- **Devign**: https://github.com/saikat107/Devign
- **LineVul**: https://github.com/awsm-research/LineVul
- **LineVD**: https://github.com/davidhin/linevd

[Back to ToC](#table-of-contents)