# Creating online managed endpoints in Azure ML

For a detailed explanation of the code, check out the accompanying blog post: [Creating managed online endpoints in Azure ML](https://bea.stollnitz.com/blog/managed-endpoint/).

I include instructions to setup and run the code at the end of this page. However, I've included all the folders created when running the code in this project, so you can go straight to working with the endpoints!

You can find below the steps needed to create and invoke the endpoints in this project.


## Create the models

Execute the following commands:

```
az ml model create -f managed-endpoint/cloud/model-pytorch-fashion.yml
az ml model create -f managed-endpoint/cloud/model-tf-fashion.yml
```


## Create and invoke endpoints

Execute the following commands, replacing `<ENDPOINTX>` with names you choose for your endpoints.


### Endpoint 1

```
az ml online-endpoint create -f managed-endpoint/cloud/endpoint-1/endpoint.yml --name <ENDPOINT1>
az ml online-deployment create -f managed-endpoint/cloud/endpoint-1/deployment.yml --endpoint-name <ENDPOINT1> --all-traffic
az ml online-endpoint invoke -n <ENDPOINT1> --request-file managed-endpoint/sample-request/sample_request.json
```


### Endpoint 2

```
az ml online-endpoint create -f managed-endpoint/cloud/endpoint-2/endpoint.yml --name <ENDPOINT2>
az ml online-deployment create -f managed-endpoint/cloud/endpoint-2/deployment.yml --endpoint-name <ENDPOINT2> --all-traffic
az ml online-endpoint invoke -n <ENDPOINT2> --request-file managed-endpoint/sample-request/sample_request.json
```


### Endpoint 3

```
az ml online-endpoint create -f managed-endpoint/cloud/endpoint-3/endpoint.yml --name <ENDPOINT3>
az ml online-deployment create -f managed-endpoint/cloud/endpoint-3/deployment.yml --endpoint-name <ENDPOINT3> --all-traffic
az ml online-endpoint invoke -n <ENDPOINT3> --request-file managed-endpoint/sample-request/sample_request.json
```


### Endpoint 4

```
az ml online-endpoint create -f managed-endpoint/cloud/endpoint-4/endpoint.yml --name <ENDPOINT4>
az ml online-deployment create -f managed-endpoint/cloud/endpoint-4/deployment.yml --endpoint-name <ENDPOINT4> --all-traffic
az ml online-endpoint invoke -n <ENDPOINT4> --request-file managed-endpoint/sample-request/sample_request.json
```


### Endpoint 5

```
az ml online-endpoint create -f managed-endpoint/cloud/endpoint-5/endpoint.yml --name <ENDPOINT5>
az ml online-deployment create -f managed-endpoint/cloud/endpoint-5/deployment.yml --endpoint-name <ENDPOINT5> --all-traffic
az ml online-endpoint invoke -n <ENDPOINT5> --request-file managed-endpoint/sample-request/sample_request.json
```


### Endpoint 6

```
az ml online-endpoint create -f managed-endpoint/cloud/endpoint-6/endpoint.yml --name <ENDPOINT6>
az ml online-deployment create -f managed-endpoint/cloud/endpoint-6/deployment-blue.yml --endpoint-name <ENDPOINT6> --all-traffic
az ml online-deployment create -f managed-endpoint/cloud/endpoint-6/deployment-green.yml --endpoint-name <ENDPOINT6>
az ml online-endpoint update --name <ENDPOINT6> --traffic "blue=90 green=10"
az ml online-endpoint invoke -n <ENDPOINT6> --request-file managed-endpoint/sample-request/sample_request.json
```


## Setup

Setting up and running the project re-creates the folders `pytorch-model`, `tf-model`, and `sample-request`. Since I already include these in the project, you can go straight to working with endpoints without running the code. I include setup and running instructions here, just in case you'd like to re-create these folders.

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace". Alternatively, you can set up your local machine using the following steps.

Install conda environment:

```
conda env create -f conda.yml
```

Activate conda environment:

```
conda activate fashion-mnist
```


## Run

Within VS Code, open the following files and press F5:
* `managed-endpoint/pytorch-src/train.py` &mdash; This re-creates the folder `pytorch-model`.
* `managed-endpoint/pytorch-src/create-sampple-request.py` &mdash; This re-creates the folder `sample-request`.
* `managed-endpoint/tf-src/train.py` &mdash; This re-creates the folder `tf-model`.
