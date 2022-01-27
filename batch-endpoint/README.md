# Creating batch endpoints in Azure ML

For a detailed explanation of the code, check out the accompanying blog post: [Creating batch endpoints in Azure ML](https://bea.stollnitz.com/blog/batch-endpoint/).

I include instructions to setup and run the code at the end of this page. However, I've included all the folders created when running the code in this project, so you can go straight to working with the endpoints!

You can find below the steps needed to create and invoke the endpoints in this project.


## Create the models

Execute the following commands:

```
az ml model create -f batch-endpoint/cloud/model-pytorch-batch-fashion.yml
az ml model create -f batch-endpoint/cloud/model-tf-batch-fashion.yml
```


## Create the cluster

```
az ml compute create -f batch-endpoint/cloud/cluster-cpu.yml
```


## Create the endpoints

Execute the following commands, replacing `<ENDPOINT1>` and `<ENDPOINT2>` with names you choose for your endpoints.

```
az ml batch-endpoint create -f batch-endpoint/cloud/endpoint-1/endpoint.yml --name <ENDPOINT1>
az ml batch-deployment create -f batch-endpoint/cloud/endpoint-1/deployment.yml --endpoint-name <ENDPOINT1> --set-default
az ml batch-endpoint create -f batch-endpoint/cloud/endpoint-2/endpoint.yml --name <ENDPOINT2>
az ml batch-deployment create -f batch-endpoint/cloud/endpoint-2/deployment.yml --endpoint-name <ENDPOINT2> --set-default
```


## Invoke the endpoints

Execute the following commands, replacing `<ENDPOINT1>` and `<ENDPOINT2>` with the names you chose for your endpoints.

```
az ml batch-endpoint invoke --name <ENDPOINT1> --input-local-path batch-endpoint/sample-request
az ml batch-endpoint invoke --name <ENDPOINT2> --input-local-path batch-endpoint/sample-request
```


## Get the prediction results

Go to the [Azure ML portal](https://ml.azure.com), click on "Endpoints", "Batch endpoints", and click on the name of the endpoint. Then click on "Runs", and on the top "Display name", which is the latest run. Once the run has completed, click on "batchscoring", "Outputs + logs", and "Show data outputs". Clicking on the "Access data" icon will take you to the blob storage location where the prediction results are located.


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
* `batch-endpoint/pytorch-src/train.py` &mdash; This re-creates the folder `pytorch-model`.
* `batch-endpoint/pytorch-src/create-sampple-request.py` &mdash; This re-creates the folder `sample-request`.
* `batch-endpoint/tf-src/train.py` &mdash; This re-creates the folder `tf-model`.
