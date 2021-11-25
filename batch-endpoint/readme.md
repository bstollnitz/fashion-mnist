# Creating batch endpoints in Azure ML

You can find below the steps needed to create and invoke the endpoints in this project.
For a detailed explanation of the code, check out the accompanying blog post: https://bea.stollnitz.com/blog/batch-endpoint/.

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
