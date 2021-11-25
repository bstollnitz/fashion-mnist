# Creating online managed endpoints in Azure ML

You can find below the steps needed to create and invoke the endpoints in this project.
For a detailed explanation of the code, check out the accompanying blog post: https://bea.stollnitz.com/blog/managed-endpoint/.

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
