ENDPOINT_NAME=endpoint-managed-fashion-1

SCORING_URI=$(az ml online-endpoint show --name $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "SCORING_URI: $SCORING_URI"

PRIMARY_KEY=$(az ml online-endpoint get-credentials --name $ENDPOINT_NAME --query primaryKey -o tsv)
echo "PRIMARY_KEY: $PRIMARY_KEY"

OUTPUT=$(curl --location \
     --request POST $SCORING_URI \
     --header "Authorization: Bearer $PRIMARY_KEY" \
     --header "Content-Type: application/json" \
     --data @fashion-mnist/managed-endpoint/sample-request/sample_request.json)
echo "OUTPUT: $OUTPUT"