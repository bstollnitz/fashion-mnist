ENDPOINT_NAME=endpoint-managed-fashion-5

SCORING_URI=$(az ml online-endpoint show --name $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "SCORING_URI: $SCORING_URI"

ACCESS_TOKEN=$(az ml online-endpoint get-credentials --name $ENDPOINT_NAME --query accessToken -o tsv)
echo "PRIMARY_KEY: $ACCESS_TOKEN"

OUTPUT=$(curl --location \
     --request POST $SCORING_URI \
     --header "Authorization: Bearer $ACCESS_TOKEN" \
     --header "Content-Type: application/json" \
     --data @fashion-mnist/managed-endpoint/sample-request/sample_request.json)
echo "OUTPUT: $OUTPUT"
