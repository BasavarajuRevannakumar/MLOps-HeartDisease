# Kubernetes Deployment

## Apply Deployment

kubectl apply -f deployment.yaml

## Apply Service

kubectl apply -f service.yaml

## Check Pods

kubectl get pods

## Check Services

kubectl get svc

## View Logs

kubectl logs <pod-name>

## Delete Resources

kubectl delete -f service.yaml
kubectl delete -f deployment.yaml