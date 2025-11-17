# 🚀 Production Deployment Guide

Comprehensive guide for deploying the Lo-Fi Music Generator to production environments.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Serverless Deployment](#serverless-deployment)
- [Cloud Platform Guides](#cloud-platform-guides)
- [Monitoring & Observability](#monitoring--observability)
- [Performance Tuning](#performance-tuning)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum (CPU-only):**
- CPU: 4 cores, 3.0 GHz+
- RAM: 16 GB
- Storage: 50 GB SSD
- Network: 100 Mbps

**Recommended (GPU):**
- CPU: 8+ cores, 3.5 GHz+
- GPU: NVIDIA RTX 3090 or better (24GB+ VRAM)
- RAM: 32 GB+
- Storage: 100 GB NVMe SSD
- Network: 1 Gbps

### Software Requirements

- Docker 20.10+
- docker-compose 1.29+
- NVIDIA Container Toolkit (for GPU)
- Python 3.8-3.11
- CUDA 11.8+ (for GPU)

---

## Deployment Options

### 1. **Docker (Recommended)**
Best for: Single-server deployments, development, testing

**Pros:**
- Easy setup and reproducibility
- Isolated environment
- Resource management
- Multi-stage builds for optimization

**Cons:**
- Single-point of failure
- Limited horizontal scaling

### 2. **Kubernetes**
Best for: Multi-server deployments, high availability, auto-scaling

**Pros:**
- Horizontal scaling
- Load balancing
- Self-healing
- Rolling updates

**Cons:**
- Complex setup
- Higher operational overhead

### 3. **Serverless**
Best for: Variable workloads, pay-per-use pricing

**Pros:**
- Auto-scaling
- Pay-per-use
- No server management
- Global distribution

**Cons:**
- Cold starts
- Limited GPU support
- Vendor lock-in

---

## Docker Deployment

### Production Docker Compose

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/models/lofi-gpt2
      - NUM_WORKERS=4
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models:ro
      - ./output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

### Running Production Stack

```bash
# Build production image
docker-compose -f docker-compose.prod.yml build

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f api

# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Kubernetes Deployment

### K8s Manifests

**1. Namespace**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lofi-generator
```

**2. ConfigMap**

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lofi-config
  namespace: lofi-generator
data:
  config.yaml: |
    model:
      embedding_dim: 768
      num_layers: 12
      num_heads: 12
      context_length: 2048
    # ... rest of config
```

**3. Secret**

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: lofi-secrets
  namespace: lofi-generator
type: Opaque
data:
  api-key: <base64-encoded-key>
```

**4. Deployment**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lofi-api
  namespace: lofi-generator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lofi-api
  template:
    metadata:
      labels:
        app: lofi-api
    spec:
      containers:
      - name: api
        image: lofi-generator:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/lofi-gpt2"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: config
        configMap:
          name: lofi-config
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

**5. Service**

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lofi-api-service
  namespace: lofi-generator
spec:
  selector:
    app: lofi-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**6. HorizontalPodAutoscaler**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lofi-api-hpa
  namespace: lofi-generator
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lofi-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n lofi-generator

# View logs
kubectl logs -f deployment/lofi-api -n lofi-generator

# Get service endpoint
kubectl get svc lofi-api-service -n lofi-generator

# Scale deployment
kubectl scale deployment lofi-api --replicas=5 -n lofi-generator

# Rolling update
kubectl set image deployment/lofi-api api=lofi-generator:v2.0.1 -n lofi-generator
```

---

## Serverless Deployment

### AWS Lambda + API Gateway

**1. Create Lambda Layer**

```bash
# Create layer with dependencies
mkdir python
pip install -r requirements.txt -t python/
zip -r layer.zip python/

# Upload to AWS Lambda
aws lambda publish-layer-version \
  --layer-name lofi-dependencies \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.9
```

**2. Lambda Handler**

```python
# lambda_handler.py
import json
import os
from src.api import LoFiAPI

# Initialize API (cold start)
api = None

def handler(event, context):
    """AWS Lambda handler."""
    global api

    # Initialize on cold start
    if api is None:
        config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
        model_path = os.environ.get('MODEL_PATH')

        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        api = LoFiAPI(config, model_path)

    # Parse request
    body = json.loads(event.get('body', '{}'))

    # Generate
    response = api.generate(body)

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps(response),
    }
```

**3. Deploy with SAM**

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  LoFiFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: lofi-generator
      Runtime: python3.9
      Handler: lambda_handler.handler
      MemorySize: 10240  # 10 GB
      Timeout: 900  # 15 minutes
      Layers:
        - !Ref LoFiDependenciesLayer
      Environment:
        Variables:
          MODEL_PATH: /opt/models/lofi-gpt2
          CONFIG_PATH: /opt/config.yaml
      Events:
        GenerateAPI:
          Type: Api
          Properties:
            Path: /generate
            Method: POST

  LoFiDependenciesLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: lofi-dependencies
      ContentUri: ./layer.zip
      CompatibleRuntimes:
        - python3.9
```

```bash
# Deploy
sam build
sam deploy --guided
```

---

## Cloud Platform Guides

### AWS Deployment

**Using ECS (Elastic Container Service)**

```bash
# 1. Build and push Docker image
aws ecr create-repository --repository-name lofi-generator
docker build -t lofi-generator:latest .
docker tag lofi-generator:latest <account-id>.dkr.ecr.<region>.amazonaws.com/lofi-generator:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/lofi-generator:latest

# 2. Create ECS task definition (task-def.json)
{
  "family": "lofi-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "lofi-api",
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/lofi-generator:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MODEL_PATH", "value": "/models/lofi-gpt2"},
        {"name": "CUDA_VISIBLE_DEVICES", "value": "0"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lofi-api",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}

# 3. Register task definition
aws ecs register-task-definition --cli-input-json file://task-def.json

# 4. Create ECS service
aws ecs create-service \
  --cluster lofi-cluster \
  --service-name lofi-api \
  --task-definition lofi-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Google Cloud Platform (GCP)

**Using Cloud Run**

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/lofi-generator

# 2. Deploy to Cloud Run
gcloud run deploy lofi-api \
  --image gcr.io/<project-id>/lofi-generator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 900 \
  --max-instances 10 \
  --set-env-vars MODEL_PATH=/models/lofi-gpt2

# 3. Get service URL
gcloud run services describe lofi-api --region us-central1 --format 'value(status.url)'
```

### Azure

**Using Azure Container Instances**

```bash
# 1. Push to Azure Container Registry
az acr create --resource-group lofi-rg --name lofiregistry --sku Basic
az acr login --name lofiregistry
docker tag lofi-generator:latest lofiregistry.azurecr.io/lofi-generator:latest
docker push lofiregistry.azurecr.io/lofi-generator:latest

# 2. Create container instance
az container create \
  --resource-group lofi-rg \
  --name lofi-api \
  --image lofiregistry.azurecr.io/lofi-generator:latest \
  --cpu 4 \
  --memory 16 \
  --ports 8000 \
  --environment-variables MODEL_PATH=/models/lofi-gpt2 \
  --registry-login-server lofiregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password>
```

---

## Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'lofi-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import the following dashboard JSON for comprehensive monitoring:

**Key Metrics:**
- Generation requests per second
- Average generation duration
- Quality score distribution
- Active generations
- Queue size
- GPU utilization
- Memory usage
- Error rate

### Application Logging

Configure structured logging:

```python
# config.yaml
logging:
  level: INFO
  format: json
  handlers:
    - type: file
      path: /var/log/lofi-api.log
      rotation: daily
      retention: 30
    - type: stdout
      format: json
    - type: syslog
      host: syslog-server
      port: 514
```

---

## Performance Tuning

### Model Optimization

```python
# Apply quantization for 2-4x speedup
from src.optimization import optimize_model_for_production

model = optimize_model_for_production(
    model,
    quantization='fp16',  # or 'int8'
    pruning_amount=0.2,   # 20% pruning
    device='cuda',
)
```

### Batch Processing

```python
# Process multiple requests in batches
from src.optimization import BatchInferenceOptimizer

batches = BatchInferenceOptimizer.dynamic_batching(
    requests,
    max_batch_size=32,
    max_wait_time=0.1,
)
```

### Caching

```python
# Enable generation caching
from src.optimization import GenerationCache

cache = GenerationCache(max_cache_size=1000)
# Integrate with API
```

---

## Security Best Practices

### 1. **API Authentication**

```python
# Add API key authentication
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
```

### 2. **Rate Limiting**

```python
# Add rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/generate")
@limiter.limit("10/minute")
async def generate(...):
    ...
```

### 3. **HTTPS/TLS**

```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name api.lofi-generator.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://api:8000;
    }
}
```

### 4. **Input Validation**

All inputs are validated using Pydantic models in the API.

---

## Troubleshooting

### Common Issues

**1. Out of Memory**

```bash
# Reduce batch size
export BATCH_SIZE=1

# Use model quantization
export QUANTIZATION=int8

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**2. Slow Generation**

```bash
# Enable FP16
export USE_FP16=true

# Reduce max_length
export MAX_LENGTH=512

# Use KV-cache
export USE_KV_CACHE=true
```

**3. API Timeouts**

```yaml
# Increase timeouts in docker-compose
environment:
  - API_TIMEOUT=900
```

**4. GPU Not Detected**

```bash
# Check NVIDIA driver
nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Maintenance

### Backup Strategy

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to S3
aws s3 cp models-backup-*.tar.gz s3://lofi-backups/

# Backup database (if using)
pg_dump lofi_db > lofi_db_backup.sql
```

### Updates

```bash
# Rolling update in Kubernetes
kubectl set image deployment/lofi-api api=lofi-generator:v2.1.0 -n lofi-generator

# Blue-green deployment in Docker
docker-compose -f docker-compose.blue.yml up -d
# Test
docker-compose -f docker-compose.green.yml down
```

---

For more help, see:
- [README.md](README.md) - Main documentation
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Usage examples
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
