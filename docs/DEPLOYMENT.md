# Deployment Guide

**Live API:** https://java-sentiment-api.onrender.com

## Docker

```bash
docker build -t sentiment-api .
docker run -p 8080:8080 sentiment-api
```

With custom configuration:
```bash
docker run -d \
  -p 8080:8080 \
  -e SPRING_PROFILES_ACTIVE=production \
  -e MAX_HEAP=1g \
  sentiment-api
```

## Local Development

```bash
mvn clean package -DskipTests
mvn spring-boot:run
```

Requires production model in `models/production/` (run `./scripts/promote_to_production.sh` first).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SPRING_PROFILES_ACTIVE` | (none) | Set to `production` for stricter limits |
| `SENTIMENT_SVM_MODEL` | `/app/models/production/sentiment_model.ser` | Model path |
| `MAX_HEAP` | `512m` | JVM max heap |
| `SERVER_PORT` | `8080` | API port |

**Rate Limits:**
- Default: 100 single/min, 20 batch/min
- Production profile: 60 single/min, 10 batch/min

## Health Check

```bash
curl http://localhost:8080/api/v1/health
```

Actuator endpoints: `/actuator/health`, `/actuator/prometheus`

## Performance

| Metric | Value |
|--------|-------|
| Startup time | < 10 seconds |
| Memory | 512MB - 1GB |
| Latency | <10ms mean, <3ms p99 |
| Batch (100 texts) | 50-100ms |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `./scripts/promote_to_production.sh` |
| Out of memory | Increase `-e MAX_HEAP=1g` |
| Slow responses | Increase CPU: `--cpus 2` |

## Production Checklist

- [ ] Model exists: `models/production/sentiment_model.ser`
- [ ] Health endpoint responds: `/api/v1/health`
- [ ] Memory limits configured
