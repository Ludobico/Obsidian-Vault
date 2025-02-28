
## Reload

### docker-compose

```
docker-compose exec {service_name} nginx -s reload 
```

### docker

```
docker exec {container_name} nginx -s reload
```

## Test

### docker-compose

```
docker-compose exec {service_name} nginx -t
```

### docker

```
docker exec {container_name} nginx -t
```

## Stop

### docker-compose

```
docker-compose exec {service_name} nginx -s stop
```

### docker

```
docker exec {container_name} nginx -s stop
```

## Service restart

### docker-compose

```
docker-compose restart {service_name}
```

## Check logs

### docker-compose

```
docker-compose logs {service_name}
```

