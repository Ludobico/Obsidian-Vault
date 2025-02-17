
## Information

```
ollama
```

```
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```

## Start Server

```
ollama serve
```

## Download model

```
ollama pull [model-name]
```
## Download and Run model

```
ollama run [model-name]
```

## Download model weight

```
ollama pull [model-name]
```

## List Models

```
ollama list
```

```
NAME              ID              SIZE      MODIFIED
deepseek-r1:7b    0a8c26691023    4.7 GB    10 days ago
```

## Remove Model

```
ollama rm [model-name]
```

