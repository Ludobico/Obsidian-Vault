[[Detectron2]] 를 사용하기에 앞서 공통적인 인자들을 생성하는 함수입니다.

```py
def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
```

## <font color="#ffc000">--config-file</font>
`config-file` 의 위치를 지정합니다.

## <font color="#ffc000">--resume</font>
사용자가 이전에 저장된 체크포인트로부터 학습을 이어나가고자 할 때 사용됩니다. 만약 사용자가 `--resume` 옵션을 사용하여 학습을 재개하려면 이전 체크포인트의 경로를 지정해주어야 합니다.

## <font color="#ffc000">--eval-only</font>
학습된 모델을 기반으로 평가를 진행할 때 사용됩니다.

## <font color="#ffc000">--num-gpus</font>
* <font color="#2DC26B">type= int</font>
* <font color="#2DC26B">default = 1</font>
학습, 평가를 진행할 기기의 gpu 갯수를 설정합니다.

## <font color="#ffc000">--num-machines</font>
* <font color="#2DC26B">type = int</font>
* <font color="#2DC26B">default = 1</font>
학습,평가를 진행할 기기의 총 갯수를 설정합니다.

## <font color="#ffc000">--machine-rank</font>
* <font color="#2DC26B">type = int</font>
* <font color="#2DC26B">default = 0</font>
분산 학습시 여러 gpu 또는 기기가 사용되는 경우, 각 gpu 또는 기기에 부여되는 고유한 식별 번호값입니다.

Machine rank는 0부터 시작하여 각 gpu 또는 기기에 대해 순차적으로 할당됩니다. 주로 분산 학습 시에 사용되며, 예를 들어 4개의 gpu를 사용하여 학습할 때는 각 gpu에 0부터 3까지의 숫자가 차례로 부여됩니다.