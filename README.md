# myFS
```bash
python setup.py install
# Or (for dev mode)
pip install -e .[dev]
pre-commit install
```
```bash
python federatedscope/main.py --cfg federatedscope/gfl/baseline/config.yaml --client_cfg federatedscope/gfl/baseline/fedavg_gin_minibatch_on_cikmcup_per_client.yaml federate.total_round_num 500
```
（我也记不清运行多少round了）


## TODO
1. fedmaml算法已修改完成，但显存占用过大，速度很慢（40G显存都不行），运行下面代码
```bash
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedmaml.yaml --client_cfg federatedscope/gfl/baseline/fedavg_gin_minibatch_on_cikmcup_per_client.yaml federate.total_round_num 1000
```
