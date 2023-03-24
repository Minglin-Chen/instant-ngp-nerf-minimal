@echo off

set CUDA_VISIBLE_DEVICES=0
set NGPPATH=..\..\..\build
set PYTHONPATH=%PYTHONPATH%;%NGPPATH%;..\python

set DATA_ROOT=E:\data\NeRF_Resource\data\nerf_synthetic
set RESULT_ROOT=..\checkpoint

for %%s in (chair, drums, ficus, hotdog, lego, materials, mic, ship) do (
    echo "%%s eval >>>"
    python -m run ^
        --load_snapshot %RESULT_ROOT%\%%s\%%s_ckpt.msgpack ^
        --nerf_compatibility ^
        --test_transforms %DATA_ROOT%\%%s\transforms_test.json
)