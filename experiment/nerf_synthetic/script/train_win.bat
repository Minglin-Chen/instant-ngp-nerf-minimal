@echo off

set CUDA_VISIBLE_DEVICES=0
set NGPPATH=..\..\..\build
set PYTHONPATH=%PYTHONPATH%;%NGPPATH%;..\python

set DATA_ROOT=E:\data\NeRF_Resource\data\nerf_synthetic
set RESULT_ROOT=..\checkpoint
set N_STEPS=50000

for %%s in (chair, drums, ficus, hotdog, lego, materials, mic, ship) do (
    echo "%%s train >>>"
    mkdir %RESULT_ROOT%\%%s
    python -m run ^
        --scene %DATA_ROOT%\%%s\transforms_train.json ^
        --network ..\config\base.json ^
        --save_snapshot %RESULT_ROOT%\%%s\%%s_ckpt.msgpack ^
        --nerf_compatibility ^
        --test_transforms %DATA_ROOT%\%%s\transforms_test.json ^
        --n_steps %N_STEPS%
)

rem compute-sanitizer is a helpful tool to debug CUDA program