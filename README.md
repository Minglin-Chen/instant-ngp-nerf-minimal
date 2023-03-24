# Instant NGP (NeRF)

This repository simplifies the code of NeRF from [Instant Neural Graphics Primitives]([GitHub - NVlabs/instant-ngp: Instant neural graphics primitives: lightning fast NeRF and more](https://github.com/NVlabs/instant-ngp)), and provides a more clean and readable code for academic purposes. The performance is higher than that reported in the original paper. 

## Requirements

* An **NVIDIA GPU**

* A **C++14** capable compiler. The following choices are recommended:
  
  * **Windows**: Visual Studio 2019 or 2022
  
  * **Linux**: GCC/G++ 8 or higher

* A recent version of **CUDA**. The following choices are recommended:
  
  * Windows: CUDA 11.5 or higher
  
  * Linux: CUDA 10.2 or higher

* **CMake** v3.19 or higher

* **Python** 3.7 or higher

* **PyTorch** 1.11.0

## Compilation

```shell
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

## Acknowledgement

The code is highly from [Instant-NGP](https://github.com/NVlabs/instant-ngp), if it is helpful for you, please cite their paper:

```latex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```
