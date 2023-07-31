# TORCH

[英文原版教程地址](https://pytorch.org/docs/stable/torch.html)

torch 库包含了用于多维张量的数据结构，并定义了在这些张量上进行的数学操作。此外，它还提供了许多实用程序，以用于有效地序列化张量和任意类型，以及其他有用的实用程序。

它可以调用 CUDA 加速，使您能够在 *计算能力* $\geq 3.0$ 的 NVIDIA GPU (英伟达显卡)上运行张量计算。

## Tensors （张量）

|                           |                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `is_tensor`               | 如果对象是一个 PyTorch tensor，返回 `True`。                                                                                          |
| `is_storage`              | 如果对象是一个 PyTorch storage，返回 `True` 。                                                                                        |
| `is_complex`              | 如果`input`的数据类型是一个复数数据类型，即`torch.complex64`和 `torch.complex128`的其中一个，返回 `True` 。                           |
| `is_conj`                 | 如果`input`是共轭张量，则返回`True`，即其共轭位被设置为`True`。                                                                       |
| `is_floating_point`       | 如果`input`的数据类型是浮点类型，即`torch.float64`、 `torch.float32`、`torch.float16`以及`torch.bfloat16`中的其中一个，返回 `True` 。 |
| `is_nonzero`              | 如果`input`是单元素张量，且在类型转换后不等于零，则返回`True`。                                                                       |
| `set_default_dtype`       | 设置默认的浮点dtype为`d`。                                                                                                            |
| `get_default_dtype`       | 获取当前的默认浮点`torch.dtype`                                                                                                       |
| `get_default_device`      | 设置在`device`上默认分配的`torch.Tensor`                                                                                              |
| `set_default_tensor_type` | 设置默认`torch.Tensor`为浮点张量类型`t`。                                                                                             |
| `numel`                   | 返回`input`张量中的元素个数。                                                                                                         |
| `set_printoptions`        | 设置用于打印的选项                                                                                                                    |
| `set_flush_denormal`      | 在CPU上禁用非规格浮点数。                                                                                                             |

## 新建操作

* <font color=＃00BFFF>NOTE</font>
  
随机抽样的新建操作在 <font color=orange> Random sampling </font>列出，包括：<font color=＃00BFFF>`torch.rand()` `torch.rand_like()` `torch.randn()` `torch.randn_like` `torch.randint()` `torch.randint_like()` `torch.randperm()`</font>。你也可以使用<font color=＃00BFFF>`torch.empty()`</font>和<font color=orange>原址随机抽样方法</font>从更大范围的分布中采样值来创建`torch.Tensor`。

|                      |                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `tensor`             | 通过复制数据构造一个没有自动求导历史的张量(也称为“叶张量”，参见 <font color=orange>自动求导机制</font>)。 |
| `sparse_coo_tensor`  | 在给定的`indices`(索引)处以坐标格式构造具有指定值的稀疏张量。                                             |
| `sparse_csr_tensor`  | 在给定的`crow_indecies` 和 `col_indecies`下构建一个CSR(Compressed Sparse Row)格式的稀疏张量               |
| `sparse_csc_tensor`  | 在给定的`ccol_indecies` 和 `row_indecies`下构建一个CSC(Compressed Sparse Column)格式的稀疏张量            |
| `sparse_bsr_tensor`  | 在给定的`crow_indecies` 和 `col_indecies`下构建一个BSR(Block Compressed Sparse Row)格式的稀疏张量         |
| `sparse_bsc_tensor`  | 在给定的`ccol_indecies` 和 `row_indecies`下构建一个BSC(Block Compressed Sparse Column)格式的稀疏张量      |
| `asarray`            | 将对象转换为一个张量|
| `as_tensor`          | 将数据转换为一个张量，共享数据并且保留自动求导历史（如果存在）                                            |
| `as_strided` |使用特定的`size` `stride`和`storage_offset`构建一个现存的*torch.Tensor* `input`的视图|
| `from_numpy` |从一个`numpy.ndarray`数据创造一个`Tensor`|
| `from_dlpack` |将外部库的数据转换为一个`torch.Tensor`|
| `frombuffer` |从一个实现Python缓冲区协议的对象创建一个一维`Tensor`。|
| `zeros` |返回一个用标量值0填充的张量，其形状由变量参数`size`定义。|
| `zeros_like` |返回一个用标量值0填充的张量，其形状与`input`相同。|
| `ones` |返回一个用标量值1填充的张量，其形状由变量参数`size`定义。|
| `ones_like` |返回一个用标量值0填充的张量，其形状与`input`相同。|
| `arange` |返回一个大小为$\lceil \frac{end - start}{step} \rceil$ 的1维向量，取值区间为`[start, end)`，公差为`step`，起始值为`start`|
| `range` |回一个大小为$\lceil \frac{end - start}{step} \rceil +1$ 的1维向量，步长为$step$，从$start$至$end$的1维向量|
| `linspace` |创建大小为`steps`的1维张量，其值从`start`到`end`均匀间隔(包括在内)。|
| `logspace` |创建大小为`steps`的1维张量，其值从$base^{start}$到$base^{end}$均匀间隔(包括在内)，对数的底数为`base`。|
| `eye` |返回一个对角线为1，其余元素为0的2维张量|
| `empty` |返回一个由未初始化数据填充的张量|
| `empty_like` |返回一个与`input`大小相同的未初始化张量|
| `empty_strided` |用特定的`size` 和`stride`创建一个由未定义数据填充的张量|
| `full` |创建一个大小为`size`，填充值为`fill_value`的张量|
| `full_like` |返回一个与`input`大小相同，填充值为`fill_value`的张量|
| `quantize_per_tensor` |将浮点张量转换为具有给定scale和zero point的量化张量。|
| `quantize_per_channel` |将浮点张量转换为具有给定scale和zero point的每通道量化张量。|
| `dequantize` |通过对量化张量进行反量化，返回一个fp32张量|
| `complex` |构造一个实部等于`real`，虚部等于`imag`的复张量。|
| `polar` |构造一个复张量，其元素为与极坐标对应的笛卡尔坐标，具有绝对值`abs`和角度`angle`。|
| `heaviside` |为`input`中的每个元素计算Heaviside阶跃函数。|
