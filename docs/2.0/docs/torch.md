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

## Creation Ops （新建操作）

* <font color=＃00BFFF>NOTE</font>
  
随机抽样的新建操作在 <font color=orange> Random sampling </font>列出，包括：<font color=＃00BFFF>`torch.rand()` `torch.rand_like()` `torch.randn()` `torch.randn_like` `torch.randint()` `torch.randint_like()` `torch.randperm()`</font>。你也可以使用<font color=＃00BFFF>`torch.empty()`</font>和<font color=orange>原址随机抽样方法</font>从更大范围的分布中采样值来创建`torch.Tensor`。

|                        |                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `tensor`               | 通过复制数据构造一个没有自动求导历史的张量(也称为“叶张量”，参见 <font color=orange>自动求导机制</font>)。                 |
| `sparse_coo_tensor`    | 在给定的`indices`(索引)处以坐标格式构造具有指定值的稀疏张量。                                                             |
| `sparse_csr_tensor`    | 在给定的`crow_indecies` 和 `col_indecies`下构建一个CSR(Compressed Sparse Row)格式的稀疏张量                               |
| `sparse_csc_tensor`    | 在给定的`ccol_indecies` 和 `row_indecies`下构建一个CSC(Compressed Sparse Column)格式的稀疏张量                            |
| `sparse_bsr_tensor`    | 在给定的`crow_indecies` 和 `col_indecies`下构建一个BSR(Block Compressed Sparse Row)格式的稀疏张量                         |
| `sparse_bsc_tensor`    | 在给定的`ccol_indecies` 和 `row_indecies`下构建一个BSC(Block Compressed Sparse Column)格式的稀疏张量                      |
| `asarray`              | 将对象转换为一个张量                                                                                                      |
| `as_tensor`            | 将数据转换为一个张量，共享数据并且保留自动求导历史（如果存在）                                                            |
| `as_strided`           | 使用特定的`size` `stride`和`storage_offset`构建一个现存的*torch.Tensor* `input`的视图                                     |
| `from_numpy`           | 从一个`numpy.ndarray`数据创造一个`Tensor`                                                                                 |
| `from_dlpack`          | 将外部库的数据转换为一个`torch.Tensor`                                                                                    |
| `frombuffer`           | 从一个实现Python缓冲区协议的对象创建一个一维`Tensor`。                                                                    |
| `zeros`                | 返回一个用标量值0填充的张量，其形状由变量参数`size`定义。                                                                 |
| `zeros_like`           | 返回一个用标量值0填充的张量，其形状与`input`相同。                                                                        |
| `ones`                 | 返回一个用标量值1填充的张量，其形状由变量参数`size`定义。                                                                 |
| `ones_like`            | 返回一个用标量值0填充的张量，其形状与`input`相同。                                                                        |
| `arange`               | 返回一个大小为$\lceil \frac{end - start}{step} \rceil$ 的1维向量，取值区间为`[start, end)`，公差为`step`，起始值为`start` |
| `range`                | 回一个大小为$\lceil \frac{end - start}{step} \rceil +1$ 的1维向量，步长为$step$，从$start$至$end$的1维向量                |
| `linspace`             | 创建大小为`steps`的1维张量，其值从`start`到`end`均匀间隔(包括在内)。                                                      |
| `logspace`             | 创建大小为`steps`的1维张量，其值从$base^{start}$到$base^{end}$均匀间隔(包括在内)，对数的底数为`base`。                    |
| `eye`                  | 返回一个对角线为1，其余元素为0的2维张量                                                                                   |
| `empty`                | 返回一个由未初始化数据填充的张量                                                                                          |
| `empty_like`           | 返回一个与`input`大小相同的未初始化张量                                                                                   |
| `empty_strided`        | 用特定的`size` 和`stride`创建一个由未定义数据填充的张量                                                                   |
| `full`                 | 创建一个大小为`size`，填充值为`fill_value`的张量                                                                          |
| `full_like`            | 返回一个与`input`大小相同，填充值为`fill_value`的张量                                                                     |
| `quantize_per_tensor`  | 将浮点张量转换为具有给定scale和zero point的量化张量。                                                                     |
| `quantize_per_channel` | 将浮点张量转换为具有给定scale和zero point的每通道量化张量。                                                               |
| `dequantize`           | 通过对量化张量进行反量化，返回一个fp32张量                                                                                |
| `complex`              | 构造一个实部等于`real`，虚部等于`imag`的复张量。                                                                          |
| `polar`                | 构造一个复张量，其元素为与极坐标对应的笛卡尔坐标，具有绝对值`abs`和角度`angle`。                                          |
| `heaviside`            | 为`input`中的每个元素计算Heaviside阶跃函数。                                                                              |

## Indexing, Slicing, Joining and Mutating Ops （索引，切片，组合，变换操作）

|||
|-|-|
|`adjoint`|返回共轭张量的视图，并将最后两个维度进行转置。|
|`argwhere`|返回一个张量，其中包含`input`的所有非零元素的索引。|
|`cat`|在给定维数中拼接给定序列的`seq`张量。|
|`concat`|`torch.cat()` 的别名|
|`concatenate`|`torch.cat()` 的别名|
|`conj`|返回具有翻转共轭位的`input`视图。|
|`chunk`|尝试将张量拆分为指定数量的块。|
|`dsplit`|根据`indices_or_sections`将具有三维或更多维度的`input`张量分割为多个张量。|
|`column_stack`|通过水平叠加`tensors`来创建一个新的张量。|
|`dstack`|按顺序深度堆叠张量(沿着第三个轴)。|
|`gather`|沿着由*dim*指定的轴取值。|
|`hsplit`|根据`indices_or_sections`将`input`(一个或多个维度的张量)水平分割成多个张量。|
|`hstack`|按水平顺序堆叠张量(按列堆叠)。|
|`index_add`|函数描述请参见`index_add_()`。|
|`index_copy`|函数描述请参见`index_add_()`。|
|`index_reduce`|函数描述请参见`index_reduce_()`。|
|`index_select`|返回一个新的张量，该张量使用`index`中的项沿维度`dim`对`input`张量进行索引，该索引是一个*LongTensor*。|
|`masked_select`|返回一个新的1-D张量，它根据布尔掩码`mask`(*BoolTensor*)对`input`张量进行索引。|
|`movedim`|将`source`位置处的`input`维数移动到`destination`位置。|
|`moveaxis`|`torch.movedim()`的别名|
|`narrow`|返回一个新的张量，它是`input`张量的缩小版本。|
|`narrow_copy`|与`Tensor.narrow()`相同，但返回副本而不是共享存储。|
|`permute`|返回原始张量`input`的视图，其维度进行了重新排列。|
|`reshape`|返回具有与`input`相同数据和元素数量的张量，但具有指定的形状。|
|`row_stack`|`torch.vstack()`的别名|
|`select`|在给定索引处沿选定维度对`input`张量进行切片。|
|`scatter`|`torch.Tensor.scatter_()`的异址(Out-of-place)版本|
|`diagnal_scatter`|将`src`张量的值沿着`input`的对角线元素嵌入，取`dim1`和`dim2`两个维度。|
|`select_scatter`|将`src`张量的值嵌入到`input`中的给定索引处。|
|`slice_scatter`|将`src`张量的值嵌入到`input`中的给定维度。|
|`scatter_add`|`torch.scatter_add_()`的异址(Out-of-place)版本|
|`scatter_reduce`|`torch.scatter_reduce_()`的异址(Out-of-place)版本|
|`split`|将张量分成块。|
|`squeeze`|返回一个张量，为`input`删除了所有大小为1的维度。|
|`stack`|沿着一个新的维度拼接一个张量序列。|
|`swapaxes`|`torch.transpose()`的别名|
|`swapdims`|`torch.transpose()`的别名|
|`t`|期望`input`为维度不大于2的张量，并将维度0和1转置。|
|`take`|返回由`input`指定索引处的元素组成的张量|
|`take_along_dim`|沿着给定`dim`，从输入中`indices`的一维索引处选择值。|
|`tensor_split`|沿着维度`dim`根据索引或由`indices_or_sections`指定的节数,将一个张量分成多个子张量，所有子张量都是`input`的视图。|
|`tile`|通过重复`input`的元素来构造一个张量。|
|`transpose`|返回`input`转置后的一个张量|
|`unbind`|移除张量的一个维度|
|`unsuqeeze`|在张量的特定位置插入一个新的维度，维度大小为1|
|`vsplit`|根据`indices_or_sections`将具有两个或多个维度的张量垂直分割为多个张量。|
|`vstack`|垂直顺序堆叠张量(逐行)。|
|`where`|根据`condition`，返回从`input`或`other`中选择的元素张量。|

## Generators （生成器）

|||
|-|-|
|`Generator`|创建并返回一个生成器对象，该对象管理生成伪随机数的算法的状态。|

## Random sampling （随机抽样）

|||
|-|-|
|`seed`|将生成随机数的种子设置为不确定随机数。|
|`manual_seed`|为生成随机数设置种子。|
|`initial_seed`|以Python Long形式返回生成随机数的初始种子|
|`get_rng_state`|以torch.ByteTensor 形式返回随机数生成器状态|
|`set_rng_state`|设置随机数生成器状态|

> torch.default_generator 返回默认的CPU torch.Generator

|||
|-|-|
|`bernoulli`|从伯努利分布中采样二进制随机数(0或1)。|
|`multinomial`|返回一个张量，从`input`的相应行中的多项概率分布中采样，其中每行包含`num_samples`索引。|
|`normal`|返回一个随机数字的张量，这些随机数是从单独的正态分布中抽取的，这些正态分布的平均值和标准差都是给定的。|
|`poisson`|返回一个与`input`相同大小的张量，其中每个元素从泊松分布中采样，其速率参数由输入中的相应元素给出。|
|`rand`|返回一个由区间$[0,1)$上均匀分布的随机数填充的张量。|
|`rand_like`|返回一个由区间$[0,1)$上均匀分布的随机数填充的张量，大小与`input`相同。|
|`randint`|返回一个张量，由$[low,high)$区间均匀生成的随机整数填充。|
|`randint_like`|返回一个张量，由$[low,high)$区间均匀生成的随机整数填充，大小与`input`相同。|
|`randn`|返回一个张量，以均值为0，方差为1的正态分布(也称为标准正态分布)的随机数填充。|
|`randn_like`|返回一个张量，以均值为0，方差为1的正态分布(也称为标准正态分布)的随机数填充，大小与`input`相同。|
|`randperm`|返回从$0$到$n - 1$的整数的随机排列。|

### In-place random sampling （原址随机抽样）

还有一些在张量上定义的原址随机抽样函数。点击查看它们的文档:

TODO：添加超链接

* `torch.Tensor.bernoulli_()` - `torch.bernoulli()`的原址版本。
* `torch.Tensor.cauchy_()` - 从Cauchy分布中采样的数据。
* `torch.Tensor.exponential_()` - 从指数分布中采样的数据。
* `torch.Tensor.geometric_()` - 从几何分布中采样的元素。
* `torch.Tensor.log_normal_()` - 从对数正态分布中采样。
* `torch.Tensor.normal_()` - `torch.normal()`的原址版本。
* `torch.Tensor.random_()` - 从离散均匀分布中采样的数据。
* `torch.Tensor.uniform_()` - 从连续均匀分布中采样的数据。

### Quasi-random sampling（拟随机抽样）

|||
|-|-|
|`quasirandom.SobolEngine`|torch.quasirandom.SobolEngine是一个生成(乱序)Sobol序列的引擎。|

## Serialization （序列化）

|||
|-|-|
|`save`|将对象存储为磁盘文件|
|`load`|从磁盘文件中载入以`torch.save()`存储的对象|

## Parallelism （并行）

|||
|-|-|
|`get_num_threads`|返回用于并行处理CPU操作的线程数|
|`set_num_threads`|设置CPU上用于操作内并行性的线程数。|
|`get_num_interop_threads`|返回CPU上用于操作间并行性的线程数。|
|`set_num_interop_threads`|设置用于操作间并行性的线程数。|

## Locally disabling gradient computation （局部禁用梯度计算）

上下文管理器`torch.no_grad()`、`torch.enable_grad()`和`torch.set_grad_enabled()`有助于在局部禁用和启用梯度计算。有关其用法的更多详细信息，请参阅 Locally disabling gradient computation。这些上下文管理器是线程本地的，所以如果你使用`threading`模块将工作发送给另一个线程，它们将不起作用。

示例：

```py
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False

>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True

>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False

```

|||
|-|-|
|`no_grad`|禁用梯度计算|
|`enable_grad`|启用梯度计算|
|`set_grad_enabled`|设置梯度计算启用或禁用|
|`is_grad_enabled`|如果梯度模式启用，返回True|
|`inference_mode`|启用或禁用推理模式|
|`is_inference_mode_enabled`|如果当前推理模式启用，返回True|

## Math operations （数学操作）

### Pointwise Ops （点级操作）

|||
|-|-|
|`abs`|计算`input`中每个元素的绝对值|
|`absolute`|`torch.abs()`的别名|
|`acos`|计算`input`中每个元素的反余弦函数值|
|`arccos`|`torch.acos()`的别名|
|`acosh`|计算`input`中每个元素的反双曲余弦函数值|
|`arccosh`|`torch.acosh()`的别名|
|`add`|将`other`乘以`alpha`倍，加到`input`上|
|`addcdiv`|执行`tensor1`除以`tensor2`的逐元素除法，将结果乘以标量`value`并将其加到`input`。|
|`addcmul`|执行`tensor1`乘`tensor2`的逐元素乘法，将结果乘以标量`value`并将其加到`input`。|
|`angle`|计算`input`中的元素的弧度值|
|`asin`|返回一个张量，为`input`中各个元素的acrsin值|
|`arcsin`|`torch.asin()`的别名|
|`asinh`|返回一个张量，为`input`中各个元素的反双曲正弦值|
|`arcsinh`|`torch.asin()`的别名|
|`atan`|返回一个张量，为`input`中各个元素的arctan值|
|`arctan`|`torch.atan()`的别名|
|`atanh`|返回一个张量，为`input`中各个元素的双曲正切值|
|`arctanh`|`torch.atanh()`的别名|
|`atan2`|考虑象限的$\frac{input_i}{other_i}$的逐元素反正切。|
|`arctan2`|`torch.atan2()`的别名|
|`bitwise_not`|计算给定输入张量的按位非|
|`bitwise_and`|计算给定输入张量`input`和`other`的按位与|
|`bitwise_or`|计算给定输入张量`input`和`other`的按位或|
|`bitwise_xor`|计算给定输入张量`input`和`other`的按位异或|
|`bitwise_left_shift`|计算`input`的左算术移`other`位。|
|`bitwise_right_shift`|计算`input`的右算术移`other`位。|
|`ceil`|返回输入张量的上界，即大于等于所有元素的最小整数|
|`clamp`|将张量的元素夹在区间[`min`,`max`]中|
|`clip`|`torch.clamp()`的别名|
|`conj_physical`|计算给定输入张量各元素的共轭复数。|
|`copysign`|创建一个新的浮点张量，其大小为`input`，符号为`other`。|
|`cos`|计算`input`中各元素的余弦值|
|`cosh`|计算`input`中各元素的双曲余弦值|
|`deg2rad`|将`input`中的角度值转为弧度|
|`div`|返回输入`input`的每个元素除以`other`的对应元素。|
|`divide`|`torch.div()`的别名|
|`digamma`|`torch.special.digamma()`的别名|
|`erf`|`torch.special.erf()`的别名|
|`erfc`|`torch.special.erfc()`的别名|
|`exp`|返回一个新的张量，其中包含输入张量`input`元素的指数。|
|`exp2`|`torch.special.exp2()`的别名|
|`expm1`|`torch.special.expm1()`的别名|

### Reduction Ops (归纳操作)

|||
|-|-|
|`argmax`|返回`input`张量中所有最大元素的索引|
|`argmin`|返回展开的`input`张量，或沿一个维度中所有最小元素的索引|
|`amax`|返回沿给定维度将`input`切片后，每个切片的最大值|
|`amin`|返回沿给定维度将`input`切片后，每个切片的最大值|
|`aminmax`|计算`input`张量中最小值和最大值|
|`all`|测试`input`中是否所有元素求值为True。|
|`any`|测试`input`中是否存在元素求值为True。|
|`max`|返回`input`张量中所有元素的最大值。|
|`min`|返回`input`张量中所有元素的最小值。|
|`dist`|返回(`input` - `other`)的p-范数|
|`logsumexp`|返回`input`张量给定维度dim中的每一行求和指数的对数。|
|`mean`|返回`input`张量中所有元素的平均值。|
|`nanmean`|返回`input`张量中所有非Nan元素的最大值。|
|`median`|返回`input`张量中元素的中位数。|
|`nanmedian`|返回`input`张量中非Nan元素的中位数。|
|`mode`|返回一个namedtuple`(values, indexes)`，其中`values`是给定维度`dim`中输入张量的每一行的模值，即在该行中出现最频繁的值，而`indexes`是找到的每个模式值的索引位置。|
|`norm`|返回给定张量的矩阵范数或向量范数。|
|`nansum`|返回所有元素的和，将非数字(Nan)视为零。|
|`prod`|返回`input`张量中所有元素的乘积。|
|`quantile`|沿维度`dim`计算`input`张量的每一行的第q个分位数。|
|`nanquantile`|这是`torch.quantile()`的一个变体，它“忽略”NaN值，计算分位数q，就好像输入中的NaN值不存在一样。|
|`std`|计算由`dim`指定的维度上的标准差。|
|`std_mean`|计算由`dim`指定的维度上的标准差和平均数。|
|`sum`|返回`input`张量中所有元素的和。|
|`unique`|返回输入张量的唯一元素。|
|`unique_consecutive`|从每个连续的等效元素组中除去第一个元素以外的所有元素。|
|`var`|计算由`dim`指定的维度上的方差。|
|`var_mean`|计算由`dim`指定的维度上的方差和平均数。|
|`count_nonzero`|沿指定维度为`input`张量的非零元素计数。|

### Comparison Ops (比较操作)

|||
|-|-|
|`allclose`|检验 `input` 和 `others` 是否满足条件|
|`argsort`|返回沿给定维度按值升序对张量排序的索引。|
|`eq`|计算元素相等性|
|`equal`|如果两个张量尺寸和元素都相同为`True`，否则为`False`|
|`ge`|按元素计算`input`$\geq$ `other`|
|`greater_equal`|`torch.ge()`的别名|
|`gt`|按元素计算`input`$>$ `other`|
|`greater`|`torch.gt()`的别名|
|`isclose`|返回一个新的张量，其中包含布尔元素，表示`input`的每个元素是否与`other`的对应元素“接近”|
|`isfinite`|返回一个新的张量，其中包含布尔元素，表示每个元素是否有限。|
|`isin`|测试是否`elements`的所有元素都在`test_elements`中|
|`isinf`|测试`input`的每个元素是否为无穷大(正无穷大或负无穷大)|
|`isposinf`|测试`input`的每个元素是否为正无穷大|
|`isneginf`|测试`input`的每个元素是否为负无穷大|
|`isnan`|返回一个带有布尔元素的新张量，表示`input`的每个元素是否为`NaN`。|
|`isreal`|返回一个带有布尔元素的新张量，表示`input`的每个元素是否为实数。|
|`kthvalue`|返回一个命名元组`(values, indexes)`，其中`values`是给定维度`dim`中`input`张量每行的第`k`个最小元素。|
|`le`|按元素计算`input`$\leq$`other`|
|`less_equal`|`torch.le()`的别名|
|`lt`|按元素计算`input`$<$ `other`|
|`less`|`torch.lt()`的别名|
|`maximum`|计算`input`和`other`元素的最大值。|
|`minimum`|计算`input`和`other`元素的最小值。|
|`fmax`|与`torch.maximum()`相同，但处理`Nan`的方式不同|
|`fmin`|与`torch.minimum()`相同，但处理`Nan`的方式不同|
|`ne`|按元素计算`input`$\neq$`other`|
|`not_euqal`|`torch.ne()`的别名|
|`sort`|将`input`张量的元素沿给定维度按值升序排序。|
|`topk`|返回给定`input`张量沿给定维度的k个最大元素。|
|`msort`|将`input`张量的元素沿其第一维按值升序排序。|

### Spectral Ops

### Other Operations

### BLAS and LAPACK Operations

### Foreach Operations

