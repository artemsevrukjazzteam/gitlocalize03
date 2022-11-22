# 将元数据添加到 TensorFlow Lite 模型

TensorFlow Lite 元数据为模型描述提供了标准。元数据是有关模型功能及其输入/输出信息的重要知识来源。元数据由两者组成

- 传达使用模型时的最佳实践的人类可读部分，以及
- 代码生成器可以利用的机器可读部分，例如[TensorFlow Lite Android 代码生成器](../../inference_with_metadata/codegen#generate-code-with-tensorflow-lite-android-code-generator)和[Android Studio ML 绑定功能](../../inference_with_metadata/codegen#generate-code-with-android-studio-ml-model-binding)。

在[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite)上发布的所有图像模型都已填充元数据。

## 具有元数据格式的模型

<center><img src="../../images/convert/model_with_metadata.png" alt="model_with_metadata" width="70%"></center>
<center>图 1. 带有元数据和关联文件的 TFLite 模型。</center>

模型元数据在[metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs)中定义，这是一个[FlatBuffer](https://google.github.io/flatbuffers/index.html#flatbuffers_overview)文件。如图 1 所示，它存储在[TFLite 模型模式](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs)的[元数据](https://github.com/tensorflow/tensorflow/blob/bd73701871af75539dd2f6d7fdba5660a8298caf/tensorflow/lite/schema/schema.fbs#L1208)字段中，名称为`"TFLITE_METADATA"` 。一些模型可能带有相关文件，例如[分类标签文件](https://github.com/tensorflow/examples/blob/dd98bc2b595157c03ac9fa47ac8659bb20aa8bbd/lite/examples/image_classification/android/models/src/main/assets/labels.txt#L1)。使用 ZipFile [“附加”模式](https://pymotw.com/2/zipfile/#appending-to-files)（ `'a'`模式）将这些文件作为 ZIP 连接到原始模型文件的末尾。 TFLite Interpreter 可以像以前一样使用新文件格式。有关详细信息，请参阅[打包关联文件](#pack-the-associated-files)。

请参阅下面有关如何填充、可视化和读取元数据的说明。

## 设置元数据工具

在将元数据添加到模型之前，您需要设置 Python 编程环境来运行 TensorFlow。[这里](https://www.tensorflow.org/install)有关于如何设置的详细指南。

设置 Python 编程环境后，您将需要安装额外的工具：

```sh
pip install tflite-support
```

TensorFlow Lite 元数据工具支持 Python 3。

## 使用 Flatbuffers Python API 添加元数据

注意：要为[TensorFlow Lite Task Library](../../inference_with_metadata/task_library/overview)中支持的流行 ML 任务创建元数据，请使用[TensorFlow Lite Metadata Writer Library](metadata_writer_tutorial.ipynb)中的高级 API。

[模式](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs)中的模型元数据分为三个部分：

1. **模型信息**- 模型的总体描述以及许可条款等项目。请参阅[模型元数据](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L640)。
2. **输入信息**- 输入的描述和所需的预处理，例如规范化。请参阅[SubGraphMetadata.input_tensor_metadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L590) 。
3. **输出信息**- 输出和所需后处理的描述，例如映射到标签。请参阅[SubGraphMetadata.output_tensor_metadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L599) 。

由于此时 TensorFlow Lite 仅支持单个子图，因此在显示元数据和生成代码时， [TensorFlow Lite 代码生成器](../../inference_with_metadata/codegen#generate-code-with-tensorflow-lite-android-code-generator)和[Android Studio ML 绑定功能](../../inference_with_metadata/codegen#generate-code-with-android-studio-ml-model-binding)将使用`ModelMetadata.name`和`ModelMetadata.description` ，而不是`SubGraphMetadata.name`和`SubGraphMetadata.description` 。

### 支持的输入/输出类型

输入和输出的 TensorFlow Lite 元数据在设计时并未考虑特定的模型类型，而是输入和输出类型。不管模型的功能是什么，只要输入和输出类型包含以下内容或以下内容的组合，TensorFlow Lite 元数据就支持它：

- 特征 - 无符号整数或 float32 的数字。
- 图像 - 元数据目前支持 RGB 和灰度图像。
- 边界框 - 矩形边界框。该架构支持[多种编号方案](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L214)。

### 打包相关文件

TensorFlow Lite 模型可能带有不同的关联文件。例如，自然语言模型通常有将单词片段映射到单词 ID 的 vocab 文件；分类模型可能具有指示对象类别的标签文件。如果没有关联文件（如果有），模型将无法正常运行。

现在可以通过元数据 Python 库将关联文件与模型捆绑在一起。新的 TensorFlow Lite 模型成为一个包含模型和关联文件的 zip 文件。可以用常用的zip工具解压。这种新的模型格式继续使用相同的文件扩展名`.tflite` 。它与现有的 TFLite 框架和解释器兼容。有关详细信息，请参阅[将元数据和关联文件打包到模型](#pack-metadata-and-associated-files-into-the-model)中。

关联的文件信息可以记录在元数据中。根据文件类型和文件附加到的位置（即`ModelMetadata` 、 `SubGraphMetadata`和`TensorMetadata` ）， [TensorFlow Lite Android 代码生成器](../../inference_with_metadata/codegen)可能会自动对对象应用相应的预处理/后处理。有关更多详细信息，请参阅架构[中每个关联文件类型的 &lt;Codegen usage&gt; 部分](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L77-L127)。

### 归一化和量化参数

归一化是机器学习中常见的数据预处理技术。规范化的目标是将值更改为通用尺度，而不扭曲值范围内的差异。

[模型量化](https://www.tensorflow.org/lite/performance/model_optimization#model_quantization)是一种允许降低权重表示精度的技术，并且可以选择用于存储和计算的激活。

就预处理和后处理而言，归一化和量化是两个独立的步骤。这是详细信息。

 | 正常化 | 量化
:-: | --- | ---
\ | **浮动模型**：\ | **浮动模型**：\
: 一个例子: - 均值: 127.5 \ : - zeroPoint: 0 \ : |  |
: 的参数值: - std: 127.5 \ : - scale: 1.0 \ : |  |
: input image in :**量化模型**: \ :**量化模型**: \ : |  |
: MobileNet for float and : - mean: 127.5 \ : - zeroPoint: 128.0 \ : |  |
：量化模型，：-标准：127.5：-比例：0.0078125f \： |  |
： 分别。 : : : |  |
\ | \ | **浮动模型**确实
: \ : \ : 不需要量化。 \ : |  |
：\：**输入**：如果输入：**量化模型**可能： |  |
: \ : 数据标准化为 : 或可能不需要 : |  |
: 什么时候调用？ ：训练，输入：前/后量化： |  |
: : 推理需要的数据 : 处理。这取决于 ： |  |
: : 规范化 : 在数据类型上 : |  |
： ： 因此。 \ ：输入/输出张量。 \ : |  |
：：**输出**：输出：-浮点张量：否： |  |
: : 数据不会是 : 前/后量化 : |  |
: : 一般归一化。 : 需要处理。数量： |  |
: : : op 和 dequant op 是： |  |
: : : 烘焙到模型中： |  |
: : : 图。 \ : |  |
: : : - int8/uint8 张量: : |  |
: : : 需要量化： |  |
: : : 前/后处理。 : |  |
\ | \ | **量化输入**：
: \ : \ : \ : |  |
: 公式 : normalized_input = : q = f / scale + : |  |
: : (输入 - 平均值) / std : zeroPoint \ : |  |
: : : **反量化： |  |
：：：输出**：\： |  |
: : : f = (q - zeroPoint) * : |  |
： ： ： 规模 ： |  |
\ | 由模型创建者填写 | 自动填写
: : 和存储在模型 : TFLite 转换器中的位置，以及 : |  |
：参数：元数据，如：存储在 tflite 模型中： |  |
：： `NormalizationOptions`选项：文件。 : |  |
如何获得 | 通过 | 通过 TFLite
： 参数？ : `MetadataExtractor` API : `Tensor` API [1] 或 : |  |
: : [2] : 通过 : |  |
`MetadataExtractor` API： |  |
: : : [2] : |  |
做浮动和量化 | 是的，浮动和量化 | 不，浮动模型确实如此
: models share the same : models have the same : 不需要量化。 : |  |
： 价值？ ：归一化：： |  |
： ： 参数 ： ： |  |
TFLite 代码 | \ | \
: 发电机或 Android : 是 : 是 : |  |
: Studio ML 绑定 : : : |  |
: 自动生成 : : : |  |
: 它在数据处理？ : : : |  |

[1] [TensorFlow Lite Java API](https://github.com/tensorflow/tensorflow/blob/09ec15539eece57b257ce9074918282d88523d56/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Tensor.java#L73)和[TensorFlow Lite C++ API](https://github.com/tensorflow/tensorflow/blob/09ec15539eece57b257ce9074918282d88523d56/tensorflow/lite/c/common.h#L391) 。
 [2][元数据提取器库](#read-the-metadata-from-models)

处理 uint8 模型的图像数据时，有时会跳过归一化和量化。当像素值在 [0, 255] 范围内时这样做是可以的。但一般来说，您应该始终根据适用的归一化和量化参数来处理数据。

如果您在元数据中设置`NormalizationOptions` ， [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/overview)可以为您处理规范化。量化和反量化处理始终是封装的。

### 例子

注意：指定的导出目录必须在运行脚本之前存在；它不会作为流程的一部分创建。

您可以在此处找到有关如何为不同类型的模型填充元数据的示例：

#### 图片分类

在[此处](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py)下载脚本，它将元数据填充到[mobilenet_v1_0.75_160_quantized.tflite](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/default/1) 。像这样运行脚本：

```sh
python ./metadata_writer_for_image_classifier.py \
    --model_file=./model_without_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --label_file=./model_without_metadata/labels.txt \
    --export_directory=model_with_metadata
```

要为其他图像分类模型填充元数据，请将模型规范添加[到](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py#L63-L74)脚本中。本指南的其余部分将突出显示图像分类示例中的一些关键部分，以说明关键元素。

### 深入研究图像分类示例

#### 型号信息

元数据首先创建一个新的模型信息：

```python
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

""" ... """
"""Creates the metadata for an image classifier."""

# Creates model info.

model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "MobileNetV1 image classifier"
model_meta.description = ("Identify the most prominent object in the "
                          "image from a set of 1,001 categories such as "
                          "trees, animals, food, vehicles, person etc.")
model_meta.version = "v1"
model_meta.author = "TensorFlow"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")
```

#### 输入/输出信息

本节向您展示如何描述模型的输入和输出签名。自动代码生成器可以使用此元数据来创建预处理和后处理代码。要创建有关张量的输入或输出信息：

```python
# Creates input info.

input_meta = _metadata_fb.TensorMetadataT()

# Creates output info.

output_meta = _metadata_fb.TensorMetadataT()
```

#### 图片输入

图像是机器学习的常见输入类型。 TensorFlow Lite 元数据支持颜色空间等信息和归一化等预处理信息。图像的维度不需要手动指定，因为它已经由输入张量的形状提供并且可以自动推断。

```python
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(160, 160))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5]
input_normalization.options.std = [127.5]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats
```

#### 标签输出

可以使用`TENSOR_AXIS_LABELS`通过关联文件将标签映射到输出张量。

```python
# Creates output info.

output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 1001 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename("your_path_to_label_file")
label_file.description = "Labels for objects that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]
```

#### 创建元数据 Flatbuffers

以下代码将模型信息与输入和输出信息组合在一起：

```python
# Creates subgraph info.

subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()
```

#### 将元数据和相关文件打包到模型中

创建元数据 Flatbuffers 后，元数据和标签文件将通过`populate`方法写入 TFLite 文件：

```python
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["your_path_to_label_file"])
populator.populate()
```

您可以通过`load_associated_files`将任意数量的关联文件打包到模型中。但是，至少需要打包元数据中记录的那些文件。在这个例子中，打包标签文件是强制性的。

## 可视化元数据

您可以使用[Netron](https://github.com/lutzroeder/netron)可视化您的元数据，或者您可以使用`MetadataDisplayer`将 TensorFlow Lite 模型中的元数据读取为 json 格式：

```python
displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
export_json_file = os.path.join(FLAGS.export_directory,
                    os.path.splitext(model_basename)[0] + ".json")
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file

with open(export_json_file, "w") as f:
  f.write(json_file)
```

Android Studio 还支持通过[Android Studio ML Binding 功能](https://developer.android.com/studio/preview/features#tensor-flow-lite-models)显示元数据。

## 元数据版本控制

[元数据模式](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs)由跟踪模式文件更改的语义版本号和指示真实版本兼容性的 Flatbuffers 文件标识进行版本控制。

### 语义版本号

元数据模式由[语义版本号控制](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L53)，例如 MAJOR.MINOR.PATCH。它根据[此处](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L32-L44)的规则跟踪架构更改。查看版本`1.0.0`之后添加[的字段的历史记录](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L63)。

### Flatbuffers文件识别

语义版本控制在遵循规则的情况下保证兼容性，但并不意味着真正的不兼容。当增加 MAJOR 编号时，并不一定意味着向后兼容性被破坏。因此，我们使用[Flatbuffers 文件标识](https://google.github.io/flatbuffers/md__schemas.html)[file_identifier](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L61)来表示元数据模式的真正兼容性。文件标识符正好是 4 个字符长。它固定在一定的元数据模式中，用户不得更改。如果由于某种原因必须破坏元数据模式的向后兼容性，则 file_identifier 将会增加，例如，从“M001”到“M002”。预计 File_identifier 的更改频率远低于 metadata_version。

### 最低必需的元数据解析器版本

[最低必需的元数据解析器版本](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L681)是可以完整读取元数据 Flatbuffers 的元数据解析器（Flatbuffers 生成的代码）的最低版本。该版本实际上是所有填充字段的版本中最大的版本号和文件标识符指示的最小兼容版本。当元数据填充到 TFLite 模型中时， `MetadataPopulator`会自动填充最低必需的元数据解析器版本。有关如何使用最低必要元数据解析器版本的更多信息，请参阅[元数据提取器](#read-the-metadata-from-models)。

## 从模型中读取元数据

Metadata Extractor 库是一种方便的工具，可以从不同平台的模型中读取元数据和相关文件（请参阅[Java 版本](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/java)和[C++ 版本](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/cc)）。您可以使用 Flatbuffers 库以其他语言构建自己的元数据提取器工具。

### 读取 Java 中的元数据

要在您的 Android 应用程序中使用元数据提取器库，我们建议使用[托管在 MavenCentral 的 TensorFlow Lite 元数据 AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-metadata) 。它包含`MetadataExtractor`类，以及[元数据模式](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs)和[模型模式](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs)的 FlatBuffers Java 绑定。

您可以在`build.gradle`依赖项中指定它，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.1.0'
}
```

要使用夜间快照，请确保您已添加[Sonatype 快照存储库](https://www.tensorflow.org/lite/android/lite_build#use_nightly_snapshots)。

您可以使用指向模型的`ByteBuffer`初始化`MetadataExtractor`对象：

```java
public MetadataExtractor(ByteBuffer buffer);
```

`ByteBuffer`必须在`MetadataExtractor`对象的整个生命周期内保持不变。如果模型元数据的 Flatbuffers 文件标识符与元数据解析器的标识符不匹配，初始化可能会失败。有关详细信息，请参阅[元数据版本控制](#metadata-versioning)。

有了匹配的文件标识符，由于 Flatbuffers 的向前和向后兼容机制，元数据提取器将成功读取从所有过去和未来模式生成的元数据。但是，旧的元数据提取器无法提取未来模式中的字段。[最低必需的元数据解析器版本](#the-minimum-necessary-metadata-parser-version)表示可以完整读取元数据 Flatbuffers 的最低元数据解析器版本。您可以使用以下方法来验证是否满足最低必要的解析器版本条件：

```java
public final boolean isMinimumParserVersionSatisfied();
```

允许传入没有元数据的模型。但是，调用从元数据中读取的方法会导致运行时错误。您可以通过调用`hasMetadata`方法检查模型是否具有元数据：

```java
public boolean hasMetadata();
```

`MetadataExtractor`为您提供了方便的函数来获取输入/输出张量的元数据。例如，

```java
public int getInputTensorCount();
public TensorMetadata getInputTensorMetadata(int inputIndex);
public QuantizationParams getInputTensorQuantizationParams(int inputIndex);
public int[] getInputTensorShape(int inputIndex);
public int getoutputTensorCount();
public TensorMetadata getoutputTensorMetadata(int inputIndex);
public QuantizationParams getoutputTensorQuantizationParams(int inputIndex);
public int[] getoutputTensorShape(int inputIndex);
```

尽管[TensorFlow Lite 模型架构](https://github.com/tensorflow/tensorflow/blob/aa7ff6aa28977826e7acae379e82da22482b2bf2/tensorflow/lite/schema/schema.fbs#L1075)支持多个子图，但 TFLite 解释器目前仅支持单个子图。因此， `MetadataExtractor`在其方法中省略了子图索引作为输入参数。

## 从模型中读取关联文件

带有元数据和关联文件的 TensorFlow Lite 模型本质上是一个 zip 文件，可以使用常用的 zip 工具解压以获取关联文件。比如可以解压[mobilenet_v1_0.75_160_quantized](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/metadata/1) ，提取模型中的label文件，如下：

```sh
$ unzip mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
Archive:  mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
 extracting: labels.txt
```

您还可以通过元数据提取器库读取关联文件。

在 Java 中，将文件名传递给`MetadataExtractor.getAssociatedFile`方法：

```java
public InputStream getAssociatedFile(String fileName);
```

同样，在 C++ 中，这可以通过方法`ModelMetadataExtractor::GetAssociatedFile` ：

```c++
tflite::support::StatusOr<absl::string_view> GetAssociatedFile(
      const std::string& filename) const;
```
