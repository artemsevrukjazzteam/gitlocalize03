# Добавление метаданных в модели TensorFlow Lite
# qqq Добавление метаданных в модели TensorFlow Lite

Метаданные TensorFlow Lite обеспечивают стандарт для описания моделей. Метаданные являются важным источником знаний о том, что делает модель, и ее входной/выходной информации. Метаданные состоят из обоих

- удобочитаемые части, передающие лучшие практики использования модели, и
- машиночитаемые части, которые могут использоваться генераторами кода, такими как генератор [кода TensorFlow Lite для Android](../../inference_with_metadata/codegen#generate-code-with-tensorflow-lite-android-code-generator) и [функция Android Studio ML Binding](../../inference_with_metadata/codegen#generate-code-with-android-studio-ml-model-binding) .

Все модели изображений, опубликованные в [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) , были заполнены метаданными.

## Модель с форматом метаданных
## ffff Модель с форматом метаданных

<center><img src="../../images/convert/model_with_metadata.png" alt="model_with_metadata" width="70%"></center>
<center>Рис. 1. Модель TFLite с метаданными и соответствующими файлами.</center>

Метаданные модели определяются в файле [metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) [FlatBuffer](https://google.github.io/flatbuffers/index.html#flatbuffers_overview) . Как показано на рис. 1, он хранится в поле [метаданных](https://github.com/tensorflow/tensorflow/blob/bd73701871af75539dd2f6d7fdba5660a8298caf/tensorflow/lite/schema/schema.fbs#L1208) [схемы модели TFLite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs) под именем `"TFLITE_METADATA"` . Некоторые модели могут поставляться с соответствующими файлами, такими как [файлы классификационных меток](https://github.com/tensorflow/examples/blob/dd98bc2b595157c03ac9fa47ac8659bb20aa8bbd/lite/examples/image_classification/android/models/src/main/assets/labels.txt#L1) . Эти файлы объединяются в конец исходного файла модели в виде ZIP-файла с использованием режима «дополнения [»](https://pymotw.com/2/zipfile/#appending-to-files) ZipFile (режим `'a'` ). Интерпретатор TFLite может использовать новый формат файла так же, как и раньше. Дополнительные сведения см. в разделе [Упаковка связанных файлов](#pack-the-associated-files) .

См. инструкции ниже о том, как заполнять, визуализировать и читать метаданные.

## Настройте инструменты метаданных

Перед добавлением метаданных в вашу модель вам потребуется настроить среду программирования Python для запуска TensorFlow. Подробное руководство о том, как это настроить, есть [здесь](https://www.tensorflow.org/install) .

После настройки среды программирования Python вам потребуется установить дополнительные инструменты:

```sh
pip install tflite-support
```

Инструменты метаданных TensorFlow Lite поддерживают Python 3.

## Добавление метаданных с помощью Flatbuffers Python API

Примечание: для создания метаданных для популярных задач машинного обучения, поддерживаемых в библиотеке задач [TensorFlow Lite](../../inference_with_metadata/task_library/overview) , используйте высокоуровневый API в [библиотеке записи метаданных TensorFlow Lite](metadata_writer_tutorial.ipynb) .

В [схеме](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) метаданные модели состоят из трех частей:

1. **Информация о модели** — общее описание модели, а также такие элементы, как условия лицензии. См. [Метаданные модели](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L640) .
2. **Входная информация** — описание входных данных и необходимая предварительная обработка, например нормализация. См. [SubGraphMetadata.input_tensor_metadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L590) .
3. **Выходная информация** — описание требуемых выходных данных и постобработки, например сопоставление с метками. См. [SubGraphMetadata.output_tensor_metadata](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L599) .

Поскольку на данный момент TensorFlow Lite поддерживает только один подграф, [генератор кода TensorFlow Lite](../../inference_with_metadata/codegen#generate-code-with-tensorflow-lite-android-code-generator) и [функция Android Studio ML Binding](../../inference_with_metadata/codegen#generate-code-with-android-studio-ml-model-binding) будут использовать `ModelMetadata.name` и `ModelMetadata.description` вместо `SubGraphMetadata.name` и `SubGraphMetadata.description` при отображении метаданных и создании кода.

### Поддерживаемые типы ввода/вывода

Метаданные TensorFlow Lite для ввода и вывода разрабатываются не с учетом конкретных типов моделей, а скорее с учетом типов ввода и вывода. Неважно, что функционально делает модель, пока типы ввода и вывода состоят из следующих или их комбинации, они поддерживаются метаданными TensorFlow Lite:

- Особенность - числа, которые являются целыми числами без знака или float32.
- Изображение. В настоящее время метаданные поддерживают изображения RGB и оттенки серого.
- Ограничивающий прямоугольник — ограничивающие прямоугольники прямоугольной формы. Схема поддерживает [множество схем нумерации](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L214) .

### Упакуйте связанные файлы

Модели TensorFlow Lite могут поставляться с разными связанными файлами. Например, модели естественного языка обычно имеют словарные файлы, которые сопоставляют части слов с идентификаторами слов; модели классификации могут иметь файлы меток, которые указывают категории объектов. Без связанных файлов (если они есть) модель не будет работать должным образом.

Связанные файлы теперь можно объединить с моделью через библиотеку метаданных Python. Новая модель TensorFlow Lite становится ZIP-файлом, содержащим как модель, так и связанные файлы. Его можно распаковать обычными zip-инструментами. Этот новый формат модели продолжает использовать то же расширение файла, `.tflite` . Он совместим с существующей структурой TFLite и интерпретатором. Дополнительные сведения см. в разделе [«Упаковать метаданные и связанные файлы в модель»](#pack-metadata-and-associated-files-into-the-model) .

Связанная информация о файле может быть записана в метаданные. В зависимости от типа файла и того, к чему прикреплен файл (например, `ModelMetadata` , `SubGraphMetadata` и `TensorMetadata` ), [генератор кода Android TensorFlow Lite](../../inference_with_metadata/codegen) может автоматически применять к объекту соответствующую предварительную/постобработку. Дополнительную информацию см. в разделе [&lt;Использование Codegen&gt; каждого ассоциированного типа файла](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L77-L127) в схеме.

### Параметры нормализации и квантования

Нормализация — это распространенный метод предварительной обработки данных в машинном обучении. Цель нормализации — привести значения к единой шкале, не искажая различий в диапазонах значений.

[Квантование модели](https://www.tensorflow.org/lite/performance/model_optimization#model_quantization) — это метод, который позволяет представлять веса с уменьшенной точностью и, при необходимости, активировать как для хранения, так и для вычислений.

С точки зрения предварительной обработки и постобработки нормализация и квантование являются двумя независимыми шагами. Вот подробности.

 | Нормализация<a>Квантование модели</a> — это метод, который позволяет представлять веса с уменьшенной точностью и, при необходимости, активировать как для хранения, так и для вычислений.<a>Квантование модели</a> — это метод, который позволяет представлять веса с уменьшенной точностью и, при необходимости, активировать как для хранения, так и для вычислений. | Квантование
:-: | --- | ---
1 | 1 | **Поплавковая модель** : \
: Пример: - среднее значение: 127,5 \ : - нулевая точка: 0 \ : |  |
: значения параметров : - станд.: 127,5 \ : - шкала: 1,0 \ : |  |
: входное изображение в : **Квантовая модель** : \ : **Квантовая модель** : \ : |  |
1 |  |
: количественные модели, : - станд.: 127,5 : - масштаб: 0,0078125f \ : |  |
: соответственно. : : : |  |
\ | \ | **Поплавковые модели**
:\ :\ : квантование не требуется. \ : |  |
: \ : **Входы** : Если вход : **Квантовая модель** может : |  |
: \ : данные нормализуются в : или могут не понадобиться : |  |
: Когда вызывать? : обучение, ввод : квантование в pre/post : |  |
: : данные логического вывода: обработка. Это зависит : |  |
: : для нормализации : по типу данных : |  |
: : соответственно. \ : тензоры ввода/вывода. \ : |  |
: : **Выходы** : выход : - плавающие тензоры: нет : |  |
: : данных не будет : квантование в pre/post : |  |
:: нормализовалось в целом. : требуется обработка. Количество: |  |
: : : op и dequant op: |  |
: : : встроено в модель : |  |
: : : граф. \ : |  |
: : : - тензоры int8/uint8: : |  |
: : : нужно квантование в : |  |
: : : пре/постобработка. : |  |
\ | \ | **Квантование для входов** :
: \ : \ : \ : |  |
: Формула : normalized_input = : q = f/масштаб + : |  |
: : (ввод - среднее значение) / std : zeroPoint \ : |  |
: : : **Деквантовать для : |  |
: : : выходы**: \ : |  |
: : : f = (q - нулевая точка) * : |  |
: : : шкала : |  |
\ | Заполнено создателем модели | Заполняется автоматически
: Где находятся : и хранятся в модели : преобразователь TFLite и : |  |
: параметры : метаданные, как : хранятся в модели tflite: |  |
: : `NormalizationOptions` : файл. : |  |
Как получить | Сквозь | Через TFLite
: параметры? : API `MetadataExtractor` : `Tensor` API [1] или : |  |
: : [2] : через : |  |
: : : API `MetadataExtractor` : |  |
: : : [2] : |  |
Делайте плавающие и количественные | Да, float и quant | Нет, плавающая модель
: у моделей одинаковые : у моделей одинаковые : квантование не требуется. : |  |
: ценность? : Нормализация : : |  |
: : параметры : : |  |
Есть ли код TFLite | \ | \
: генератор или Android : Да : Да : |  |
: Привязка Studio ML : : : |  |
: автоматически генерировать : : : |  |
: это в обработке данных? : : : |  |

[1] [TensorFlow Lite Java API](https://github.com/tensorflow/tensorflow/blob/09ec15539eece57b257ce9074918282d88523d56/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Tensor.java#L73) и [TensorFlow Lite C++ API](https://github.com/tensorflow/tensorflow/blob/09ec15539eece57b257ce9074918282d88523d56/tensorflow/lite/c/common.h#L391) .
 [2] Библиотека [извлечения метаданных](#read-the-metadata-from-models)

При обработке данных изображения для моделей uint8 нормализация и квантование иногда пропускаются. Это нормально делать, когда значения пикселей находятся в диапазоне [0, 255]. Но в целом всегда следует обрабатывать данные в соответствии с параметрами нормализации и квантования, когда это применимо.

[Библиотека задач TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/overview) может выполнить нормализацию за вас, если вы настроите `NormalizationOptions` в метаданных. Обработка квантования и деквантования всегда инкапсулирована.

### Примеры

Примечание. Указанный каталог экспорта должен существовать до запуска скрипта; он не создается как часть процесса.

Вы можете найти примеры заполнения метаданных для различных типов моделей здесь:

#### Классификация изображений

Загрузите скрипт [здесь](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py) , который заполняет метаданными [файл mobilenet_v1_0.75_160_quantized.tflite](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/default/1) . Запустите скрипт следующим образом:

```sh
python ./metadata_writer_for_image_classifier.py \
    --model_file=./model_without_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --label_file=./model_without_metadata/labels.txt \
    --export_directory=model_with_metadata
```

Чтобы заполнить метаданные для других моделей классификации изображений, добавьте в скрипт спецификации модели, подобные [этой](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py#L63-L74) . В оставшейся части этого руководства будут выделены некоторые ключевые разделы примера классификации изображений, чтобы проиллюстрировать ключевые элементы.

### Глубокое погружение в пример классификации изображений

#### Информация о модели

Метаданные начинаются с создания новой информации о модели:

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

#### Входная/выходная информация

В этом разделе показано, как описать входную и выходную сигнатуру вашей модели. Эти метаданные могут использоваться автоматическими генераторами кода для создания кода предварительной и последующей обработки. Чтобы создать входную или выходную информацию о тензоре:

```python
# Creates input info.

input_meta = _metadata_fb.TensorMetadataT()

# Creates output info.

output_meta = _metadata_fb.TensorMetadataT()
```

#### Ввод изображения

Изображение является распространенным типом ввода для машинного обучения. Метаданные TensorFlow Lite поддерживают такую информацию, как цветовое пространство, и данные предварительной обработки, такие как нормализация. Размер изображения не требует указания вручную, поскольку он уже задан формой входного тензора и может быть определен автоматически.

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

#### Вывод этикетки

Метка может быть сопоставлена с выходным тензором через связанный файл с помощью `TENSOR_AXIS_LABELS` .

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

#### Создайте метаданные Flatbuffers

Следующий код объединяет информацию о модели с входной и выходной информацией:

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

#### Упакуйте метаданные и связанные файлы в модель

После создания метаданных Flatbuffers метаданные и файл метки записываются в файл TFLite с помощью метода `populate` :

```python
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["your_path_to_label_file"])
populator.populate()
```

Вы можете упаковать в модель столько связанных файлов, сколько хотите, с помощью `load_associated_files` . Однако требуется упаковать хотя бы те файлы, которые задокументированы в метаданных. В этом примере упаковка файла этикетки обязательна.

## Визуализируйте метаданные

Вы можете использовать [Netron](https://github.com/lutzroeder/netron) для визуализации ваших метаданных или прочитать метаданные из модели TensorFlow Lite в формат json с помощью `MetadataDisplayer` :

```python
displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
export_json_file = os.path.join(FLAGS.export_directory,
                    os.path.splitext(model_basename)[0] + ".json")
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file

with open(export_json_file, "w") as f:
  f.write(json_file)
```

Android Studio также поддерживает отображение метаданных с помощью функции [Android Studio ML Binding](https://developer.android.com/studio/preview/features#tensor-flow-lite-models) .

## Управление версиями метаданных

Схема [метаданных](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) версионируется как по семантическому номеру версии, который отслеживает изменения файла схемы, так и по идентификации файла Flatbuffers, что указывает на истинную совместимость версий.

### Семантический номер версии

Схема метаданных управляется [номером семантической версии](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L53) , например MAJOR.MINOR.PATCH. Он отслеживает изменения схемы в соответствии с правилами [здесь](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L32-L44) . Смотрите [историю полей,](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L63) добавленных после версии `1.0.0` .

### Идентификация файла Flatbuffers

Семантическое управление версиями гарантирует совместимость при соблюдении правил, но не означает истинной несовместимости. Повышение ОСНОВНОГО номера не обязательно означает, что обратная совместимость нарушена. Поэтому мы используем [идентификацию файла](https://google.github.io/flatbuffers/md__schemas.html) Flatbuffers, [file_identifier](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L61) , для обозначения истинной совместимости схемы метаданных. Идентификатор файла состоит ровно из 4 символов. Он привязан к определенной схеме метаданных и не подлежит изменению пользователями. Если по какой-либо причине обратная совместимость схемы метаданных должна быть нарушена, идентификатор файла поднимется, например, с «М001» до «М002». Ожидается, что File_identifier будет изменяться гораздо реже, чем metadata_version.

### Минимально необходимая версия парсера метаданных

[Минимальная необходимая версия анализатора метаданных](https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L681) — это минимальная версия анализатора метаданных (код, сгенерированный Flatbuffers), который может полностью считывать Flatbuffers метаданных. Версия фактически представляет собой самый большой номер версии среди версий всех заполненных полей и наименьшую совместимую версию, указанную идентификатором файла. Минимально необходимая версия парсера метаданных автоматически заполняется `MetadataPopulator` при заполнении метаданных в модели TFLite. См. [экстрактор метаданных](#read-the-metadata-from-models) для получения дополнительной информации о том, как используется минимально необходимая версия анализатора метаданных.

## Чтение метаданных из моделей

Библиотека Metadata Extractor — удобный инструмент для чтения метаданных и связанных файлов из моделей на разных платформах (см. версию для [Java и версию](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/java) для [C++](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/cc) ). Вы можете создать свой собственный инструмент для извлечения метаданных на других языках, используя библиотеку Flatbuffers.

### Чтение метаданных в Java

Чтобы использовать библиотеку извлечения метаданных в вашем приложении для Android, мы рекомендуем использовать [AAR метаданных TensorFlow Lite, размещенный на MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-metadata) . Он содержит класс `MetadataExtractor` , а также Java-привязки FlatBuffers для [схемы метаданных и схемы](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs) [модели](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs) .

Вы можете указать это в своих зависимостях `build.gradle` следующим образом:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.1.0'
}
```

Чтобы использовать ночные снимки, убедитесь, что вы добавили [репозиторий снимков Sonatype](https://www.tensorflow.org/lite/android/lite_build#use_nightly_snapshots) .

Вы можете инициализировать объект `MetadataExtractor` с помощью `ByteBuffer` , который указывает на модель:

```java
public MetadataExtractor(ByteBuffer buffer);
```

`ByteBuffer` должен оставаться неизменным в течение всего времени существования объекта `MetadataExtractor` . Инициализация может завершиться ошибкой, если идентификатор файла Flatbuffers метаданных модели не совпадает с идентификатором анализатора метаданных. Дополнительные сведения см. в разделе [Управление версиями метаданных](#metadata-versioning) .

С совпадающими идентификаторами файлов средство извлечения метаданных будет успешно считывать метаданные, сгенерированные из всех прошлых и будущих схем благодаря механизму прямой и обратной совместимости Flatbuffers. Однако поля из будущих схем не могут быть извлечены более старыми экстракторами метаданных. [Минимально необходимая версия анализатора](#the-minimum-necessary-metadata-parser-version) метаданных указывает минимальную версию анализатора метаданных, которая может полностью считывать Flatbuffers метаданных. Вы можете использовать следующий метод, чтобы проверить, выполняется ли условие минимальной необходимой версии анализатора:

```java
public final boolean isMinimumParserVersionSatisfied();
```

Допускается передача модели без метаданных. Однако вызов методов, считывающих метаданные, вызовет ошибки времени выполнения. Вы можете проверить, есть ли у модели метаданные, вызвав метод `hasMetadata` :

```java
public boolean hasMetadata();
```

`MetadataExtractor` предоставляет удобные функции для получения метаданных тензоров ввода/вывода. Например,

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

Хотя [схема модели TensorFlow Lite](https://github.com/tensorflow/tensorflow/blob/aa7ff6aa28977826e7acae379e82da22482b2bf2/tensorflow/lite/schema/schema.fbs#L1075) поддерживает несколько подграфов, интерпретатор TFLite в настоящее время поддерживает только один подграф. Поэтому `MetadataExtractor` опускает индекс подграфа в качестве входного аргумента в своих методах.

## Чтение связанных файлов из моделей

Модель TensorFlow Lite с метаданными и связанными файлами по сути представляет собой zip-файл, который можно распаковать с помощью обычных инструментов zip, чтобы получить связанные файлы. Например, вы можете разархивировать [mobilenet_v1_0.75_160_quantized](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/metadata/1) и извлечь файл метки в модели следующим образом:

```sh
$ unzip mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
Archive:  mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
 extracting: labels.txt
```

Вы также можете прочитать связанные файлы с помощью библиотеки извлечения метаданных.

В Java передайте имя файла в метод `MetadataExtractor.getAssociatedFile` :

```java
public InputStream getAssociatedFile(String fileName);
```

Точно так же в C++ это можно сделать с помощью метода `ModelMetadataExtractor::GetAssociatedFile` :

```c++
tflite::support::StatusOr<absl::string_view> GetAssociatedFile(
      const std::string& filename) const;
```
