### 利用 libtorch进行pytorch的部署

#####　1. 下载libtorch

首先要去下载libtorch的库, 　然后解压

~~~ｓｈｅｌｌ
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
~~~

##### 2. 在Python端，将训练的文件导出．可以参考如下代码:

~~~python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output)

traced_script_module.save("model.pt")
~~~

##### 3.　建立如下目录:

~~~shell
.
├── build
├── CMakeLists.txt 
├── example-app.cpp
├── libtorch #　下载并解压的libtorch文件
└── model.pt #　刚才导出的模型文件
98 directories, 1164 files
~~~

其中的CMakeLists.txt的内容为:

~~~
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
~~~

example-app.cpp文件则为

~~~python
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::ones({1, 3, 224, 224}));

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    } 
    std::cout << "ok\n";
}
~~~

##### 4. 编译并执行即可

~~~shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
$ make -j8
$ ./example-app ../model.pt 
-0.1832 -0.4897  0.4933 -0.0538 -0.0885
[ Variable[CPUFloatType]{1,5} ]
ok
~~~



参考网址:

[LOADING A PYTORCH MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html#)

[INSTALLING C++ DISTRIBUTIONS OF PYTORCH](https://pytorch.org/cppdocs/installing.html)

