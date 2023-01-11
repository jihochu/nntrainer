#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <cifar_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

/** cache loss values post training for test */
float training_loss = 0.0;

template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

std::vector<LayerHandle> resnetBlock(const std::string &block_name,
                                     const std::string &input_name, int filters,
                                     int kernel_size, bool downsample) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto create_conv = [&with_name, filters](const std::string &name,
                                           int kernel_size, int stride,
                                           const std::string &padding,
                                           const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  /** residual path */
  LayerHandle a1 = create_conv("a1", 3, downsample ? 2 : 1, "same", input_name);
  LayerHandle a2 = createLayer(
    "batch_normalization", {with_name("a2"), withKey("activation", "relu")});
  LayerHandle a3 = create_conv("a3", 3, 1, "same", scoped_name("a2"));

  /** skip path */
  LayerHandle b1 = nullptr;
  if (downsample) {
    b1 = create_conv("b1", 1, 2, "same", input_name);
  }

  const std::string skip_name = b1 ? scoped_name("b1") : input_name;

  LayerHandle c1 = createLayer(
    "Addition",
    {with_name("c1"), withKey("input_layers", {scoped_name("a3"), skip_name})});

  LayerHandle c2 =
    createLayer("batch_normalization",
                {withKey("name", block_name), withKey("activation", "relu")});

  if (downsample) {
    return {b1, a1, a2, a3, c1, c2};
  } else {
    return {a1, a2, a3, c1, c2};
  }
}

std::vector<LayerHandle> createResnet18Graph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "3:32:32")}));

  layers.push_back(
    createLayer("conv2d", {
                            withKey("name", "conv0"),
                            withKey("filters", 64),
                            withKey("kernel_size", {3, 3}),
                            withKey("stride", {1, 1}),
                            withKey("padding", "same"),
                            withKey("bias_initializer", "zeros"),
                            withKey("weight_initializer", "xavier_uniform"),
                          }));

  layers.push_back(
    createLayer("batch_normalization", {withKey("name", "first_bn_relu"),
                                        withKey("activation", "relu")}));

  std::vector<std::vector<LayerHandle>> blocks;

  blocks.push_back(resnetBlock("conv1_0", "first_bn_relu", 64, 3, false));
  blocks.push_back(resnetBlock("conv1_1", "conv1_0", 64, 3, false));
  blocks.push_back(resnetBlock("conv2_0", "conv1_1", 128, 3, true));
  blocks.push_back(resnetBlock("conv2_1", "conv2_0", 128, 3, false));
  blocks.push_back(resnetBlock("conv3_0", "conv2_1", 256, 3, true));
  blocks.push_back(resnetBlock("conv3_1", "conv3_0", 256, 3, false));
  blocks.push_back(resnetBlock("conv4_0", "conv3_1", 512, 3, true));
  blocks.push_back(resnetBlock("conv4_1", "conv4_0", 512, 3, false));

  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  layers.push_back(createLayer("pooling2d", {withKey("name", "last_p1"),
                                             withKey("pooling", "average"),
                                             withKey("pool_size", {4, 4})}));

  layers.push_back(createLayer("flatten", {withKey("name", "last_f1")}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("unit", 100), withKey("activation", "softmax")}));

  return layers;
}

ModelHandle createResnet18() {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse"),
                                              withKey("memory_swap", "true"),
                                              withKey("memory_swap_lookahead", "0")
                                              });
  for (auto layer : createResnet18Graph()) {
    model->addLayer(layer);
  }

  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data) {
  ModelHandle model = createResnet18();
  model->setProperty(
    {withKey("batch_size", batch_size), withKey("epochs", epochs)});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  model->compile();
  model->initialize();

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));

  model->train();
}

std::array<UserDataType, 1> createFakeDataGenerator(unsigned int batch_size) {

  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}}, batch_size));

  return {std::move(train_data)};
}

int main(int argc, char *argv[]) {
  std::cout << "Resnet18" << std::endl;
  
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " [batch_size] [epoch]" << std::endl;
    return 0;
  }
  unsigned int batch_size = std::stoul(argv[1]);
  unsigned int epoch = std::stoul(argv[2]);

  std::array<UserDataType, 1> user_datas;
  user_datas = createFakeDataGenerator(batch_size);

  auto &[train_user_data] = user_datas;
  createAndRun(epoch, batch_size, train_user_data);
  return 0;
}
