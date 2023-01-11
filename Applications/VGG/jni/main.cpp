// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main.cpp
 * @date   07 December 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Simple Linear Example with
 *
 *
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <cifar_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

unsigned int DATA_SIZE;
unsigned int BATCH_SIZE;
unsigned int INPUT_SHAPE[3];
unsigned int OUTPUT_SHAPE[3];
float training_loss = 0.0;

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

std::array<UserDataType, 1> createFakeDataGenerator(unsigned int batch_size) {

  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]}},
    {{batch_size, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]}},
    DATA_SIZE));

  return {std::move(train_data)};
}

int main(int argc, char *argv[]) {
  int status = 0;

  if (argc != 10) {
    std::cerr << "Usage: " << argv[0] << " config.ini input_shape[0] "
              << "input_shape[1] input_shape[2] output_shape[0] "
              << "output_shape[1] output_shape[2] data_size batch_size"
              << std::endl;
    return 1;
  }

  auto config = argv[1];
  INPUT_SHAPE[0] = atoi(argv[2]);
  INPUT_SHAPE[1] = atoi(argv[3]);
  INPUT_SHAPE[2] = atoi(argv[4]);
  OUTPUT_SHAPE[0] = atoi(argv[5]);
  OUTPUT_SHAPE[1] = atoi(argv[6]);
  OUTPUT_SHAPE[2] = atoi(argv[7]);
  DATA_SIZE = atoi(argv[8]);
  BATCH_SIZE = atoi(argv[9]);

  std::array<UserDataType, 1> user_datas;
  user_datas = createFakeDataGenerator(DATA_SIZE);
  auto &[train_user_data] = user_datas;

  std::unique_ptr<ml::train::Model> model;

  model = createModel(ml::train::ModelType::NEURAL_NET);
  model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI);

  model->compile();
  model->initialize();

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));

  model->train();

  return status;
}
