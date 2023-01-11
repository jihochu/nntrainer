// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cstring>

#if defined(ENABLE_TEST)
#include <gtest/gtest.h>
#endif

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <cifar_dataloader.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

bool gradient_clipping=false;
bool gradient_clipping_case = false;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
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

unsigned int data_size = 200000;
unsigned int iteration = 0;
unsigned int input_size = 784;
unsigned int label_size = 10;
unsigned int batch_size = 4096;
unsigned int num_epochs= 10;
unsigned int fc1_unit = 4096;
unsigned int fc2_unit = 256;
std::string opt_str ="sgd";

void next(float **input, float **label, bool *last) {

  auto fill_input =[](float*input, unsigned int length){
		      for(unsigned int i=0;i<length; ++i){
			*input = 0.001*i;
			input++;
			}
		    };
  
  auto fill_label = [](float*label, unsigned int length){
		      memset(label, 0, length*sizeof(float));
		      *(label+length) = 1;
		      label += length;
		    };
  if(iteration++ == data_size){
    iteration = 0;
    *last = true;
    return;
  }

  *last = false;

  fill_input(*input, input_size);
  fill_label(*label, label_size);

  return;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  next(input, label, last);
  return 0;
}

int main(int argc, char *argv[]) {

  if (gradient_clipping) {
    data_size = 2000;
    input_size = 2000;
    fc1_unit = 2000;
    fc2_unit = 1000;
    label_size = 100;
    batch_size = 64;
    opt_str = "adam";
  }

  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,{withKey("loss", "mse")});

  std::vector<LayerHandle> layers;

  std::string input_str = "1:1:"+std::to_string(input_size);

//  layers.push_back(createLayer("input",{withKey("name", "input0"), withKey("input_shape", input_str)}));
  layers.push_back(createLayer("fully_connected", {withKey("unit", fc1_unit),withKey("input_shape", input_str)}));
  // layers.push_back(createLayer("fully_connected", {withKey("input_layers", "input0"), withKey("unit", 4096), withKey("activation","relu"), withKey("trainable", "false")})); 
  layers.push_back(createLayer("fully_connected", {withKey("unit", fc2_unit)})); 
  layers.push_back(createLayer("fully_connected", {withKey("unit", label_size)}));

  for(auto layer : layers){

    model->addLayer(layer);
  }

  if(gradient_clipping && gradient_clipping_case){
        model->setProperty({withKey("batch_size", batch_size),
 		       withKey("epochs", num_epochs),		      
                        withKey("clip_grad_by_norm", 1.0)});
  } else {
    model->setProperty({withKey("batch_size", batch_size),
			withKey("epochs", num_epochs)});
  }
  
  auto optimizer = ml::train::createOptimizer(opt_str, {"learning_rate=0.001"});
  std::cout << opt_str<< std::endl;
  
  model->setOptimizer(std::move(optimizer));
  std::cout << "opt set"<<std::endl;  
  int status = model->compile();
  std::cout << "compile"<<std::endl;  
  status = model->initialize();
  std::cout << "ini"<<std::endl;  


  auto dataset_train = ml::train::createDataset(
						ml::train::DatasetType::GENERATOR, trainData_cb, nullptr);

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(dataset_train));

  std::cout << "train"<<std::endl;

  model->train();
  
  return status;
}
