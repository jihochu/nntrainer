// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   loss_layer.h
 * @date   12 June 2020
 * @brief  This is Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LOSS_LAYER_H__
#define __LOSS_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

#include <tensor.h>

namespace nntrainer {

/**
 * @class   LossLayer
 * @brief   loss layer
 */
class LossLayer : public Layer {
public:
  /**
   * @brief     Constructor of Loss Layer
   */
  LossLayer();

  /**
   * @brief     Destructor of Loss Layer
   */
  virtual ~LossLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  virtual void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  virtual bool supportBackwarding() const override { return true; }

  /**
   * @brief Set loss scale factor
   */
  virtual void setLossScale(float scale) override { loss_scale = scale; }

private:
  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  float loss_scale; /**< loss scale factor */

protected:
  /**
   * @brief     update loss
   * @param     context Run context to update loss in
   * @param     l Tensor data to calculate
   */
  void updateLoss(RunLayerContext &context, const Tensor &l);

  /**
   * @brief     apply loss scale
   */
  void applyLossScale(Tensor &derivative) {
    if (loss_scale != 0.0f)
      derivative.multiply_i(loss_scale);
  }

  Tensor
    l; /**< loss tensor to store intermediate value to calculate loss value */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
