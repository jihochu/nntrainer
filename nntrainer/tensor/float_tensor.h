// SPDX-License-Identifier: Apache-2.0
/**
 * @file	float_tensor.h
 * @date	01 December 2023
 * @brief	This is FloatTensor class for 32-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __FLOAT_TENSOR_H__
#define __FLOAT_TENSOR_H__
#ifdef __cplusplus

#include <tensor_base.h>
#include <tensor_v2.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

namespace nntrainer {

/**
 * @class FloatTensor class
 * @brief FloatTensor class for 32-bit floating point calculation
 */
class FloatTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  FloatTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  FloatTensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  FloatTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d data for the Tensor
   * @param fm format for the Tensor
   */
  FloatTensor(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
    Tformat fm);

  /**
   * @brief Basic Destructor
   */
  ~FloatTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator==(const FloatTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator!=(const FloatTensor &rhs) const { return !(*this == rhs); }

  /**
   * @copydoc TensorV2::allocate()
   */
  void allocate() override;

  /**
   * @copydoc TensorV2::deallocate()
   */
  void deallocate() override;

  /**
   * @copydoc TensorV2::getData()
   */
  void *getData() const override;

  /**
   * @copydoc TensorV2::getData(size_t idx)
   */
  void *getData(size_t idx) const override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  void *getAddress(unsigned int i) override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const void *getAddress(unsigned int i) const override;

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  void setValue(float value) override;

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  void setValue(unsigned int batch, unsigned int c, unsigned int h,
                unsigned int w, float value) override;

  /**
   * @copydoc TensorV2::setZero()
   */
  void setZero() override;

  /**
   * @copydoc TensorV2::initialize()
   */
  void initialize() override;

  /**
   * @copydoc TensorV2::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc TensorV2::print(std::ostream &out)
   */
  void print(std::ostream &out) const override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLOAT_TENSOR_H__ */
