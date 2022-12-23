// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   tracer.h
 * @date   23 December 2022
 * @brief  Trace abstract class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TRACER_H__
#define __TRACER_H__

#include <exception>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>

namespace nntrainer {

class Tracer {
public:
  Tracer(const std::string &name) : name(name) {}
  virtual ~Tracer() = default;

  virtual Tracer &traceStart(const std::string &tag, const std::string &msg) = 0;
  virtual Tracer &traceEnd(const std::string &tag) = 0;
  virtual Tracer &tracePoint(const std::string &msg) = 0;

protected:
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
    write(std::ofstream &out, std::tuple<Tp...>& t)
    {}

  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
    write(std::ofstream &out, std::tuple<Tp...>& t) {
      out << std::get<I>(t) << "\t";
      write<I + 1, Tp...>(out, t);
    }

  template <typename T>
  void writeToFile(
    std::string filename,
    std::list<T> &trace_info) {
    std::ofstream file(filename, std::fstream::app);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    for (auto &i : trace_info) {
      write(file, i);
      file << std::endl;
    }

    trace_info.clear();
  }

  const std::string name;
};

class MemoryTracer : public Tracer {
public:
  virtual ~MemoryTracer();

  static std::unique_ptr<MemoryTracer> &getInstance();

  Tracer &traceStart(const std::string &tag, const std::string &msg) override {return (*this);}
  Tracer &traceEnd(const std::string &tag) override {return (*this);}
  Tracer &tracePoint(const std::string &msg) override;

  Tracer &operator<< (const std::string &msg) {return tracePoint(msg);}

private:
  MemoryTracer(const std::string &name, bool flush = false);

  std::list<std::tuple<unsigned long, std::string>>
    trace_info; /**< memory usage, msg */
  bool flush;
};

class TimeTracer : public Tracer {
public:
  virtual ~TimeTracer();

  static std::unique_ptr<TimeTracer> &getInstance();

  Tracer &traceStart(const std::string &tag, const std::string &msg) override {return (*this);}
  Tracer &traceEnd(const std::string &tag) override {return (*this);}
  Tracer &tracePoint(const std::string &msg) override;

  Tracer &operator<< (const std::string &msg) {return tracePoint(msg);}

private:
  TimeTracer(const std::string &name, bool flush = false);

  std::list<std::tuple<unsigned long, std::string>>
    trace_info; /**< time point (ms), msg */
  bool flush;
};

} // namespace nntrainer

#ifndef TRACE

#define TRACE_MEMORY_POINT(msg)
#define TRACE_MEMORY()
#define TRACE_TIME_POINT(msg)
#define TRACE_TIME()

#else

#define TRACE_MEMORY_POINT(msg) \
  nntrainer::MemoryTracer::getInstance()->tracePoint(msg)
#define TRACE_MEMORY() \
  *(nntrainer::MemoryTracer::getInstance())
#define TRACE_TIME_POINT(msg) \
  nntrainer::TimeTracer::getInstance()->tracePoint(msg)
#define TRACE_TIME() \
  *(nntrainer::TimeTracer::getInstance())

#endif

#endif /* __TRACER_H__ */
