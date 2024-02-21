/*
 * Copyright (c) 2024 Beijing Volcano Engine Technology Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef IO_HELPER_H
#define IO_HELPER_H

#include "load_utils.h"

class IOHelper
{
  private:
    char *pin_mem = NULL;
    bool use_pinmem_ = false;
    size_t buffer_size_ = 0;

  public:
    ~IOHelper();
    void load_file_to_tensor(std::string file_path, torch::Tensor res_tensor, torch::Tensor sample_tensor,
                             int64_t offset, int64_t device_id, int64_t num_thread, bool use_pinmem, bool use_sfcs_sdk,
                             bool use_direct_io, bool use_cipher, pybind11::array_t<char> key_arr,
                             pybind11::array_t<char> iv_arr);
    void init_buffer(string file_path, int64_t file_size, bool use_pinmem, bool use_sfcs_sdk);
    void free_buffer();
};

#endif