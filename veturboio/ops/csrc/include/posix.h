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
#ifndef LOAD_UTILS_H
#define LOAD_UTILS_H

#include "common.h"
#include "cipher.h"

class POSIXFile
{
  public:
    std::string file_path;
    // cipher related
    CipherInfo cipher_info;

    POSIXFile(std::string file_path);
    POSIXFile(std::string file_path, CipherInfo cipher_info);
    POSIXFile(std::string file_path, bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr,
              size_t header_size);

    size_t read_file_to_address_parallel(char *addr, int device_id, char *dev_mem, int num_thread, size_t total_size,
                                         size_t global_offset, bool use_direct_io);
    size_t read_file_to_array(pybind11::array_t<char> arr, size_t length, size_t offset, int num_thread,
                              bool use_direct_io);
    size_t write_file_from_addr(char *addr, size_t length, bool append);

  private:
    void read_file_to_address_thread(int thread_id, char *addr, int device_id, char *dev_mem, size_t block_size,
                                     size_t total_size, size_t global_offset, bool use_direct_io,
                                     CipherInfo cipher_info);
};

#endif