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

#ifndef SFCS_H
#define SFCS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common.h"
#include "cfs.h"
#include "logging.h"

#define SFCS_NAME_NODE "default"
#define SFCS_USER_NAME "demo-user"

using namespace std;

class CipherInfo
{
  public:
    bool use_cipher = false;
    unsigned char *key = NULL;
    unsigned char *iv = NULL;
    CipherInfo(bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr);
    CipherInfo(){};
};

class SFCSFile
{
  public:
    cfsFS fs;
    std::string file_path;
    // cipher related
    CipherInfo cipher_info;

    SFCSFile(std::string file_path);
    SFCSFile(std::string file_path, bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr);
    SFCSFile(std::string file_path, CipherInfo cipher_info);
    ~SFCSFile();
    size_t get_file_size();
    size_t read_file_parallel(char *addr, char *dev_mem, int num_thread, size_t total_size, size_t global_offset);
    size_t read_file_to_array(pybind11::array_t<char> arr, size_t length, size_t offset, int num_thread);
    size_t write_file_from_array(pybind11::array_t<char> arr, size_t length);
    void delete_file();

  private:
    size_t read_file(char *addr, size_t length, size_t offset);
    void read_file_thread(int thread_id, char *addr, char *dev_mem, size_t block_size, size_t total_size,
                          size_t global_offset);
    size_t write_file(char *addr, size_t length);
};

#endif