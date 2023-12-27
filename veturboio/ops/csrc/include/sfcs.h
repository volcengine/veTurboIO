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
#include "cipher.h"

#define SFCS_NAME_NODE "default"
#define SFCS_USER_NAME "demo-user"

using namespace std;

class SFCSFs
{
  public:
    cfsFS fs;

    SFCSFs();
    ~SFCSFs();
    void concat_files(std::string file_name, vector<const char *> file_paths);
    void rename_file(const char *file_path, const char *file_name);
    void mkdir(std::string file_path);
    int64_t get_block_size();
    size_t read_file_to_addr(std::string file_name, CipherInfo cipher_info, char *addr, size_t length, size_t offset);
    size_t write_file_from_addr(std::string file_name, CipherInfo cipher_info, char *addr, size_t length,
                                size_t offset);
    void read_multi_files(pybind11::list file_paths, pybind11::list tensors, pybind11::list lengths,
                          pybind11::list offsets, int num_thread, bool use_cipher, pybind11::array_t<char> key_arr,
                          pybind11::array_t<char> iv_arr, size_t header_size);
    void write_multi_files(pybind11::list file_paths, pybind11::list tensors, pybind11::list lengths,
                           pybind11::list offsets, int num_thread, bool use_cipher, pybind11::array_t<char> key_arr,
                           pybind11::array_t<char> iv_arr, size_t header_size);
    void get_file_size(std::string file_name, size_t *size);
    void get_multi_file_size(pybind11::list file_paths, pybind11::list sizes, int num_thread);
};

class SFCSFile
{
  public:
    cfsFS fs;
    bool fs_owner;
    SFCSFs *sfcs_fs;
    std::string file_path;
    // cipher related
    CipherInfo cipher_info;

    SFCSFile(std::string file_path);
    SFCSFile(std::string path, SFCSFs *sfcs_fs);
    SFCSFile(std::string file_path, bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr,
             size_t header_size);
    SFCSFile(std::string file_path, CipherInfo cipher_info);
    SFCSFile(std::string file_path, SFCSFs *sfcs_fs, CipherInfo cipher_info);
    ~SFCSFile();
    size_t get_file_size();
    size_t read_file_to_address_parallel(char *addr, int device_id, char *dev_mem, int num_thread, size_t total_size,
                                         size_t global_offset);
    size_t read_file_to_addr(char *addr, size_t length, size_t offset);
    size_t read_file_to_array(pybind11::array_t<char> arr, size_t length, size_t offset, int num_thread);
    size_t write_file_from_array(pybind11::array_t<char> arr, size_t length, bool append);
    size_t write_file_from_tensors(pybind11::list tensors, pybind11::list sizes, pybind11::list offsets,
                                   std::string concat_dir, std::string concat_file);
    size_t write_file_from_addr(char *addr, size_t length, size_t offset, bool append);
    void delete_file();

  private:
    void read_file_to_address_thread(int thread_id, char *addr, int device_id, char *dev_mem, size_t block_size,
                                     size_t total_size, size_t global_offset);
    void write_file_from_tensor(torch::Tensor tensor, size_t length, size_t offset, std::string file_name);
};

#endif