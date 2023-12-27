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
#ifndef VETURBOIO_CIPHER_H
#define VETURBOIO_CIPHER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include "fastcrypto.h"

class CipherInfo
{
  public:
    bool use_cipher = false;
    std::string mode = "CTR-128";
    size_t header_size = 0;
    unsigned char *key = NULL;
    unsigned char *iv = NULL;
    CipherInfo(bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr, size_t header_size);
    CipherInfo() = default;
};

class CtrEncWrap
{
  private:
    std::unique_ptr<CtrEncrypter> enc_;

  public:
    CtrEncWrap() = default;
    CtrEncWrap(std::string mode, pybind11::array_t<unsigned char> key_arr, pybind11::array_t<unsigned char> iv_arr,
               size_t global_offset);
    size_t encrypt_update(pybind11::array_t<unsigned char> pt, pybind11::array_t<unsigned char> ct);
};

class CtrDecWrap
{
  private:
    std::unique_ptr<CtrDecrypter> dec_;

  public:
    CtrDecWrap() = default;
    CtrDecWrap(std::string mode, pybind11::array_t<unsigned char> key_arr, pybind11::array_t<unsigned char> iv_arr,
               size_t global_offset);
    size_t decrypt_update(pybind11::array_t<unsigned char> ct, pybind11::array_t<unsigned char> pt);
};

#endif
