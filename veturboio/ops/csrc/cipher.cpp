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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "include/cipher.h"
#include <iostream>

CipherInfo::CipherInfo(bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr,
                       size_t header_size)
    : use_cipher(use_cipher), header_size(header_size)
{
    if (use_cipher)
    {
        pybind11::buffer_info key_info = key_arr.request();
        size_t key_size = key_info.size;
        if (key_size == 16)
        {
            mode = "CTR-128";
        }
        else if (key_size == 32)
        {
            mode = "CTR-256";
        }
        else
        {
            throw std::runtime_error("Cipher Exception: key length invalid");
        }
        key = reinterpret_cast<unsigned char *>(key_info.ptr);

        pybind11::buffer_info iv_info = iv_arr.request();
        if ((size_t)iv_info.size != AES_BLOCK_SIZE)
        {
            throw std::runtime_error("Cipher Exception: iv length invalid");
        }
        iv = reinterpret_cast<unsigned char *>(iv_info.ptr);
    }
}

CtrEncWrap::CtrEncWrap(std::string mode, pybind11::array_t<unsigned char> key_arr,
                       pybind11::array_t<unsigned char> iv_arr, size_t global_offset)
{
    pybind11::buffer_info key_info = key_arr.request();
    pybind11::buffer_info iv_info = iv_arr.request();
    enc_.reset(new CtrEncrypter(mode, (unsigned char *)key_info.ptr, (unsigned char *)iv_info.ptr, global_offset));
}

size_t CtrEncWrap::encrypt_update(pybind11::array_t<unsigned char> pt, pybind11::array_t<unsigned char> ct)
{
    pybind11::buffer_info pt_info = pt.request();
    pybind11::buffer_info ct_info = ct.request();
    unsigned char *pt_ptr = (unsigned char *)pt_info.ptr;
    unsigned char *ct_ptr = (unsigned char *)ct_info.ptr;
    return enc_->encrypt_update(pt_ptr, pt_info.size, ct_ptr);
}

CtrDecWrap::CtrDecWrap(std::string mode, pybind11::array_t<unsigned char> key_arr,
                       pybind11::array_t<unsigned char> iv_arr, size_t global_offset)
{
    pybind11::buffer_info key_info = key_arr.request();
    pybind11::buffer_info iv_info = iv_arr.request();
    dec_.reset(new CtrDecrypter(mode, (unsigned char *)key_info.ptr, (unsigned char *)iv_info.ptr, global_offset));
}

size_t CtrDecWrap::decrypt_update(pybind11::array_t<unsigned char> ct, pybind11::array_t<unsigned char> pt)
{
    pybind11::buffer_info pt_info = pt.request();
    pybind11::buffer_info ct_info = ct.request();
    unsigned char *pt_ptr = (unsigned char *)pt_info.ptr;
    unsigned char *ct_ptr = (unsigned char *)ct_info.ptr;
    return dec_->decrypt_update(ct_ptr, ct_info.size, pt_ptr);
}
