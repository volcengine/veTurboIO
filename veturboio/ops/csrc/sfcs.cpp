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
#include "include/sfcs.h"
#include "include/fastcrypto.h"

SFCSFile::SFCSFile(std::string path)
{
    file_path = path;

    // construct builder
    struct cfsBuilder *bld = cfsNewBuilder();
    if (bld == NULL)
    {
        logError("Failed to construct bld", cfsGetLastError());
        throw std::runtime_error("SFCS Exception: construct bld");
    }

    cfsBuilderSetNameNode(bld, SFCS_NAME_NODE);
    cfsBuilderSetUserName(bld, SFCS_USER_NAME);

    // connect to cfs
    fs = cfsBuilderConnect(bld, NULL);
    if (fs == NULL)
    {
        logError("Failed to connect to cfs", cfsGetLastError());
        cfsFreeBuilder(bld);
        throw std::runtime_error("SFCS Exception: connect to cfs");
    }
}

SFCSFile::SFCSFile(std::string file_path, CipherInfo cipher_info) : SFCSFile(file_path)
{
    this->cipher_info = cipher_info;
}

SFCSFile::SFCSFile(std::string file_path, bool use_cipher, pybind11::array_t<char> key_arr,
                   pybind11::array_t<char> iv_arr)
    : SFCSFile(file_path)
{
    this->cipher_info = CipherInfo(use_cipher, key_arr, iv_arr);
}

SFCSFile::~SFCSFile()
{
    cfsDisconnect(fs);
}

size_t SFCSFile::get_file_size()
{
    size_t size;
    // get path info
    cfsFileInfo *file_info = cfsGetPathInfo(fs, file_path.c_str());
    if (file_info == NULL)
    {
        logError("Failed to get path info of relative path", file_path, cfsGetLastError());
        cfsDisconnect(fs);
        throw std::runtime_error("SFCS Exception: get path info");
    }
    size = file_info->mSize;
    cfsFreeFileInfo(file_info, 1);
    return size;
}

size_t SFCSFile::read_file(char *addr, size_t length, size_t offset)
{
    size_t count;
    int32_t ret;
    char *dst;

    cfsFile file = cfsOpenFile(fs, file_path.c_str(), O_RDONLY, 0, 0, 0);
    if (file == NULL)
    {
        logError("Failed to open file", file_path, cfsGetLastError());
        throw std::runtime_error("SFCS Exception: open file");
    }

    ret = cfsSeek(fs, file, offset);
    if (ret == -1)
    {
        logError("Failed to seek file", file_path, cfsGetLastError());
        cfsCloseFile(fs, file);
        throw std::runtime_error("SFCS Exception: seek file");
    }

    dst = addr;
    count = length;
    while (count > 0)
    {
        ret = cfsRead(fs, file, dst, count);
        // EOF
        if (ret == 0)
            break;

        if (ret < 0)
        {
            logError("Failed to read file", file_path, cfsGetLastError());
            throw std::runtime_error("SFCS Exception: read file");
        }
        count -= ret;
        dst += ret;
    }

    cfsCloseFile(fs, file);

    // Decrypt if use_cipher is true
    if (cipher_info.use_cipher)
    {
        CtrDecrypter dec(cipher_info.key, cipher_info.iv, offset);
        unsigned char *ct = reinterpret_cast<unsigned char *>(addr);
        int cipher_ret = dec.decrypt_update(ct, length - count, ct);
        if (!cipher_ret)
        {
            throw std::runtime_error("Cipher Exception: decrypt fail");
        }
    }

    return length - count;
}

void SFCSFile::read_file_thread(int thread_id, char *addr, char *dev_mem, size_t block_size, size_t total_size,
                                size_t global_offset)
{
    size_t offset = thread_id * block_size;
    size_t read_size = block_size;

    if (offset + read_size >= total_size)
    {
        read_size = (total_size > offset) ? total_size - offset : 0;
    }

    read_file(addr + offset, read_size, global_offset + offset);

    if (dev_mem != NULL)
        cudaMemcpyAsync(dev_mem + offset, addr + offset, read_size, cudaMemcpyHostToDevice);
}

size_t SFCSFile::read_file_parallel(char *addr, char *dev_mem, int num_thread, size_t total_size, size_t global_offset)
{
    vector<thread> threads(num_thread);

    if (total_size == 0)
    {
        return total_size;
    }

    size_t block_size = (size_t)ceil((double)total_size / num_thread);
    // align the block_size;
    block_size = (block_size + BUF_ALIGN_SIZE - 1) / BUF_ALIGN_SIZE * BUF_ALIGN_SIZE;
    // re-caculate the real needed thread num;
    num_thread = (total_size + block_size - 1) / block_size;

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id] = std::thread(&SFCSFile::read_file_thread, this, thread_id, addr, dev_mem, block_size,
                                         total_size, global_offset);
    }

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id].join();
    }

    return total_size;
}

size_t SFCSFile::read_file_to_array(pybind11::array_t<char> arr, size_t length, size_t offset, int num_thread)
{
    pybind11::buffer_info buf_info = arr.request();
    char *addr = static_cast<char *>(buf_info.ptr);
    return read_file_parallel(addr, NULL, num_thread, length, offset);
}

size_t SFCSFile::write_file(char *addr, size_t length)
{
    size_t count;
    int32_t ret;
    char *dst;

    if (cipher_info.use_cipher)
    {
        CtrEncrypter enc(cipher_info.key, cipher_info.iv, 0);
        unsigned char *pt = reinterpret_cast<unsigned char *>(addr);
        int cipher_ret = enc.encrypt_update(pt, length, pt);
        if (!cipher_ret)
        {
            throw std::runtime_error("Cipher Exception: encrypt fail");
        }
    }

    cfsFile file = cfsOpenFile(fs, file_path.c_str(), O_WRONLY | O_ASYNC, 0, 0, 0);
    if (file == NULL)
    {
        logError("Failed to open file", file_path, cfsGetLastError());
        throw std::runtime_error("SFCS Exception: open file");
    }

    dst = addr;
    count = length;
    while (count > 0)
    {
        ret = cfsWrite(fs, file, dst, count);
        // EOF
        if (ret == 0)
            break;

        if (ret < 0)
        {
            logError("Failed to write file", file_path, cfsGetLastError());
            throw std::runtime_error("SFCS Exception: write file");
        }
        count -= ret;
        dst += ret;
    }

    cfsCloseFile(fs, file);
    return length - count;
}

size_t SFCSFile::write_file_from_array(pybind11::array_t<char> arr, size_t length)
{
    pybind11::buffer_info buf_info = arr.request();
    char *addr = static_cast<char *>(buf_info.ptr);
    return write_file(addr, length);
}

void SFCSFile::delete_file()
{
    int ret;

    ret = cfsDelete(fs, file_path.c_str(), 1);
    if (ret == -1)
    {
        logError("Failed to delete file", file_path, cfsGetLastError());
        throw std::runtime_error("SFCS Exception: delete file");
    }
}

CipherInfo::CipherInfo(bool use_cipher, pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr)
{
    this->use_cipher = use_cipher;
    if (use_cipher)
    {
        pybind11::buffer_info key_info = key_arr.request();
        if ((size_t)key_info.size != CTR_BLOCK_SIZE)
        {
            throw std::runtime_error("Cipher Exception: key length invalid");
        }
        key = reinterpret_cast<unsigned char *>(key_info.ptr);
        pybind11::buffer_info iv_info = iv_arr.request();
        if ((size_t)iv_info.size != CTR_BLOCK_SIZE)
        {
            throw std::runtime_error("Cipher Exception: iv length invalid");
        }
        iv = reinterpret_cast<unsigned char *>(iv_info.ptr);
    }
}