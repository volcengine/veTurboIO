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
#include "include/posix.h"
#include "include/logging.h"
#include "include/cipher.h"
#include "include/fastcrypto.h"
#include <errno.h>

POSIXFile::POSIXFile(std::string file_path)
{
    this->file_path = file_path;
}

POSIXFile::POSIXFile(std::string file_path, CipherInfo cipher_info)
{
    this->file_path = file_path;
    this->cipher_info = cipher_info;
}

POSIXFile::POSIXFile(std::string file_path, bool use_cipher, pybind11::array_t<char> key_arr,
                     pybind11::array_t<char> iv_arr, size_t header_size)
    : POSIXFile(file_path)
{
    this->cipher_info = CipherInfo(use_cipher, key_arr, iv_arr, header_size);
}

void POSIXFile::read_file_to_address_thread(int thread_id, char *addr, int device_id, char *dev_mem, size_t block_size,
                                            size_t total_size, size_t global_offset, bool use_direct_io,
                                            CipherInfo cipher_info)
{
    size_t offset = thread_id * block_size;
    size_t read_size = block_size;
    int fd = -1;
    int ret = 0;
    size_t size_read = 0;

    if (offset + read_size >= total_size)
    {
        read_size = (total_size > offset) ? total_size - offset : 0;
    }
    // TODO: use_direct_io if sfcs file detected
    if (use_direct_io)
    {
        if ((fd = open(file_path.c_str(), O_RDONLY | O_DIRECT)) < 0)
        {
            if (errno == EINVAL)
            {
                logWarn("open file using directIO failed, fall back to bufferIO", file_path.c_str(),
                        std::strerror(EINVAL));
            }
            else
            {
                logError("open file using directIO failed", file_path.c_str(), std::strerror(errno));
                throw std::runtime_error("veTurboIO Exception: can't apply open operation");
            }
        }
    }

    if (fd == -1)
    {
        if ((fd = open(file_path.c_str(), O_RDONLY)) < 0)
        {
            logError("open file using bufferIO failed", file_path.c_str(), std::strerror(errno));
            throw std::runtime_error("veTurboIO Exception: can't apply open operation");
        }
    }

    FILE *fp = fdopen(fd, "rb");
    if (fp == NULL)
    {
        logError("can't apply fdopen to file", file_path.c_str(), std::strerror(errno));
        throw std::runtime_error("veTurboIO Exception: can't apply fdopen operation");
    }

    if ((ret = fseek(fp, global_offset + offset, SEEK_SET)) < 0)
    {
        logError("can't apply fseek to file", file_path.c_str(), std::strerror(errno));
        throw std::runtime_error("veTurboIO Exception: can't apply fseek operation");
    }

    if ((size_read = fread(addr + offset, 1, read_size, fp)) == 0)
    {
        logWarn("read file with 0 bytes returned", file_path.c_str(), offset, read_size);
    }

    if ((ret = fclose(fp)) < 0)
    {
        logError("can't apply fclose to file", file_path.c_str(), std::strerror(errno));
        throw std::runtime_error("veTurboIO Exception: can't apply fclose operation");
    }

    // Decrypt if use_cipher is true
    if (cipher_info.use_cipher)
    {
        CtrDecrypter dec(cipher_info.mode, cipher_info.key, cipher_info.iv,
                         global_offset + offset - cipher_info.header_size);
        unsigned char *ct = reinterpret_cast<unsigned char *>(addr + offset);
        int cipher_ret = dec.decrypt_update(ct, read_size, ct);
        if (!cipher_ret)
        {
            throw std::runtime_error("Cipher Exception: decrypt fail");
        }
    }

#if defined(USE_CUDA)
    if (dev_mem != NULL && device_id >= 0)
    {
        cudaSetDevice(device_id);
        cudaMemcpyAsync(dev_mem + offset, addr + offset, read_size, cudaMemcpyHostToDevice);
    }
#elif defined(USE_NPU)
#else
#endif
}

size_t POSIXFile::read_file_to_address_parallel(char *addr, int device_id, char *dev_mem, int num_thread,
                                                size_t total_size, size_t global_offset, bool use_direct_io)
{
    vector<thread> threads(num_thread);

    size_t block_size = (size_t)ceil((double)total_size / num_thread);
    // align the block_size;
    block_size = (block_size + BUF_ALIGN_SIZE - 1) / BUF_ALIGN_SIZE * BUF_ALIGN_SIZE;
    // re-caculate the real needed thread num;
    num_thread = (total_size + block_size - 1) / block_size;

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id] = std::thread(&POSIXFile::read_file_to_address_thread, this, thread_id, addr, device_id,
                                         dev_mem, block_size, total_size, global_offset, use_direct_io, cipher_info);
    }

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id].join();
    }

    return total_size;
}

size_t POSIXFile::read_file_to_array(pybind11::array_t<char> arr, size_t length, size_t offset, int num_thread,
                                     bool use_direct_io)
{
    pybind11::buffer_info buf_info = arr.request();
    char *addr = static_cast<char *>(buf_info.ptr);
    madvise(addr, length, MADV_HUGEPAGE);
    return read_file_to_address_parallel(addr, -1, NULL, num_thread, length, offset, use_direct_io);
}

size_t POSIXFile::write_file_from_addr(char *addr, size_t length, bool append)
{
    int fd;
    int flags = O_WRONLY;
    size_t ret;
    size_t count;
    char *src = addr;
    size_t offset = 0;

    if (append)
    {
        struct stat st;
        stat(file_path.c_str(), &st);
        offset = st.st_size;
        flags |= O_APPEND;
    }

    if (cipher_info.use_cipher)
    {
        size_t h_off = cipher_info.header_size;
        CtrEncrypter enc(cipher_info.mode, cipher_info.key, cipher_info.iv, offset - h_off);
        unsigned char *pt = reinterpret_cast<unsigned char *>(addr);
        int cipher_ret = enc.encrypt_update(pt, length, pt);
        if (!cipher_ret)
        {
            throw std::runtime_error("Cipher Exception: encrypt fail");
        }
    }

    fd = open(file_path.c_str(), flags);
    if (fd < 0)
    {
        logError("open failed", file_path.c_str(), std::strerror(errno));
        throw std::runtime_error("veTurboIO Exception: open failed");
    }

    count = length;
    while (count > 0)
    {
        ret = write(fd, src, count);
        if (ret < 0)
        {
            logError("Failed to write file", file_path.c_str());
            throw std::runtime_error("veTurboIO Exception: write file");
        }
        count -= ret;
        src += ret;
    }
    close(fd);
    return length;
}
