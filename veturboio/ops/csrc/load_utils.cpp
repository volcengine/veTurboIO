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
#include "include/load_utils.h"
#include "include/cipher.h"
#include "include/fastcrypto.h"

void read_file_thread_fread(int thread_id, string file_path, char *addr, char *dev_mem, size_t block_size,
                            size_t total_size, size_t global_offset, bool use_direct_io, CipherInfo cipher_info)
{
    size_t offset = thread_id * block_size;
    size_t read_size = block_size;
    int fd;
    if (offset + read_size >= total_size)
    {
        read_size = (total_size > offset) ? total_size - offset : 0;
    }
    // TODO: use_direct_io if sfcs file detected
    if (use_direct_io)
    {
        fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    }
    else
    {
        fd = open(file_path.c_str(), O_RDONLY);
    }
    FILE *fp = fdopen(fd, "rb");
    fseek(fp, global_offset + offset, SEEK_SET);
    fread(addr + offset, 1, read_size, fp);
    fclose(fp);

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

    if (dev_mem != NULL)
        cudaMemcpyAsync(dev_mem + offset, addr + offset, read_size, cudaMemcpyHostToDevice);
}

void read_file(string file_path, char *addr, char *dev_mem, int num_thread, size_t total_size, size_t global_offset,
               bool use_sfcs_sdk, bool use_direct_io, CipherInfo cipher_info)
{
    if (total_size == 0)
    {
        return;
    }

    vector<thread> threads(num_thread);

    size_t block_size = (size_t)ceil((double)total_size / num_thread);
    // align the block_size;
    block_size = (block_size + BUF_ALIGN_SIZE - 1) / BUF_ALIGN_SIZE * BUF_ALIGN_SIZE;
    // re-caculate the real needed thread num;
    num_thread = (total_size + block_size - 1) / block_size;

    if (use_sfcs_sdk)
    {
        SFCSFile sfcs_file(file_path, cipher_info);
        sfcs_file.read_file_parallel(addr, dev_mem, num_thread, total_size, global_offset);
    }
    else
    {
        for (int thread_id = 0; thread_id < num_thread; thread_id++)
        {
            threads[thread_id] = std::thread(read_file_thread_fread, thread_id, file_path, addr, dev_mem, block_size,
                                             total_size, global_offset, use_direct_io, cipher_info);
        }

        for (int thread_id = 0; thread_id < num_thread; thread_id++)
        {
            threads[thread_id].join();
        }
    }
}

size_t get_file_size(const char *file_name, bool use_sfcs_sdk)
{
    if (use_sfcs_sdk)
    {
        SFCSFile sfcs_file(file_name);
        return sfcs_file.get_file_size();
    }
    else
    {
        struct stat st;
        stat(file_name, &st);
        return st.st_size;
    }
}
