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
#include "include/fastcrypto.h"
#include "include/io_helper.h"

IOHelper::~IOHelper()
{
    free_buffer();
}

// init buffer with given positive size or the size of the file in specified
// path
void IOHelper::init_buffer(string file_path, int64_t buffer_size, bool use_pinmem, bool use_sfcs_sdk)
{
    if (buffer_size <= 0)
    {
        buffer_size = get_file_size(file_path.c_str(), use_sfcs_sdk);
    }

    if (buffer_size_ > 0)
    {
        free_buffer();
    }

    buffer_size_ = buffer_size;
    if (use_pinmem)
    {
        use_pinmem_ = true;
        cudaMallocHost(&pin_mem, buffer_size, cudaHostAllocMapped);
    }
    else
    {
        pin_mem = (char *)mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        madvise(pin_mem, buffer_size, MADV_HUGEPAGE);
    }
}

void IOHelper::free_buffer()
{
    if (pin_mem != NULL)
    {
        if (use_pinmem_)
            cudaFreeHost(pin_mem);
        else
            munmap(pin_mem, buffer_size_);
    }
}

void read_unaligned_part(std::string file_path, torch::Tensor res_tensor, int64_t *offset, int64_t device_id,
                         size_t *total_size, bool use_sfcs_sdk, bool use_direct_io, size_t *read_unaligned_size,
                         CipherInfo cipher_info)
{
    // cpu align only read head part, while gpu align read both head and tail part
    if (device_id < 0)
    {
        // head is aligned
        if ((*offset & (BUF_ALIGN_SIZE - 1)) == 0)
        {
            return;
        }
        *read_unaligned_size = min(BUF_ALIGN_SIZE - (*offset & (BUF_ALIGN_SIZE - 1)), *total_size);
        if ((uint64_t)res_tensor.data_ptr() % BUF_ALIGN_SIZE != *offset % BUF_ALIGN_SIZE)
        {
            throw std::runtime_error("data ptr does not satisfy the align purpose");
        }
        read_file(file_path, (char *)res_tensor.data_ptr(), NULL, 1, *read_unaligned_size, *offset, use_sfcs_sdk,
                  use_direct_io, cipher_info);

        *total_size -= *read_unaligned_size;
        *offset += *read_unaligned_size;
    }
    else
    {
        size_t end_offset = *offset + *total_size;
        // both head and tail are aligned
        if ((*offset & (BUF_ALIGN_SIZE - 1)) == 0 && ((end_offset) & (BUF_ALIGN_SIZE - 1)) == 0)
        {
            return;
        }
        char tmp_buf_head[BUF_ALIGN_SIZE] = {};
        char tmp_buf_tail[BUF_ALIGN_SIZE] = {};
        cudaSetDevice(device_id);
        // read head unaligned
        if ((*offset & (BUF_ALIGN_SIZE - 1)) != 0)
        {
            size_t read_head_size = min(BUF_ALIGN_SIZE - (*offset & (BUF_ALIGN_SIZE - 1)), *total_size);
            read_file(file_path, tmp_buf_head, (char *)res_tensor.data_ptr(), 1, read_head_size, *offset, use_sfcs_sdk,
                      use_direct_io, cipher_info);
            *read_unaligned_size = read_head_size;
            *offset += read_head_size;
            *total_size -= read_head_size;
        }
        // read tail unaligned
        if (*total_size > 0 && (end_offset & (BUF_ALIGN_SIZE - 1)) != 0)
        {
            size_t tail_offset = end_offset - (end_offset & (BUF_ALIGN_SIZE - 1));
            size_t tensor_offset = tail_offset - *offset + *read_unaligned_size;
            read_file(file_path, tmp_buf_tail, (char *)res_tensor.data_ptr() + tensor_offset, 1,
                      end_offset - tail_offset, tail_offset, use_sfcs_sdk, use_direct_io, cipher_info);
            *total_size -= end_offset - tail_offset;
        }
        cudaDeviceSynchronize();
    }
}

void IOHelper::load_file_to_tensor(std::string file_path, torch::Tensor res_tensor, torch::Tensor sample_tensor,
                                   int64_t offset, int64_t device_id, int64_t num_thread, bool use_pinmem,
                                   bool use_sfcs_sdk, bool use_direct_io, bool use_cipher,
                                   pybind11::array_t<char> key_arr, pybind11::array_t<char> iv_arr)
{
    size_t file_size = get_file_size(file_path.c_str(), use_sfcs_sdk);
    size_t total_size = file_size - offset;
    size_t read_unaligned_size = 0;
    // set cipher
    CipherInfo cipher_info(use_cipher, key_arr, iv_arr);
    if (device_id < 0)
    {
        read_unaligned_part(file_path, res_tensor, &offset, device_id, &total_size, use_sfcs_sdk, use_direct_io,
                            &read_unaligned_size, cipher_info);
        read_file(file_path, (char *)res_tensor.data_ptr() + read_unaligned_size, NULL, num_thread, total_size, offset,
                  use_sfcs_sdk, use_direct_io, cipher_info);
    }
    else
    {
        read_unaligned_part(file_path, res_tensor, &offset, device_id, &total_size, use_sfcs_sdk, use_direct_io,
                            &read_unaligned_size, cipher_info);

        // change use_pinmem attribute may introduce ambiguity
        if (buffer_size_ > 0 && use_pinmem != use_pinmem_)
        {
            throw std::runtime_error("use_pinmem attribute of an exising IOHelper should not be changed");
        }

        // TODO: HPA might be slow
        // only use pin_mem as buffer for copying data to device memory
        // the lifecycle of the pin_mem is the same as helper
        if (pin_mem == NULL || total_size > buffer_size_)
        {
            init_buffer(file_path, total_size, use_pinmem, use_sfcs_sdk);
        }
        cudaSetDevice(device_id);
        read_file(file_path, pin_mem, (char *)res_tensor.data_ptr() + read_unaligned_size, num_thread, total_size,
                  offset, use_sfcs_sdk, use_direct_io, CipherInfo());
        cudaDeviceSynchronize();
        if (cipher_info.use_cipher && total_size > 0)
        {
            if (offset % CTR_BLOCK_SIZE != 0 || total_size % CTR_BLOCK_SIZE != 0)
            {
                throw std::runtime_error("cannot decrypt because gpu read is not aligned");
            }
            unsigned char iv[CTR_BLOCK_SIZE];
            for (size_t i = 0; i < CTR_BLOCK_SIZE; i++)
            {
                iv[i] = cipher_info.iv[i];
            }
            ctr128_inc_by(iv, CTR_BLOCK_SIZE, offset / CTR_BLOCK_SIZE);
            unsigned char *iv_gpu;
            cudaMalloc((void **)&iv_gpu, CTR_BLOCK_SIZE);
            cudaMemcpy(iv_gpu, iv, CTR_BLOCK_SIZE, cudaMemcpyHostToDevice);
            unsigned char *ct = reinterpret_cast<unsigned char *>(res_tensor.data_ptr()) + read_unaligned_size;
            int cipher_ret = ctr_decrypt_gpu(cipher_info.key, iv_gpu, ct, total_size, ct);
            if (!cipher_ret)
            {
                throw std::runtime_error("Cipher Exception: gpu decrypt fail");
            }
            cudaDeviceSynchronize();
            cudaFree(iv_gpu);
        }
    }
}
