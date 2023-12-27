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
#include "include/cipher.h"
#include "include/fastcrypto.h"

SFCSFs::SFCSFs()
{
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

SFCSFs::~SFCSFs()
{
    cfsDisconnect(fs);
}

void SFCSFs::concat_files(std::string file_name, vector<const char *> file_paths)
{
    int ret;

    ret = cfsConcat(fs, file_name.c_str(), &file_paths[0], file_paths.size());
    if (ret == -1)
    {
        logError("Failed to concat files", cfsGetLastError());
        throw std::runtime_error("SFCS Exception: concat files");
    }
}

void SFCSFs::rename_file(const char *file_path, const char *file_name)
{
    int ret;

    ret = cfsRename2(fs, file_path, file_name);
    if (ret == -1)
    {
        logError("Failed to rename file", file_path, cfsGetLastError());
        throw std::runtime_error("SFCS Exception: rename file");
    }
}

int64_t SFCSFs::get_block_size()
{
    int64_t ret;

    ret = cfsGetDefaultBlockSize(fs);
    if (ret == -1)
    {
        logError("Failed to get default block size", cfsGetLastError());
        throw std::runtime_error("SFCS Exception: get block size");
    }
    return ret;
}

void SFCSFs::mkdir(std::string file_path)
{
    int ret;

    ret = cfsCreateDirectory(fs, file_path.c_str());
    if (ret == -1)
    {
        logError("Failed to create dir", file_path, cfsGetLastError());
        throw std::runtime_error("SFCS Exception: create dir");
    }
}

size_t SFCSFs::read_file_to_addr(std::string file_name, CipherInfo cipher_info, char *addr, size_t length,
                                 size_t offset)
{
    SFCSFile sfcs_file(file_name, this, cipher_info);
    return sfcs_file.read_file_to_addr(addr, length, offset);
}

void SFCSFs::read_multi_files(pybind11::list file_paths, pybind11::list tensors, pybind11::list lengths,
                              pybind11::list offsets, int num_thread, bool use_cipher, pybind11::array_t<char> key_arr,
                              pybind11::array_t<char> iv_arr, size_t header_size)
{
    vector<thread> threads(num_thread);
    auto file_names = file_paths.cast<std::vector<std::string>>();
    auto tensors_vector = tensors.cast<std::vector<torch::Tensor>>();
    auto lengths_vector = lengths.cast<std::vector<size_t>>();
    auto offsets_vector = offsets.cast<std::vector<size_t>>();

    CipherInfo cipher_info = CipherInfo(use_cipher, key_arr, iv_arr, header_size);
    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        std::string file_name = file_names[thread_id];
        size_t length = lengths_vector[thread_id];
        size_t offset = offsets_vector[thread_id];
        torch::Tensor tensor = tensors_vector[thread_id];
        char *addr = (char *)tensor.data_ptr();

        threads[thread_id] =
            std::thread(&SFCSFs::read_file_to_addr, this, file_name, cipher_info, addr, length, offset);
    }

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id].join();
    }
}

size_t SFCSFs::write_file_from_addr(std::string file_name, CipherInfo cipher_info, char *addr, size_t length,
                                    size_t offset)
{
    SFCSFile sfcs_file(file_name, this, cipher_info);
    return sfcs_file.write_file_from_addr(addr, length, offset, false);
}

void SFCSFs::write_multi_files(pybind11::list file_paths, pybind11::list tensors, pybind11::list lengths,
                               pybind11::list offsets, int num_thread, bool use_cipher, pybind11::array_t<char> key_arr,
                               pybind11::array_t<char> iv_arr, size_t header_size)
{
    vector<thread> threads(num_thread);
    auto file_names = file_paths.cast<std::vector<std::string>>();
    auto tensors_vector = tensors.cast<std::vector<torch::Tensor>>();
    auto lengths_vector = lengths.cast<std::vector<size_t>>();
    auto offsets_vector = offsets.cast<std::vector<size_t>>();

    CipherInfo cipher_info = CipherInfo(use_cipher, key_arr, iv_arr, header_size);
    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        std::string file_name = file_names[thread_id];
        size_t length = lengths_vector[thread_id];
        size_t offset = offsets_vector[thread_id];
        torch::Tensor tensor = tensors_vector[thread_id];
        char *addr = (char *)tensor.data_ptr();

        threads[thread_id] =
            std::thread(&SFCSFs::write_file_from_addr, this, file_name, cipher_info, addr, length, offset);
    }

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id].join();
    }
}

void SFCSFs::get_file_size(std::string file_name, size_t *size)
{
    SFCSFile sfcs_file(file_name, this);
    *size = sfcs_file.get_file_size();
}

void SFCSFs::get_multi_file_size(pybind11::list file_paths, pybind11::list sizes, int num_thread)
{
    vector<thread> threads(num_thread);
    auto file_names = file_paths.cast<std::vector<std::string>>();
    vector<size_t> lengths(num_thread);

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        std::string file_name = file_names[thread_id];
        threads[thread_id] = std::thread(&SFCSFs::get_file_size, this, file_name, &lengths[thread_id]);
    }

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id].join();
        sizes.append(lengths[thread_id]);
    }
}

SFCSFile::SFCSFile(std::string path)
{
    file_path = path;
    sfcs_fs = new SFCSFs();
    fs_owner = true;
    fs = sfcs_fs->fs;
}

SFCSFile::SFCSFile(std::string path, SFCSFs *sfcs_fs)
{
    file_path = path;
    this->sfcs_fs = sfcs_fs;
    fs_owner = false;
    fs = sfcs_fs->fs;
}

SFCSFile::SFCSFile(std::string file_path, CipherInfo cipher_info) : SFCSFile(file_path)
{
    this->cipher_info = cipher_info;
}

SFCSFile::SFCSFile(std::string file_path, SFCSFs *sfcs_fs, CipherInfo cipher_info) : SFCSFile(file_path, sfcs_fs)
{
    this->cipher_info = cipher_info;
}

SFCSFile::SFCSFile(std::string file_path, bool use_cipher, pybind11::array_t<char> key_arr,
                   pybind11::array_t<char> iv_arr, size_t header_size)
    : SFCSFile(file_path)
{
    this->cipher_info = CipherInfo(use_cipher, key_arr, iv_arr, header_size);
}

SFCSFile::~SFCSFile()
{
    if (fs_owner)
    {
        delete sfcs_fs;
    }
}

size_t SFCSFile::get_file_size()
{
    size_t size;
    // get path info
    cfsFileInfo *file_info = cfsGetPathInfo(fs, file_path.c_str());
    if (file_info == NULL)
    {
        logWarn("Failed to get path info of relative path", file_path, cfsGetLastError());
        cfsFreeFileInfo(file_info, 1);
        return 0;
    }
    else
    {
        size = file_info->mSize;
        cfsFreeFileInfo(file_info, 1);
        return size;
    }
}

size_t SFCSFile::read_file_to_addr(char *addr, size_t length, size_t offset)
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
        CtrDecrypter dec(cipher_info.mode, cipher_info.key, cipher_info.iv, offset - cipher_info.header_size);
        unsigned char *ct = reinterpret_cast<unsigned char *>(addr);
        int cipher_ret = dec.decrypt_update(ct, length - count, ct);
        if (!cipher_ret)
        {
            throw std::runtime_error("Cipher Exception: decrypt fail");
        }
    }

    return length - count;
}

void SFCSFile::read_file_to_address_thread(int thread_id, char *addr, int device_id, char *dev_mem, size_t block_size,
                                           size_t total_size, size_t global_offset)
{
    size_t offset = thread_id * block_size;
    size_t read_size = block_size;

    if (offset + read_size >= total_size)
    {
        read_size = (total_size > offset) ? total_size - offset : 0;
    }

    // TODO: actual number of bytes read may be less than read_size
    read_file_to_addr(addr + offset, read_size, global_offset + offset);

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

size_t SFCSFile::read_file_to_address_parallel(char *addr, int device_id, char *dev_mem, int num_thread,
                                               size_t total_size, size_t global_offset)
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
        threads[thread_id] = std::thread(&SFCSFile::read_file_to_address_thread, this, thread_id, addr, device_id,
                                         dev_mem, block_size, total_size, global_offset);
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
    madvise(addr, length, MADV_HUGEPAGE);
    return read_file_to_address_parallel(addr, -1, NULL, num_thread, length, offset);
}

size_t SFCSFile::write_file_from_addr(char *addr, size_t length, size_t offset, bool append)
{
    size_t count;
    int32_t ret;
    char *dst;

    if (append)
        offset = get_file_size();

    if (cipher_info.use_cipher)
    {
        size_t h_off = cipher_info.header_size;
        int cipher_ret;

        if (append == false && offset == 0)
        {
            CtrEncrypter enc(cipher_info.mode, cipher_info.key, cipher_info.iv, 0);
            unsigned char *pt = reinterpret_cast<unsigned char *>(addr);
            cipher_ret = enc.encrypt_update(pt + h_off, length - h_off, pt + h_off);
        }
        else
        {
            CtrEncrypter enc(cipher_info.mode, cipher_info.key, cipher_info.iv, offset - h_off);
            unsigned char *pt = reinterpret_cast<unsigned char *>(addr);
            cipher_ret = enc.encrypt_update(pt, length, pt);
        }

        if (!cipher_ret)
        {
            throw std::runtime_error("Cipher Exception: encrypt fail");
        }
    }

    cfsFile file;
    if (append)
        file = cfsOpenFileAcc(fs, file_path.c_str(), O_WRONLY | O_ASYNC | O_APPEND, 0644, false, true);
    else
        file = cfsOpenFileAcc(fs, file_path.c_str(), O_WRONLY | O_ASYNC, 0644, false, false);

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

size_t SFCSFile::write_file_from_array(pybind11::array_t<char> arr, size_t length, bool append)
{
    pybind11::buffer_info buf_info = arr.request();
    char *addr = static_cast<char *>(buf_info.ptr);
    return write_file_from_addr(addr, length, 0, append);
}

void SFCSFile::write_file_from_tensor(torch::Tensor tensor, size_t length, size_t offset, std::string file_name)
{
    char *buf, *addr;
    buf = (char *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    madvise(buf, length, MADV_HUGEPAGE);

    if (tensor.device().is_cuda())
    {
#if defined(USE_CUDA)
        cudaSetDevice(tensor.device().index());
        cudaMemcpyAsync(buf, (char *)tensor.data_ptr(), length, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        addr = buf;
#endif
    }
    else if (cipher_info.use_cipher)
    {
        memcpy(buf, (char *)tensor.data_ptr(), length);
        addr = buf;
    }
    else
    {
        addr = (char *)tensor.data_ptr();
    }

    SFCSFile sfcs_file(file_name, sfcs_fs, cipher_info);
    sfcs_file.write_file_from_addr(addr, length, offset, false);
    munmap(buf, length);
}

size_t SFCSFile::write_file_from_tensors(pybind11::list tensors, pybind11::list sizes, pybind11::list offsets,
                                         std::string concat_dir, std::string concat_file)
{
    int num_thread = tensors.size();
    size_t length = 0;

    vector<thread> threads(num_thread);
    vector<std::string> file_names;
    vector<const char *> file_paths;
    auto tensors_vector = tensors.cast<std::vector<torch::Tensor>>();
    auto sizes_vector = sizes.cast<std::vector<size_t>>();
    auto offsets_vector = offsets.cast<std::vector<size_t>>();

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        torch::Tensor tensor = tensors_vector[thread_id];
        size_t size = sizes_vector[thread_id];
        size_t offset = offsets_vector[thread_id];
        file_names.push_back(concat_dir + std::string("/") + std::to_string(thread_id));
        threads[thread_id] =
            std::thread(&SFCSFile::write_file_from_tensor, this, tensor, size, offset, file_names[thread_id]);
        file_paths.push_back(file_names[thread_id].c_str());
        length += size;
    }

    for (int thread_id = 0; thread_id < num_thread; thread_id++)
    {
        threads[thread_id].join();
    }

    sfcs_fs->concat_files(concat_file, file_paths);
    sfcs_fs->rename_file(concat_file.c_str(), file_path.c_str());

    return length;
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
