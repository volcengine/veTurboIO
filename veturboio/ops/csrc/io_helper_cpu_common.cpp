#include "include/io_helper.h"
#include "include/cipher.h"

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

void read_file(string file_path, char *addr, int device_id, char *dev_mem, int num_thread, size_t total_size,
               size_t global_offset, bool use_sfcs_sdk, bool use_direct_io, CipherInfo cipher_info)
{
    if (total_size == 0)
    {
        return;
    }

    if (use_sfcs_sdk)
    {
        SFCSFile sfcs_file(file_path, cipher_info);
        sfcs_file.read_file_to_address_parallel(addr, device_id, dev_mem, num_thread, total_size, global_offset);
    }
    else
    {
        POSIXFile posix_file(file_path, cipher_info);
        posix_file.read_file_to_address_parallel(addr, device_id, dev_mem, num_thread, total_size, global_offset,
                                                 use_direct_io);
    }
}

void load_file_to_tensor_cpu(std::string file_path, torch::Tensor res_tensor, size_t length, int64_t offset,
                             int64_t device_id, int64_t num_thread, bool use_pinmem, bool use_sfcs_sdk,
                             bool use_direct_io, bool use_cipher, pybind11::array_t<char> key_arr,
                             pybind11::array_t<char> iv_arr, int64_t header_size)
{
    size_t file_size = get_file_size(file_path.c_str(), use_sfcs_sdk);
    size_t read_unaligned_size = 0;
    size_t total_size = length > 0 ? length : file_size - offset;
    // set cipher
    CipherInfo cipher_info(use_cipher, key_arr, iv_arr, header_size);
    if (device_id < 0)
    {
        read_file(file_path, (char *)res_tensor.data_ptr() + read_unaligned_size, device_id, NULL, num_thread,
                  total_size, offset, use_sfcs_sdk, use_direct_io, cipher_info);
    }
}

void IOHelper::save_tensor_to_file_cpu(torch::Tensor tensor, std::string file_path, size_t length, bool use_pinmem,
                                       bool use_sfcs_sdk, bool use_cipher, pybind11::array_t<char> key_arr,
                                       pybind11::array_t<char> iv_arr, int64_t header_size)
{
    char *buf;

    CipherInfo cipher_info(use_cipher, key_arr, iv_arr, header_size);
    if (use_cipher)
    {
        // change use_pinmem attribute may introduce ambiguity
        if (buffer_size_ > 0 && use_pinmem != use_pinmem_)
        {
            throw std::runtime_error("use_pinmem attribute of an exising IOHelper should not be changed");
        }

        if (pin_mem == NULL || length > buffer_size_)
        {
            init_buffer(file_path, length, use_pinmem, use_sfcs_sdk);
        }

        buf = pin_mem;
        memcpy(buf, (char *)tensor.data_ptr(), length);
    }
    else
    {
        buf = (char *)tensor.data_ptr();
    }

    if (use_sfcs_sdk)
    {
        SFCSFile sfcs_file(file_path, cipher_info);
        sfcs_file.write_file_from_addr(buf, length, 0, true);
    }
    else
    {
        POSIXFile posix_file(file_path, cipher_info);
        posix_file.write_file_from_addr(buf, length, true);
    }
}
