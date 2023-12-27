#include "include/io_helper.h"
#include "include/cipher.h"

IOHelper::~IOHelper()
{
}

// init buffer with given positive size or the size of the file in specified
// path
void IOHelper::init_buffer(string file_path, int64_t buffer_size, bool use_pinmem, bool use_sfcs_sdk)
{
}

void IOHelper::free_buffer()
{
}

void IOHelper::load_file_to_tensor(std::string file_path, torch::Tensor res_tensor, size_t length, int64_t offset,
                                   int64_t device_id, int64_t num_thread, bool use_pinmem, bool use_sfcs_sdk,
                                   bool use_direct_io, bool use_cipher, pybind11::array_t<char> key_arr,
                                   pybind11::array_t<char> iv_arr, int64_t header_size)
{
    load_file_to_tensor_cpu(file_path, res_tensor, length, offset, device_id, num_thread, use_pinmem, use_sfcs_sdk,
                            use_direct_io, use_cipher, key_arr, iv_arr, header_size);
}

void IOHelper::save_tensor_to_file(torch::Tensor tensor, std::string file_path, size_t length, bool use_pinmem,
                                   bool use_sfcs_sdk, bool use_cipher, pybind11::array_t<char> key_arr,
                                   pybind11::array_t<char> iv_arr, int64_t header_size)
{
    save_tensor_to_file_cpu(tensor, file_path, length, use_pinmem, use_sfcs_sdk, use_cipher, key_arr, iv_arr,
                            header_size);
}
