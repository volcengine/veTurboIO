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
#include "include/io_helper.h"
#include "include/sfcs.h"
#include "include/cipher.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<IOHelper>(m, "IOHelper").def(py::init<>()).def("load_file_to_tensor", &IOHelper::load_file_to_tensor);

    py::class_<SFCSFile>(m, "SFCSFile")
        .def(py::init<std::string>())
        .def(py::init<std::string, bool, pybind11::array_t<char>, pybind11::array_t<char>, size_t>())
        .def("get_file_size", &SFCSFile::get_file_size)
        .def("read_file_to_array", &SFCSFile::read_file_to_array)
        .def("write_file_from_array", &SFCSFile::write_file_from_array)
        .def("delete_file", &SFCSFile::delete_file);

    py::class_<CtrEncWrap>(m, "CtrEncWrap")
        .def(py::init<std::string, pybind11::array_t<unsigned char>, pybind11::array_t<unsigned char>, size_t>())
        .def("encrypt_update", &CtrEncWrap::encrypt_update);

    py::class_<CtrDecWrap>(m, "CtrDecWrap")
        .def(py::init<std::string, pybind11::array_t<unsigned char>, pybind11::array_t<unsigned char>, size_t>())
        .def("decrypt_update", &CtrDecWrap::decrypt_update);
}
