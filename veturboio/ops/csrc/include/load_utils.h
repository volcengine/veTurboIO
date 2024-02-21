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
#ifndef LOAD_UTILS_H
#define LOAD_UTILS_H

#include "common.h"

void read_file(string file_path, char *addr, char *dev_mem, int num_thread, size_t total_size, size_t global_offset,
               bool use_sfcs_sdk, bool use_direct_io, CipherInfo cipher_info);
size_t get_file_size(const char *file_name, bool use_sfcs_sdk);

#endif