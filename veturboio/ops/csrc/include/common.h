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
#ifndef COMMON_H
#define COMMON_H

#include <torch/torch.h>
#include <torch/extension.h>
#if defined(USE_CUDA)
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include "cfs.h"
#include "logging.h"
#include "sfcs.h"

#define THREAD_NICE_ADJ -10
#define BUF_ALIGN_SIZE (size_t)4096

using namespace std;

#endif
