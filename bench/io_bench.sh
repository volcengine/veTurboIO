###
 # Copyright (c) 2024 Beijing Volcano Engine Technology Ltd.
 # 
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 # 
 #     http://www.apache.org/licenses/LICENSE-2.0
 # 
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
###

# shm
mkdir -p /dev/shm/test_files
python bench/io_bench.py --load_mode=veturboio,torch --base_dir=/dev/shm/test_files --begin=1GB --end=4GB --gen_data --fs_name=shm

# sfcs
python bench/io_bench.py --load_mode=veturboio,torch --base_dir=sfcs:// --begin=1GB --end=4GB --gen_data --fs_name=sfcs
