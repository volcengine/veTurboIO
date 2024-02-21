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
#ifndef AES_CPU_CTR_H
#define AES_CPU_CTR_H

#include <stdio.h>

extern const size_t EVP_UPDATE_MAX;
extern const size_t CTR_BLOCK_SIZE;

void ctr128_inc_by(unsigned char *counter, size_t n, size_t c);

typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_cipher_st EVP_CIPHER;

class CtrEncrypter
{
  private:
    EVP_CIPHER_CTX *ctx = NULL;
    EVP_CIPHER *cipher = NULL;

  public:
    CtrEncrypter(const unsigned char *key, const unsigned char *iv, size_t global_offset);
    ~CtrEncrypter();
    int encrypt_update(unsigned char *pt, size_t pt_size, unsigned char *ct);
};

class CtrDecrypter
{
  private:
    EVP_CIPHER_CTX *ctx = NULL;
    EVP_CIPHER *cipher = NULL;

  public:
    CtrDecrypter(const unsigned char *key, const unsigned char *iv, size_t global_offset);
    ~CtrDecrypter();
    int decrypt_update(unsigned char *ct, size_t ct_size, unsigned char *pt);
};
#endif

#ifndef AES_GPU_CTR_H
#define AES_GPU_CTR_H

#include <stdio.h>

// Both encrypt and decrypt require length of ct and pt multiple of 16

int ctr_encrypt_gpu(const unsigned char *key, const unsigned char *iv, unsigned char *pt, size_t pt_size,
                    unsigned char *ct);

int ctr_decrypt_gpu(const unsigned char *key, const unsigned char *iv, unsigned char *ct, size_t ct_size,
                    unsigned char *pt);

#endif