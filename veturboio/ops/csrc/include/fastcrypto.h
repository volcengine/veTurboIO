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
#ifndef VETURBOIO_FASTCRYPTO_H
#define VETURBOIO_FASTCRYPTO_H

#include <stdio.h>
#include <string>

#define EVP_UPDATE_MAX 0x7ffffff0
#define AES_BLOCK_SIZE 16
#define AES_BUF_MAX_SIZE 32
#define MAX_CTR_KEY_SIZE 32
#define FASTCRYPTO_MAGIC_SIZE 16

inline void counter_inc_by(unsigned char *counter, size_t n, size_t c)
{
    do
    {
        --n;
        c += counter[n];
        counter[n] = static_cast<unsigned char>(c);
        c >>= 8;
    } while (n);
}

typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_cipher_st EVP_CIPHER;
typedef struct evp_mac_ctx_st EVP_MAC_CTX;
typedef struct evp_mac_st EVP_MAC;

class CtrEncrypter
{
  private:
    EVP_CIPHER_CTX *ctx = NULL;
    EVP_CIPHER *cipher = NULL;

  public:
    CtrEncrypter() = default;
    CtrEncrypter(std::string algo, const unsigned char *key, const unsigned char *iv, size_t global_offset);
    ~CtrEncrypter();
    int encrypt_update(unsigned char *pt, size_t pt_size, unsigned char *ct);
};

class CtrDecrypter
{
  private:
    EVP_CIPHER_CTX *ctx = NULL;
    EVP_CIPHER *cipher = NULL;

  public:
    CtrDecrypter() = default;
    CtrDecrypter(std::string algo, const unsigned char *key, const unsigned char *iv, size_t global_offset);
    ~CtrDecrypter();
    int decrypt_update(unsigned char *ct, size_t ct_size, unsigned char *pt);
};

// Both encrypt and decrypt require length of ct and pt multiple of 16
int ctr_encrypt_gpu(std::string algo, const unsigned char *key, const unsigned char *iv, unsigned char *pt,
                    size_t pt_size, unsigned char *ct);

int ctr_decrypt_gpu(std::string algo, const unsigned char *key, const unsigned char *iv, unsigned char *ct,
                    size_t ct_size, unsigned char *pt);
#endif