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
#ifndef _CLOUDFS_LIBCFS3_CLIENT_CFS_AIO_H_
#define _CLOUDFS_LIBCFS3_CLIENT_CFS_AIO_H_

#include <stdint.h> /* for uint64_t, etc. */

#ifdef __cplusplus
extern "C"
{
#endif
    /**
     * Some utility decls used in libcfs.
     */
    typedef int32_t tSize;   /// size of data for read/write io ops
    typedef int64_t tOffset; /// offset within the file

    struct CfsFileSystemInternalWrapper;
    typedef struct CfsFileSystemInternalWrapper *cfsFS;

    struct CfsFileInternalWrapper;
    typedef struct CfsFileInternalWrapper *cfsFile;

    typedef enum cfsStatus
    {
        STATUS_OK = 0,
        STATUS_MISSING_BLOCK = -1002,
        STATUS_TIMEOUT = -1003,
        STATUS_INVALID_RANGE = -1004,
        STATUS_CONNECTION_CLOSED = -1005,
        STATUS_WRITE_FAILED = -1006,
        STATUS_IO_BUSY = -1007,
        STATUS_INVALID_PARAMETER = -1098,
        STATUS_UNSUPPORTED_OP = -1099,
        STATUS_UNKNOWN_ERR = -1100,
    } cfsStatus;

    typedef void (*cfsWriteCallback)(cfsStatus status, void *args);

    typedef void (*cfsReadCallback)(cfsStatus status, int32_t readLength, char *buffer, void *args);

    typedef struct cfsAsyncContext
    {
        cfsReadCallback readCallback;
        cfsWriteCallback writeCallback;
        char *buffer;
        void *args;
    } cfsAsyncContext;

    /**
     * cfsAsyncPRead - Async positional read of data from an open file.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param offset    Position from which to read.
     * @param length    The length of the buffer.
     * @param context   The callback context passed by user.
     * @return          Status of Async method.
     */
    cfsStatus cfsAsyncPRead(cfsFS fs, cfsFile file, tSize length, tOffset offset, cfsAsyncContext *context);

    /**
     * cfsAsyncWrite - Write data to the internal buffer of outputstream,
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param buffer    The buffer to copy write bytes into.
     * @param length    The length of the buffer.
     * @param context   The callback context passed by user.
     * @return          Status of Async method.
     */
    cfsStatus cfsAsyncWrite(cfsFS fs, cfsFile file, const void *buffer, tSize length, cfsAsyncContext *context);

    /**
     * cfsAsyncFlush - Wait for data is acked by remote dn.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param context   The callback context passed by user.
     * @return          Status of Async method.
     */
    cfsStatus cfsAsyncFlush(cfsFS fs, cfsFile file, cfsAsyncContext *context);

    /**
     * cfsAsyncWriteAndFlush -  Write data to remote datanode and wait for ack.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param buffer    The buffer to copy write bytes into.
     * @param length    The length of the buffer.
     * @param context   The callback context passed by user.
     * @return          Status of Async method.
     */
    cfsStatus cfsAsyncWriteAndFlush(cfsFS fs, cfsFile file, const void *buffer, tSize length, cfsAsyncContext *context);

#ifdef __cplusplus
}
#endif

#endif /* _CLOUDFS_LIBCFS3_CLIENT_CFS_AIO_H_ */
