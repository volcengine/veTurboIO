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

#ifndef _CLOUDFS_LIBCFS3_CLIENT_CFS_H_
#define _CLOUDFS_LIBCFS3_CLIENT_CFS_H_

#include <errno.h>  /* for EINTERNAL, etc. */
#include <fcntl.h>  /* for O_RDONLY, O_WRONLY */
#include <stdint.h> /* for uint64_t, etc. */
#include <time.h>   /* for time_t */
#include <stdbool.h>

#ifndef O_RDONLY
#define O_RDONLY 0x01
#endif

#ifndef O_WRONLY
#define O_WRONLY 0x02
#endif

#ifndef O_ASYNC
#define O_ASYNC 0x04
#endif

#ifndef O_LOCAL
#define O_LOCAL 0x01
#endif

#ifndef EINTERNAL
#define EINTERNAL 255
#endif

/** All APIs set errno to meaningful values */

#ifdef __cplusplus
extern "C"
{
#endif
    /**
     * Some utility decls used in libcfs.
     */
    typedef int32_t tSize;   /// size of data for read/write io ops
    typedef time_t tTime;    /// time type in seconds
    typedef int64_t tLong;   // size of data for read/write io ops
    typedef int64_t tOffset; /// offset within the file
    typedef uint16_t tPort;  /// port

    typedef enum tCfsObjectKind
    {
        kCfsObjectKindFile = 'F',
        kCfsObjectKindDirectory = 'D',
    } tCfsObjectKind;

    typedef enum tCfsObjectAccStatus
    {
        kUnknown = 0,
        kFileLocal = 1,
        kFileToBePersisted = 2,
        kFilePersisted = 3,
        kDirLocal = 4,
        kDirIncomplete = 5,
        kDirSynced = 6,
    } tCfsObjectAccStatus;

    struct CfsFileSystemInternalWrapper;
    typedef struct CfsFileSystemInternalWrapper *cfsFS;

    struct CfsFileInternalWrapper;
    typedef struct CfsFileInternalWrapper *cfsFile;

    struct cfsBuilder;

    /**
     * cfsGetLastError - Return error information of last failed operation.
     *
     * @return 			A not NULL const string point of last error information.
     * 					Caller can only read this message and keep it unchanged. No need to free it.
     * 					If last operation finished successfully, the returned message is undefined.
     */
    const char *cfsGetLastError();

    /**
     * cfsFileIsOpenForRead - Determine if a file is open for read.
     *
     * @param file     The CFS file
     * @return         1 if the file is open for read; 0 otherwise
     */
    int cfsFileIsOpenForRead(cfsFile file);

    /**
     * cfsFileIsOpenForWrite - Determine if a file is open for write.
     *
     * @param file     The CFS file
     * @return         1 if the file is open for write; 0 otherwise
     */
    int cfsFileIsOpenForWrite(cfsFile file);

    /**
     * cfsConnectAsUser - Connect to a cfs file system as a specific user
     *
     * Connect to the cfs.
     *
     * @param nn        The NameNode.  See cfsBuilderSetNameNode for details.
     * @param port      The port on which the server is listening.
     * @param user      the user name (this is hadoop domain user). Or NULL is equivelant to hcfsConnect(host, port)
     * @return          Returns a handle to the filesystem or NULL on error.
     * @deprecated      Use cfsBuilderConnect instead.
     */
    cfsFS cfsConnectAsUser(const char *nn, tPort port, const char *user);

    /**
     * cfsConnect - Connect to a cfs file system.
     *
     * @param nn        The NameNode.  See cfsBuilderSetNameNode for details.
     * @param port      The port on which the server is listening.
     * @return          Returns a handle to the filesystem or NULL on error.
     * @deprecated      Use cfsBuilderConnect instead.
     */
    cfsFS cfsConnect(const char *nn, tPort port);

    /**
     * cfsConnectAsUserNewInstance - Forces a new instance to be created
     *
     * @param nn        The NameNode.  See cfsBuilderSetNameNode for details.
     * @param port      The port on which the server is listening.
     * @param user      The user name to use when connecting
     * @return          Returns a handle to the filesystem or NULL on error.
     * @deprecated      Use cfsBuilderConnect instead.
     */
    cfsFS cfsConnectAsUserNewInstance(const char *nn, tPort port, const char *user);

    /**
     * cfsConnectNewInstance - Forces a new instance to be created
     *
     * @param nn        The NameNode.  See cfsBuilderSetNameNode for details.
     * @param port      The port on which the server is listening.
     * @return          Returns a handle to the filesystem or NULL on error.
     * @deprecated      Use cfsBuilderConnect instead.
     */
    cfsFS cfsConnectNewInstance(const char *nn, tPort port);

    /**
     * cfsBuilderConnect - Connect to CFS using the parameters defined by the builder.
     *
     * @param bld            The CFS builder
     * @param effective_user The user name.
     * @return               Returns a handle to the filesystem, or NULL on error.
     */
    cfsFS cfsBuilderConnect(struct cfsBuilder *bld, const char *effective_user);

    /**
     * cfsNewBuilder - Create an CFS builder.
     *
     * @return The CFS builder, or NULL on error.
     */
    struct cfsBuilder *cfsNewBuilder(void);

    /**
     * cfsBuilderSetForceNewInstance - Do nothing, we always create a new instance
     *
     * @param bld The CFS builder
     */
    void cfsBuilderSetForceNewInstance(struct cfsBuilder *bld);

    /**
     * cfsBuilderSetNameNode - Set the CFS NameNode to connect to.
     *
     * @param bld  The CFS builder
     * @param nn   The NameNode to use.
     *
     *             If the string given is 'default', the default NameNode
     *             configuration will be used (from the XML configuration files)
     *
     *             If NULL is given, a LocalFileSystem will be created.
     *
     *             If the string starts with a protocol type such as file:// or
     *             cfs://, this protocol type will be used.  If not, the
     *             cfs:// protocol type will be used.
     *
     *             You may specify a NameNode port in the usual way by
     *             passing a string of the format cfs://<hostname>:<port>.
     *             Alternately, you may set the port with
     *             cfsBuilderSetNameNodePort.  However, you must not pass the
     *             port in two different ways.
     */
    void cfsBuilderSetNameNode(struct cfsBuilder *bld, const char *nn);

    /**
     * cfsBuilderSetNameNodePort - Set the port of the CFS NameNode to connect to.
     *
     * @param bld   The CFS builder
     * @param port  The port.
     */
    void cfsBuilderSetNameNodePort(struct cfsBuilder *bld, tPort port);

    /**
     * cfsBuilderSetUserName - Set the username to use when connecting to the CFS cluster.
     *
     * @param bld       The CFS builder
     * @param userName  The user name.  The string will be shallow-copied.
     */
    void cfsBuilderSetUserName(struct cfsBuilder *bld, const char *userName);

    /**
     * cfsBuilderSetKerbTicketCachePath -  Set the path to the Kerberos ticket
     *                                      cache to use when connecting to
     *                                      the CFS cluster.
     *
     * @param bld                           The CFS builder
     * @param kerbTicketCachePath           The Kerberos ticket cache path.  The string
     *                                      will be shallow-copied.
     */
    void cfsBuilderSetKerbTicketCachePath(struct cfsBuilder *bld, const char *kerbTicketCachePath);

    /**
     * cfsBuilderSetToken - Set the token used to authenticate
     *
     * @param bld   The CFS builder
     * @param token The token used to authenticate
     */
    void cfsBuilderSetToken(struct cfsBuilder *bld, const char *token);

    /**
     * cfsBuilderSetToken - Set the GDPR token to authenticate
     *
     * @param fs    The configured filesystem handle.
     * @param token The custom GDPR token
     */
    int cfsSetCustomToken(cfsFS fs, const char *token);

    /**
     * cfsFreeBuilder - Free an CFS builder.
     *
     * It is normally not necessary to call this function since
     * cfsBuilderConnect frees the builder.
     *
     * @param bld The CFS builder
     */
    void cfsFreeBuilder(struct cfsBuilder *bld);

    /**
     * cfsBuilderConfSetStr - Set a configuration string for an CfsBuilder.
     *
     * @param bld      The CFS builder
     * @param key      The key to set.
     * @param val      The value, or NULL to set no value.
     *                 This will be shallow-copied.  You are responsible for
     *                 ensuring that it remains valid until the builder is
     *                 freed.
     *
     * @return         0 on success; nonzero error code otherwise.
     */
    int cfsBuilderConfSetStr(struct cfsBuilder *bld, const char *key, const char *val);

    /**
     * cfsConfGetStr - Get a configuration string.
     *
     * @param key      The key to find
     * @param val      (out param) The value.  This will be set to NULL if the
     *                 key isn't found.  You must free this string with
     *                 cfsConfStrFree.
     *
     * @return         0 on success; nonzero error code otherwise.
     *                 Failure to find the key is not an error.
     */
    int cfsConfGetStr(const char *key, char **val);

    /**
     * cfsConfGetInt - Get a configuration integer.
     *
     * @param key      The key to find
     * @param val      (out param) The value.  This will NOT be changed if the
     *                 key isn't found.
     *
     * @return         0 on success; nonzero error code otherwise.
     *                 Failure to find the key is not an error.
     */
    int cfsConfGetInt(const char *key, int32_t *val);

    /**
     * cfsConfStrFree - Free a configuration string found with cfsConfGetStr.
     *
     * @param val      A configuration string obtained from cfsConfGetStr
     */
    void cfsConfStrFree(char *val);

    /**
     * cfsDisconnect - Disconnect from the cfs file system.
     *
     * Disconnect from cfs.
     *
     * @param fs    The configured filesystem handle.
     * @return      Returns 0 on success, -1 on error.
     *              Even if there is an error, the resources associated with the
     *              cfsFS will be freed.
     * @deprecated  Use cfsBuilderConnect instead.
     */
    int cfsDisconnect(cfsFS fs);

    /**
     * cfsOpenFile - Open a cfs file in given mode.
     *
     * @param fs            The configured filesystem handle.
     * @param path          The full path to the file.
     * @param flags         an | of bits/fcntl.h file flags - supported flags are O_RDONLY, O_WRONLY (meaning create or
     * overwrite i.e., implies O_TRUNCAT), O_WRONLY|O_APPEND and O_SYNC. Other flags are generally ignored other than
     * (O_RDWR || (O_EXCL & O_CREAT)) which return NULL and set errno equal ENOTSUP.
     * @param bufferSize    Size of buffer for read/write - pass 0 if you want
     *                      to use the default configured values.
     * @param replication   Block replication - pass 0 if you want to use
     *                      the default configured values.
     * @param blocksize     Size of block - pass 0 if you want to use the
     *                      default configured values.
     * @return              Returns the handle to the open file or NULL on error.
     */
    cfsFile cfsOpenFile(cfsFS fs, const char *path, int flags, int bufferSize, short replication, tOffset blocksize);

    /**
     * cfsOpenFileV2 - Open a cfs file for Read (with ByteCool support).
     *
     * @param fs            The configured filesystem handle.
     * @param path          The full path to the file.
     * @param bufferSize    Size of buffer for read/write - pass 0 if you want
     *                      to use the default configured values.
     * @param objectNamePtr object name for ByteCool - pass NULL to disable
     *                      ByteCool support.
     * @return              Returns the handle to the open file or NULL on error.
     */
    cfsFile cfsOpenFileV2(cfsFS fs, const char *path, int bufferSize, char **objectNamePtr);

    /**
     * cfsAppendFileV2 - Open a cfs file for Append
     *
     * @param fs            The configured filesystem handle.
     * @param path          The full path to the file.
     * @param bufferSize    Size of buffer for read/write - pass 0 if you want
     *                      to use the default configured values.
     * @return              Returns the handle to the open file or NULL on error.
     */
    cfsFile cfsAppendFileV2(cfsFS fs, const char *path, int bufferSize);

    /**
     * cfsOpenFileACC - Open a cfs file in given mode. For ACC FS only.
     *
     * @param fs            The configured filesystem handle.
     * @param path          The full path to the file.
     * @param flags         an | of bits/fcntl.h file flags - supported flags are O_RDONLY, O_WRONLY (meaning create or
     * overwrite i.e., implies O_TRUNCAT), O_WRONLY|O_APPEND and O_SYNC. Other flags are generally ignored other than
     * (O_RDWR || (O_EXCL & O_CREAT)) which return NULL and set errno equal ENOTSUP.
     * @param mode          File mode for create.
     * @param createParent  if the parent does not exist, create it.
     * @param isAppendable  Specify appendable object for ACC mode.
     * @return              Returns the handle to the open file or NULL on error.
     */
    cfsFile cfsOpenFileAcc(cfsFS fs, const char *path, int flags, mode_t mode, int createParent, int isAppendable);

    /**
     * cfsCreateFileV2 - Create a cfs file
     *
     * @param fs            The configured filesystem handle.
     * @param path          The full path to the file.
     * @param overwrite     if a file with this name already exists, then if 1,
     *                      the file will be overwritten, and if 0 an error will be thrown.
     * @param bufferSize    Size of buffer for read/write - pass 0 if you want
     *                      to use the default configured values.
     * @param replication   Block replication - pass 0 if you want to use
     *                      the default configured values.
     * @param blockSize     Size of block - pass 0 if you want to use the
     *                      default configured values.
     * @return              Returns the handle to the open file or NULL on error.
     */
    cfsFile cfsCreateFileV2(cfsFS fs, const char *path, int overwrite, int bufferSize, short replication,
                            tSize blockSize);

    /**
     * cfsCloseFile - Close an open file.
     *
     * @param fs    The configured filesystem handle.
     * @param file  The file handle.
     * @return      Returns 0 on success, -1 on error.
     *              On error, errno will be set appropriately.
     *              If the cfs file was valid, the memory associated with it will
     *              be freed at the end of this call, even if there was an I/O
     *              error.
     */
    int cfsCloseFile(cfsFS fs, cfsFile file);

    /**
     * cfsExists - Checks if a given path exsits on the filesystem.
     *             Use cfsExistsExtended instead if possible.
     *
     * @param fs    The configured filesystem handle.
     * @param path  The path to look for
     * @return      Returns 0 on success, -1 on error.
     */
    int cfsExists(cfsFS fs, const char *path);

    /**
     * cfsExistsExtended - Checks if a given path exsits on the filesystem
     *
     * @param fs    The configured filesystem handle.
     * @param path  The path to look for
     * @return      Returns 1 on success, 0 if file does not exist and -1 on error.
     */
    int cfsExistsExtended(cfsFS fs, const char *path);

    /**
     * cfsSeek - Seek to given offset in file.
     *
     * This works only for files opened in read-only mode.
     *
     * @param fs            The configured filesystem handle.
     * @param file          The file handle.
     * @param desiredPos    Offset into the file to seek into.
     * @return              Returns 0 on success, -1 on error.
     */
    int cfsSeek(cfsFS fs, cfsFile file, tOffset desiredPos);

    /**
     * cfsTell - Get the current offset in the file, in bytes.
     *
     * @param fs    The configured filesystem handle.
     * @param file  The file handle.
     * @return      Current offset, -1 on error.
     */
    tOffset cfsTell(cfsFS fs, cfsFile file);

    /**
     * cfsRead - Read data from an open file.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param buffer    The buffer to copy read bytes into.
     * @param length    The length of the buffer.
     * @return          On success, a positive number indicating how many bytes
     *                  were read.
     *                  On end-of-file, 0.
     *                  On error, -1.  Errno will be set to the error code.
     *                  Just like the POSIX read function, cfsRead will return -1
     *                  and set errno to EINTR if data is temporarily unavailable,
     *                  but we are not yet at the end of the file.
     */
    int64_t cfsRead(cfsFS fs, cfsFile file, void *buffer, uint64_t length);

    /**
     * cfsPread - Positional read of data from an open file.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param position  Position from which to read
     * @param buffer    The buffer to copy read bytes into.
     * @param length    The length of the buffer.
     * @return          See cfsRead
     */
    tSize cfsPread(cfsFS fs, cfsFile file, tOffset position, void *buffer, tSize length);

    /**
     * cfsWrite - Write data into an open file.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @param buffer    The data.
     * @param length    The nymber of bytes to write.
     * @return          Returns the number of bytes written, -1 on error.
     */
    int64_t cfsWrite(cfsFS fs, cfsFile file, const void *buffer, uint64_t length);

    /**
     * cfsWrite - Flush the data.
     *
     * @param fs        The configured filesystem handle.
     * @param file      The file handle.
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsFlush(cfsFS fs, cfsFile file);

    /**
     * cfsHFlush - Flush out the data in client's user buffer. After the
     * return of this call, new readers will see the data.
     *
     * @param fs    configured filesystem handle
     * @param file  file handle
     * @return      0 on success, -1 on error and sets errno
     */
    int cfsHFlush(cfsFS fs, cfsFile file);

    /**
     * cfsSync - Flush out and sync the data in client's user buffer. After the
     * return of this call, new readers will see the data.
     *
     * @param fs    configured filesystem handle
     * @param file  file handle
     * @return      0 on success, -1 on error and sets errno
     */
    int cfsSync(cfsFS fs, cfsFile file);

    /**
     * cfsHSync - Similar to posix fsync, Flush out the data in client's
     * user buffer. all the way to the disk device (but the disk may have
     * it in its cache).
     *
     * @param fs    configured filesystem handle
     * @param file  file handle
     * @return      0 on success, -1 on error and sets errno
     */
    int cfsHSync(cfsFS fs, cfsFile file);

    /**
     * cfsHSyncAndUpdateLength - Similar to cfsHSync, but also update file
     * length in NN.
     *
     * @param fs    configured filesystem handle
     * @param file  file handle
     * @return      0 on success, -1 on error and sets errno
     */
    int cfsHSyncAndUpdateLength(cfsFS fs, cfsFile file);

    /**
     * cfsIsFileClosed - Check is file closed
     *
     * @param fs    The configured filesystem handle.
     * @param path  The path to look for
     * @return      Returns 0 on success, -1 on error and sets errno.
                    errno = 0 means file not closed.
     */
    int cfsIsFileClosed(cfsFS fs, const char *path);

    /**
     * cfsAvailable - Number of bytes that can be read from this
     * input stream without blocking.
     *
     * @param fs    The configured filesystem handle.
     * @param file  The file handle.
     * @return      Returns available bytes; -1 on error.
     */
    int cfsAvailable(cfsFS fs, cfsFile file);

    /**
     * cfsCopy - Copy file from one filesystem to another.
     *
     * @param srcFS     The handle to source filesystem.
     * @param src       The path of source file.
     * @param dstFS     The handle to destination filesystem.
     * @param dst       The path of destination file.
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsCopy(cfsFS srcFS, const char *src, cfsFS dstFS, const char *dst);

    /**
     * cfsMove - Move file from one filesystem to another.
     *
     * @param srcFS     The handle to source filesystem.
     * @param src       The path of source file.
     * @param dstFS     The handle to destination filesystem.
     * @param dst       The path of destination file.
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsMove(cfsFS srcFS, const char *src, cfsFS dstFS, const char *dst);

    /**
     * cfsDelete - Delete file.
     *
     * @param fs        The configured filesystem handle.
     * @param path      The path of the file.
     * @param recursive if path is a directory and set to
     *                  non-zero, the directory is deleted else throws an exception. In
     *                  case of a file the recursive argument is irrelevant.
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsDelete(cfsFS fs, const char *path, int recursive);

    /**
     * cfsRename - Rename file.
     *
     * @param fs        The configured filesystem handle.
     * @param oldPath   The path of the source file.
     * @param newPath   The path of the destination file.
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsRename(cfsFS fs, const char *oldPath, const char *newPath);

    /**
     * cfsRename - Rename file. cfsRename2 allows dst file to be overwrited, while cfsRename doesn't allow.
     *
     * @param fs        The configured filesystem handle.
     * @param oldPath   The path of the source file.
     * @param newPath   The path of the destination file.
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsRename2(cfsFS fs, const char *oldPath, const char *newPath);

    /**
     * cfsConcat - Concatenate (move) the blocks in a list of source
     * files into a single file deleting the source files.  Source
     * files must all have the same block size and replicationand all
     * but the last source file must be an integer number of full
     * blocks long.  The source files are deleted on successful
     * completion.
     *
     * @param fs    The configured filesystem handle.
     * @param trg   The path of target (resulting) file
     * @param scrs  A list of paths to source files
     * @return      Returns 0 on success, -1 on error.
     */
    int cfsConcat(cfsFS fs, const char *trg, const char **srcs);

    /**
     * cfsGetWorkingDirectory - Get the current working directory for
     * the given filesystem.
     *
     * @param fs            The configured filesystem handle.
     * @param buffer        The user-buffer to copy path of cwd into.
     * @param bufferSize    The length of user-buffer.
     * @return              Returns buffer, NULL on error.
     */
    char *cfsGetWorkingDirectory(cfsFS fs, char *buffer, size_t bufferSize);

    /**
     * cfsSetWorkingDirectory - Set the working directory. All relative
     * paths will be resolved relative to it.
     *
     * @param fs    The configured filesystem handle.
     * @param path  The path of the new 'cwd'.
     * @return      Returns 0 on success, -1 on error.
     */
    int cfsSetWorkingDirectory(cfsFS fs, const char *path);

    /**
     * cfsCreateDirectory - Make the given file and all non-existent
     * parents into directories.
     *
     * @param fs    The configured filesystem handle.
     * @param path  The path of the directory.
     * @return      Returns 0 on success, -1 on error.
     */
    int cfsCreateDirectory(cfsFS fs, const char *path);

    /**
     * cfsCreateDirectoryEx - Make the given file with extended options
     *
     * @param fs            The configured filesystem handle.
     * @param path          The path of the directory.
     * @param mode          The permissions for created file and directories.
     * @param createParents Controls whether to create all non-existent parent directories or not
     * @return              Returns 0 on success, -1 on error.
     */
    int cfsCreateDirectoryEx(cfsFS fs, const char *path, short mode, int createParents);

    /**
     * cfsSetReplication - Set the replication of the specified
     * file to the supplied value
     *
     * @param fs            The configured filesystem handle.
     * @param path          The path of the file.
     * @param replication   Block replication.
     * @return              Returns 0 on success, -1 on error.
     */
    int cfsSetReplication(cfsFS fs, const char *path, int16_t replication);

    /**
     * cfsEncryptionZoneInfo- Information about an encryption zone.
     */
    typedef struct
    {
        int mSuite;                 /* the suite of encryption zone */
        int mCryptoProtocolVersion; /* the version of crypto protocol */
        int64_t mId;                /* the id of encryption zone */
        char *mPath;                /* the path of encryption zone */
        char *mKeyName;             /* the key name of encryption zone */
    } cfsEncryptionZoneInfo;

    /**
     * cfsEncryptionFileInfo - Information about an encryption file/directory.
     */
    typedef struct
    {
        int mSuite;                 /* the suite of encryption file/directory */
        int mCryptoProtocolVersion; /* the version of crypto protocol */
        char *mKey;                 /* the key of encryption file/directory */
        char *mKeyName;             /* the key name of encryption file/directory */
        char *mIv;                  /* the iv of encryption file/directory */
        char *mEzKeyVersionName;    /* the version encryption file/directory */
    } cfsEncryptionFileInfo;

    typedef struct cfsBlockLocation
    {
        int numOfNodes;       // Number of Datanodes which keep the block
        bool isCached;        // Replica be cached on Datanodes
        char **hosts;         // Datanode hostnames
        char **names;         // Datanode IP:xferPort for accessing the block
        char **topologyPaths; // Full path name in network topology
        tLong offset;         // Offset of the block in the file
        tLong length;         // block length, may be 0 for the last block
        tSize corrupt;        // If the block is corrupt
    } cfsBlockLocation;

    /**
     * cfsFileInfo - Information about a file/directory.
     */
    typedef struct
    {
        tCfsObjectKind mKind;                          /* file or directory */
        char *mName;                                   /* the name of the file */
        tTime mLastMod;                                /* the last modification time for the file in seconds */
        tOffset mSize;                                 /* the size of the file in bytes */
        short mReplication;                            /* the count of replicas */
        tOffset mBlockSize;                            /* the block size for the file */
        char *mOwner;                                  /* the owner of the file */
        char *mGroup;                                  /* the group associated with the file */
        short mPermissions;                            /* the permissions associated with the file */
        tTime mLastAccess;                             /* the last access time for the file in seconds */
        cfsEncryptionFileInfo *mCfsEncryptionFileInfo; /* the encryption info of the file/directory */
        cfsBlockLocation *mBlocks;      /* LocatedBlock gives information about a block and its location. */
        int mNumOfBlocks;               /* Number of LocatedBlock */
        tCfsObjectAccStatus mAccStatus; /* File or directory status in ACC mode */
    } cfsFileInfo;

    /**
     * cfsListDirectory - Get list of files/directories for a given
     * directory-path. cfsFreeFileInfo should be called to deallocate memory.
     *
     * @param fs            The configured filesystem handle.
     * @param path          The path of the directory.
     * @param numEntries    Set to the number of files/directories in path.
     * @param startAfter    The file/directory path that begin to list in dictionary order.
     *                      When you want to list partition directory entries, remind to
     *                      pass hasRemaining and startAfter in the same time.
     *                      Pass Null to list all directory items
     * @param hasRemaining  Set to 0 if there is no directory entries remained.
     *                      Set to 1 if there are directory entries remained.
     * @return              Returns a dynamically-allocated array of cfsFileInfo
     *                      objects; NULL on error. Specially When directory is empty,
     *                      return pointer to an address with no memry.
     */
    cfsFileInfo *cfsListDirectory(cfsFS fs, const char *path, int *numEntries, const char *startAfter,
                                  int *hasRemaining);

    /**
     * cfsListPath - Get list of files/directories with block locations
     *
     * @param needLocation   if the FileStatus should contain block locations.
     */
    cfsFileInfo *cfsListPath(cfsFS fs, const char *path, int *numEntries, const char *startAfter, int *hasRemaining,
                             bool needLocation);

    /**
     * cfsGetPathInfo - Get information about a path as a (dynamically
     * allocated) single cfsFileInfo struct. cfsFreeFileInfo should be
     * called when the pointer is no longer needed.
     *
     * @param fs    The configured filesystem handle.
     * @param path  The path of the file.
     * @return      Returns a dynamically-allocated cfsFileInfo object;
     *              NULL on error.
     */
    cfsFileInfo *cfsGetPathInfo(cfsFS fs, const char *path);

    /**
     * cfsFileIsEncrypted: determine if a file is encrypted based on its
     * cfsFileInfo.
     *
     * @param cfsFileInfo  The array of dynamically-allocated cfsFileInfo
     *                      objects.
     * @return              -1 if there was an error (errno will be set), 0 if the file is
     *                      not encrypted, 1 if the file is encrypted.
     */
    int cfsFileIsEncrypted(cfsFileInfo *cfsFileInfo);

    /**
     * cfsFreeFileInfo - Free up the cfsFileInfo array (including fields)
     *
     * @param infos         The array of dynamically-allocated cfsFileInfo
     *                      objects.
     * @param numEntries    The size of the array.
     */
    void cfsFreeFileInfo(cfsFileInfo *infos, int numEntries);

    /**
     * cfsFreeEncryptionZoneInfo - Free up the cfsEncryptionZoneInfo array (including fields)
     *
     * @param infos         The array of dynamically-allocated cfsEncryptionZoneInfo
     *                      objects.
     * @param numEntries    The size of the array.
     */
    void cfsFreeEncryptionZoneInfo(cfsEncryptionZoneInfo *infos, int numEntries);

    /**
     * cfsGetHosts - Get hostnames where a particular block (determined by
     * pos & blocksize) of a file is stored. The last element in the array
     * is NULL. Due to replication, a single block could be present on
     * multiple hosts.
     *
     * @param fs        The configured filesystem handle.
     * @param path      The path of the file.
     * @param start     The start of the block.
     * @param length    The length of the block.
     * @return          Returns a dynamically-allocated 2-d array of blocks-hosts;
     *                  NULL on error.
     */
    char ***cfsGetHosts(cfsFS fs, const char *path, tOffset start, tOffset length);

    /**
     * cfsFreeHosts - Free up the structure returned by cfsGetHosts
     *
     * @param blockHosts  The array of dynamically-allocated blocks-hosts.
     */
    void cfsFreeHosts(char ***blockHosts);

    /**
     * cfsGetDefaultBlockSize - Get the default blocksize.
     *
     * @param fs            The configured filesystem handle.
     * @deprecated          Use cfsGetDefaultBlockSizeAtPath instead.
     *
     * @return              Returns the default blocksize, or -1 on error.
     */
    tOffset cfsGetDefaultBlockSize(cfsFS fs);

    /**
     * cfsGetCapacity - Return the raw capacity of the filesystem.
     *
     * @param fs The configured filesystem handle.
     * @return Returns the raw-capacity; -1 on error.
     */
    tOffset cfsGetCapacity(cfsFS fs);

    /**
     * cfsGetUsed - Return the total raw size of all files in the filesystem.
     *
     * @param fs The configured filesystem handle.
     * @return Returns the total-size; -1 on error.
     */
    tOffset cfsGetUsed(cfsFS fs);

    /**
     * Change the user and/or group of a file or directory.
     *
     * @param fs            The configured filesystem handle.
     * @param path          the path to the file or directory
     * @param owner         User string.  Set to NULL for 'no change'
     * @param group         Group string.  Set to NULL for 'no change'
     * @return              0 on success else -1
     */
    int cfsChown(cfsFS fs, const char *path, const char *owner, const char *group);

    /**
     * cfsChmod
     *
     * @param fs        The configured filesystem handle.
     * @param path      the path to the file or directory
     * @param mode      the bitmask to set it to
     * @return          0 on success else -1
     */
    int cfsChmod(cfsFS fs, const char *path, short mode);

    /**
     * cfsUtime
     *
     * @param fs        The configured filesystem handle.
     * @param path      the path to the file or directory
     * @param mtime     new modification time or -1 for no change
     * @param atime     new access time or -1 for no change
     * @return          0 on success else -1
     */
    int cfsUtime(cfsFS fs, const char *path, tTime mtime, tTime atime);

    /**
     * cfsTruncate - Truncate the file in the indicated path to the indicated size.
     *
     * @param fs            The configured filesystem handle.
     * @param path          The path to the file.
     * @param pos           The position the file will be truncated to.
     * @param shouldWait    output value, true if and client does not need to wait for block recovery,
     *                      false if client needs to wait for block recovery.
     * @return              0 on success else -1
     */
    int cfsTruncate(cfsFS fs, const char *path, tOffset pos, int *shouldWait);

    /**
     * cfsGetDelegationToken - Get a delegation token from namenode.
     * The token should be freed using cfsFreeDelegationToken after canceling the token or token expired.
     *
     * @param fs        The file system
     * @param renewer   The user who will renew the token
     *
     * @return          Return a delegation token, NULL on error.
     */
    char *cfsGetDelegationToken(cfsFS fs, const char *renewer);

    /**
     * cfsFreeDelegationToken - Free a delegation token.
     *
     * @param token     The token to be freed.
     */
    void cfsFreeDelegationToken(char *token);

    /**
     * cfsRecoverLease - Recover the lease of the file
     *
     * @param fs        The file system
     * @param path      the path whose lease should be recovered
     *
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsRecoverLease(cfsFS fs, const char *path);

    /**
     * cfsRenewDelegationToken - Renew a delegation token.
     *
     * @param fs        The file system.
     * @param token     The token to be renewed.
     *
     * @return          the new expiration time
     */
    int64_t cfsRenewDelegationToken(cfsFS fs, const char *token);

    /**
     * cfsCancelDelegationToken - Cancel a delegation token.
     *
     * @param fs        The file system.
     * @param token     The token to be canceled.
     *
     * @return          return 0 on success, -1 on error.
     */
    int cfsCancelDelegationToken(cfsFS fs, const char *token);

    typedef struct Namenode
    {
        char *rpc_addr;  // namenode rpc address and port, such as "host:8020"
        char *http_addr; // namenode http address and port, such as "host:50070"
    } Namenode;

    /**
     * cfsGetHANamenodes - If cfs is configured with HA namenode, return all namenode informations as an array.
     * Else return NULL.
     *
     * Using configure file which is given by environment parameter LIBCFS_CONF
     * or "cloudfs.xml" in working directory.
     *
     * @param nameservice   cfs name service id.
     * @param size          output the size of returning array.
     *
     * @return              return an array of all namenode information.
     */
    Namenode *cfsGetHANamenodes(const char *nameservice, int *size);

    /**
     * cfsGetHANamenodesWithConfig - If cfs is configured with HA namenode, return all namenode informations as an
     * array. Else return NULL.
     *
     * @param conf          the path of configure file.
     * @param nameservice   cfs name service id.
     * @param size          output the size of returning array.
     *
     * @return              return an array of all namenode information.
     */
    Namenode *cfsGetHANamenodesWithConfig(const char *conf, const char *nameservice, int *size);

    /**
     * cfsFreeNamenodeInformation - Free the array returned by cfsGetHANamenodesWithConfig()
     *
     * @param nameservice   array return by cfsGetHANamenodesWithConfig()
     * @param size          output the size of returning array.
     */
    void cfsFreeNamenodeInformation(Namenode *namenodes, int size);

    /**
     * cfsFileIsEncrypted - determine if a file is encrypted based on its
     * cfsFileInfo.
     *
     * @param cfsFileInfo  The array of dynamically-allocated cfsFileInfo
     *                      objects.
     * @return              -1 if there was an error (errno will be set), 0 if the file is
     *                      not encrypted, 1 if the file is encrypted.
     */
    int cfsFileIsEncrypted(cfsFileInfo *cfsFileInfo);

    /**
     * cfsGetFileBlockLocations - Get an array containing hostnames,
     * offset and size of portions of the given file.
     *
     * @param fs            The file system
     * @param path          The path to the file
     * @param start         The start offset into the given file
     * @param length        The length for which to get locations for
     * @param numOfBlock    Output the number of elements in the returned array
     *
     * @return An array of BlockLocation struct.
     */
    cfsBlockLocation *cfsGetFileBlockLocations(cfsFS fs, const char *path, tOffset start, tOffset length,
                                               int *numOfBlock);

    /**
     * cfsFreeBlockLocations - Free the BlockLocation array returned
     * by cfsGetFileBlockLocations
     *
     * @param locations     The array returned by cfsGetFileBlockLocations
     * @param numOfBlock    The number of elements in the locaitons
     */
    void cfsFreeBlockLocations(cfsBlockLocation *locations, int numOfBlock);

    typedef struct
    {
        tOffset length;
        tLong fileCount;
        tLong directoryCount;
        tOffset quota;
        tOffset spaceConsumed;
        tOffset spaceQuota;
    } cfsContentSummary;

    /**
     * cfsGetContentSummary - Get the content summary.
     *
     * @param fs        The file system
     * @param path      The path to the file
     * @return          The content summary
     */
    cfsContentSummary *cfsGetContentSummary(cfsFS fs, const char *path);

    /**
     * cfsFreeContentSummary - Free the contentSummary returned by cfsGetContentSummary
     *
     * @param contentSummary The contentSummary returned by cfsGetContentSummary
     */
    void cfsFreeContentSummary(cfsContentSummary *contentSummary);

    /**
     * cfsCreateEncryptionZone - Create encryption zone for the directory with specific key name
     *
     * @param fs        The configured filesystem handle.
     * @param path      The path of the directory.
     * @param keyname   The key name of the encryption zone
     * @return          Returns 0 on success, -1 on error.
     */
    int cfsCreateEncryptionZone(cfsFS fs, const char *path, const char *keyName);

    /**
     * cfsEncryptionZoneInfo - Get information about a path as a (dynamically
     * allocated) single cfsEncryptionZoneInfo struct. cfsEncryptionZoneInfo should be
     * called when the pointer is no longer needed.
     *
     * @param fs        The configured filesystem handle.
     * @param path      The path of the encryption zone.
     * @return          Returns a dynamically-allocated cfsEncryptionZoneInfo object;
     *                  NULL on error.
     */
    cfsEncryptionZoneInfo *cfsGetEZForPath(cfsFS fs, const char *path);

    /**
     * cfsEncryptionZoneInfo -  Get list of all the encryption zones.
     *
     * cfsFreeEncryptionZoneInfo should be called to deallocate memory.
     *
     * @param fs            The configured filesystem handle.
     * @param numEntries    The number of list entries.
     * @return              Returns a dynamically-allocated array of cfsEncryptionZoneInfo objects;
     *                      NULL on error.
     */
    cfsEncryptionZoneInfo *cfsListEncryptionZones(cfsFS fs, int *numEntries);

    /**
     * cfsGetAccFileSystemMode - Get CloudFS Filesystem Mode.
     *
     * @param fs    The configured filesystem handle.
     * @return      Return the FileSystem mode, "HDFS" or "ACC". NULL on error.
     */
    const char *cfsGetFileSystemMode(cfsFS fs);

    /**
     * @cfsGetAccFileSystemUfsPrefix - Get CloudFS underlying file system prefix
     * or working directory.
     *
     * @param fs    The configured filesystem handle.
     * @return      Return the underlying file system prefix in ACC mode. NULL on
     *              error.
     */
    const char *cfsGetAccFileSystemUfsPrefix(cfsFS fs);

    /**
     * @cfsSetCredentials - Set CloudFS credentials for IAM. Data is shadow copied.
     *                      They must be available until bld is freed.
     *
     * @param bld           The CFS builder
     * @param accessKey     The access key ID.
     * @param secretKey     The secret access key.
     * @param securityToken The security token.
     * @return              Returns 0 on success, -1 on error.
     */
    int cfsSetCredentials(struct cfsBuilder *bld, const char *accessKey, const char *secretKey,
                          const char *securityToken);

    /**
     * Copy data from the under storage system into Datanode
     * @param fs    The configured filesystem handle.
     * @param path the path which data will be copied
     * @param recursive recursively load in subdirectories
     * @param loadMetadata load metadata from UFS to Namenode
     * @param loadData load data from UFS to Datanode
     * @param replicaNum load replica number
     * @param dataCenter data be loaded
     * @return Job status
     */
    char *cfsLoad(cfsFS fs, const char *path, bool recursive, bool loadMetadata, bool loadData, int replicaNum,
                  const char *dcName, int dcId);

    /**
     * Delete data from Datanode
     * @param fs    The configured filesystem handle.
     * @param path the path which data will be deleted
     * @param recursive recursively free in subdirectories
     * @param freeMetadata free metadata
     * @return Job status
     */
    char *cfsFree(cfsFS fs, const char *path, bool recursive, bool freeMetadata);

    typedef enum tJobType
    {
        kUnknownJob = 0,
        kLoadDataJob = 1,
        kLoadMetadataJob = 2,
        kFreeJob = 3,
    } tJobType;

    typedef enum tJobStatus
    {
        kAccepted = 0,
        kSubmitted = 1,
        kRunning = 2,
        kFinished = 3,
        kCancelled = 4,
        kFailed = 5
    } tJobStatus;

    typedef struct
    {
        bool done;
        tJobType jobType;
        uint64_t createTimestamp;
        uint64_t completeTimestamp;
        uint64_t successTasks;
        uint64_t failedTasks;
        uint64_t canceledTasks;
        uint64_t timeoutTasks;
        uint64_t throttledTasks;
        uint64_t totalTasks;
    } cfsJobState;

    typedef struct
    {
        cfsJobState *jobStates;
        int numEntries;
        tJobStatus jobStatus;
        char *msg;
    } cfsLookupJobResponse;

    /**
     * Lookup job status
     * @param fs    The configured filesystem handle.
     * @param job_id
     * @return Job status
     */
    cfsLookupJobResponse cfsLookupJob(cfsFS fs, const char *job_id);

    void cfsFreeLookupResp(cfsLookupJobResponse *resp);

    /**
     * Cancel job
     * @param fs   The configured filesystem handle.
     * @param job_id
     */
    void cfsCancelJob(cfsFS fs, const char *job_id);

#ifdef __cplusplus
}
#endif

#endif /* _CLOUDFS_LIBCFS3_CLIENT_CFS_H_ */
