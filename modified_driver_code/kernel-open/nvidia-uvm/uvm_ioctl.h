/*******************************************************************************
    Copyright (c) 2013-2023 NVidia Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef _UVM_IOCTL_H
#define _UVM_IOCTL_H

#include "uvm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Please see the header file (uvm.h) for detailed documentation on each of the
// associated API calls.
//

#if defined(WIN32) || defined(WIN64)
#   define UVM_IOCTL_BASE(i)       CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800+i, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
#else
#   define UVM_IOCTL_BASE(i) i
#endif

//
// UvmReserveVa
//
#define UVM_RESERVE_VA                                                UVM_IOCTL_BASE(1)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_RESERVE_VA_PARAMS;

//
// UvmReleaseVa
//
#define UVM_RELEASE_VA                                                UVM_IOCTL_BASE(2)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_RELEASE_VA_PARAMS;

//
// UvmRegionCommit
//
#define UVM_REGION_COMMIT                                             UVM_IOCTL_BASE(3)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64           length        NV_ALIGN_BYTES(8); // IN
    UvmStream       streamId      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid gpuUuid;                         // IN
    NV_STATUS       rmStatus;                        // OUT
} UVM_REGION_COMMIT_PARAMS;

//
// UvmRegionDecommit
//
#define UVM_REGION_DECOMMIT                                           UVM_IOCTL_BASE(4)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_REGION_DECOMMIT_PARAMS;

//
// UvmRegionSetStream
//
#define UVM_REGION_SET_STREAM                                         UVM_IOCTL_BASE(5)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64           length        NV_ALIGN_BYTES(8); // IN
    UvmStream       newStreamId   NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid gpuUuid;                         // IN
    NV_STATUS       rmStatus;                        // OUT
} UVM_REGION_SET_STREAM_PARAMS;

//
// UvmSetStreamRunning
//
#define UVM_SET_STREAM_RUNNING                                        UVM_IOCTL_BASE(6)

typedef struct
{
    UvmStream  streamId NV_ALIGN_BYTES(8);  // IN
    NV_STATUS  rmStatus;                    // OUT
} UVM_SET_STREAM_RUNNING_PARAMS;


//
// Due to limitations in how much we want to send per ioctl call, the nStreams
// member must be less than or equal to about 250. That's an upper limit.
//
// However, from a typical user-space driver's point of view (for example, the
// CUDA driver), a vast majority of the time, we expect there to be only one
// stream passed in. The second most common case is something like atmost 32
// streams being passed in. The cases where there are more than 32 streams are
// the most rare. So we might want to optimize the ioctls accordingly so that we
// don't always copy a 250 * sizeof(streamID) sized array when there's only one
// or a few streams.
//
// For that reason, UVM_MAX_STREAMS_PER_IOCTL_CALL is set to 32.
//
// If the higher-level (uvm.h) call requires more streams to be stopped than
// this value, then multiple ioctl calls should be made.
//
#define UVM_MAX_STREAMS_PER_IOCTL_CALL 32

//
// UvmSetStreamStopped
//
#define UVM_SET_STREAM_STOPPED                                        UVM_IOCTL_BASE(7)

typedef struct
{
    UvmStream streamIdArray[UVM_MAX_STREAMS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     nStreams                                      NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                        // OUT
} UVM_SET_STREAM_STOPPED_PARAMS;

//
// UvmCallTestFunction
//
#define UVM_RUN_TEST                                                  UVM_IOCTL_BASE(9)

typedef struct
{
    NvProcessorUuid gpuUuid;     // IN
    NvU32           test;        // IN
    struct
    {
        NvProcessorUuid peerGpuUuid; // IN
        NvU32           peerId;      // IN
    } multiGpu;
    NV_STATUS      rmStatus;    // OUT
} UVM_RUN_TEST_PARAMS;

//
// This is a magic offset for mmap. Any mapping of an offset above this
// threshold will be treated as a counters mapping, not as an allocation
// mapping. Since allocation offsets must be identical to the virtual address
// of the mapping, this threshold has to be an offset that cannot be
// a valid virtual address.
//
#if defined(__linux__)
    #if defined(NV_64_BITS)
        #define UVM_EVENTS_OFFSET_BASE   (1UL << 63)
        #define UVM_COUNTERS_OFFSET_BASE (1UL << 62)
    #else
        #define UVM_EVENTS_OFFSET_BASE   (1UL << 31)
        #define UVM_COUNTERS_OFFSET_BASE (1UL << 30)
    #endif
#endif // defined(__linux___)

//
// UvmAddSession
//
#define UVM_ADD_SESSION                                               UVM_IOCTL_BASE(10)

typedef struct
{
    NvU32        pidTarget;                             // IN
#ifdef __linux__
    NvP64        countersBaseAddress NV_ALIGN_BYTES(8); // IN
    NvS32        sessionIndex;                          // OUT (session index that got added)
#endif
    NV_STATUS    rmStatus;                              // OUT
} UVM_ADD_SESSION_PARAMS;

//
// UvmRemoveSession
//
#define UVM_REMOVE_SESSION                                             UVM_IOCTL_BASE(11)

typedef struct
{
#ifdef __linux__
    NvS32        sessionIndex; // IN (session index to be removed)
#endif
    NV_STATUS    rmStatus;     // OUT
} UVM_REMOVE_SESSION_PARAMS;


#define UVM_MAX_COUNTERS_PER_IOCTL_CALL 32

//
// UvmEnableCounters
//
#define UVM_ENABLE_COUNTERS                                           UVM_IOCTL_BASE(12)

typedef struct
{
#ifdef __linux__
    NvS32            sessionIndex;                            // IN
#endif
    UvmCounterConfig config[UVM_MAX_COUNTERS_PER_IOCTL_CALL]; // IN
    NvU32            count;                                   // IN
    NV_STATUS        rmStatus;                                // OUT
} UVM_ENABLE_COUNTERS_PARAMS;

//
// UvmMapCounter
//
#define UVM_MAP_COUNTER                                               UVM_IOCTL_BASE(13)

typedef struct
{
#ifdef __linux__
    NvS32           sessionIndex;                   // IN
#endif
    NvU32           scope;                          // IN (UvmCounterScope)
    NvU32           counterName;                    // IN (UvmCounterName)
    NvProcessorUuid gpuUuid;                        // IN
    NvP64           addr         NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                       // OUT
} UVM_MAP_COUNTER_PARAMS;

//
// UvmCreateEventQueue
//
#define UVM_CREATE_EVENT_QUEUE                                        UVM_IOCTL_BASE(14)

typedef struct
{
#ifdef __linux__
    NvS32                 sessionIndex;                         // IN
#endif
    NvU32                 eventQueueIndex;                      // OUT
    NvU64                 queueSize          NV_ALIGN_BYTES(8); // IN
    NvU64                 notificationCount  NV_ALIGN_BYTES(8); // IN
#if defined(WIN32) || defined(WIN64)
    NvU64                 notificationHandle NV_ALIGN_BYTES(8); // IN
#endif
    NvU32                 timeStampType;                        // IN (UvmEventTimeStampType)
    NV_STATUS             rmStatus;                             // OUT
} UVM_CREATE_EVENT_QUEUE_PARAMS;

//
// UvmRemoveEventQueue
//
#define UVM_REMOVE_EVENT_QUEUE                                        UVM_IOCTL_BASE(15)

typedef struct
{
#ifdef __linux__
    NvS32         sessionIndex;       // IN
#endif
    NvU32         eventQueueIndex;    // IN
    NV_STATUS     rmStatus;           // OUT
} UVM_REMOVE_EVENT_QUEUE_PARAMS;

//
// UvmMapEventQueue
//
#define UVM_MAP_EVENT_QUEUE                                           UVM_IOCTL_BASE(16)

typedef struct
{
#ifdef __linux__
    NvS32         sessionIndex;                       // IN
#endif
    NvU32         eventQueueIndex;                    // IN
    NvP64         userRODataAddr   NV_ALIGN_BYTES(8); // IN
    NvP64         userRWDataAddr   NV_ALIGN_BYTES(8); // IN
    NvP64         readIndexAddr    NV_ALIGN_BYTES(8); // OUT
    NvP64         writeIndexAddr   NV_ALIGN_BYTES(8); // OUT
    NvP64         queueBufferAddr  NV_ALIGN_BYTES(8); // OUT
    NV_STATUS     rmStatus;                           // OUT
} UVM_MAP_EVENT_QUEUE_PARAMS;

//
// UvmEnableEvent
//
#define UVM_EVENT_CTRL                                                UVM_IOCTL_BASE(17)

typedef struct
{
#ifdef __linux__
    NvS32        sessionIndex;      // IN
#endif
    NvU32        eventQueueIndex;   // IN
    NvS32        eventType;         // IN
    NvU32        enable;            // IN
    NV_STATUS    rmStatus;          // OUT
} UVM_EVENT_CTRL_PARAMS;

//
// UvmRegisterMpsServer
//
#define UVM_REGISTER_MPS_SERVER                                       UVM_IOCTL_BASE(18)

typedef struct
{
    NvProcessorUuid gpuUuidArray[UVM_MAX_GPUS_V1];                 // IN
    NvU32           numGpus;                                       // IN
    NvU64           serverId                    NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                                      // OUT
} UVM_REGISTER_MPS_SERVER_PARAMS;

//
// UvmRegisterMpsClient
//
#define UVM_REGISTER_MPS_CLIENT                                       UVM_IOCTL_BASE(19)

typedef struct
{
    NvU64     serverId  NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_REGISTER_MPS_CLIENT_PARAMS;

//
// UvmEventGetGpuUuidTable
//
#define UVM_GET_GPU_UUID_TABLE                                        UVM_IOCTL_BASE(20)

typedef struct
{
    NvProcessorUuid gpuUuidArray[UVM_MAX_GPUS_V1]; // OUT
    NvU32           validCount;                    // OUT
    NV_STATUS       rmStatus;                      // OUT
} UVM_GET_GPU_UUID_TABLE_PARAMS;

#if defined(WIN32) || defined(WIN64)
//
// UvmRegionSetBacking
//
#define UVM_REGION_SET_BACKING                                        UVM_IOCTL_BASE(21)

typedef struct
{
    NvProcessorUuid gpuUuid;                        // IN
    NvU32           hAllocation;                    // IN
    NvP64           vaAddr       NV_ALIGN_BYTES(8); // IN
    NvU64           regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                       // OUT
} UVM_REGION_SET_BACKING_PARAMS;

//
// UvmRegionUnsetBacking
//
#define UVM_REGION_UNSET_BACKING                                      UVM_IOCTL_BASE(22)

typedef struct
{
    NvP64     vaAddr       NV_ALIGN_BYTES(8); // IN
    NvU64     regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                       // OUT
} UVM_REGION_UNSET_BACKING_PARAMS;

#endif

#define UVM_CREATE_RANGE_GROUP                                        UVM_IOCTL_BASE(23)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                       // OUT
} UVM_CREATE_RANGE_GROUP_PARAMS;

#define UVM_DESTROY_RANGE_GROUP                                       UVM_IOCTL_BASE(24)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                      // OUT
} UVM_DESTROY_RANGE_GROUP_PARAMS;

//
// UvmRegisterGpuVaSpace
//
#define UVM_REGISTER_GPU_VASPACE                                      UVM_IOCTL_BASE(25)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NvS32           rmCtrlFd; // IN
    NvHandle        hClient;  // IN
    NvHandle        hVaSpace; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_REGISTER_GPU_VASPACE_PARAMS;

//
// UvmUnregisterGpuVaSpace
//
#define UVM_UNREGISTER_GPU_VASPACE                                    UVM_IOCTL_BASE(26)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_VASPACE_PARAMS;

//
// UvmRegisterChannel
//
#define UVM_REGISTER_CHANNEL                                          UVM_IOCTL_BASE(27)

typedef struct
{
    NvProcessorUuid gpuUuid;                     // IN
    NvS32           rmCtrlFd;                    // IN
    NvHandle        hClient;                     // IN
    NvHandle        hChannel;                    // IN
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_REGISTER_CHANNEL_PARAMS;

//
// UvmUnregisterChannel
//
#define UVM_UNREGISTER_CHANNEL                                        UVM_IOCTL_BASE(28)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NvHandle        hClient;  // IN
    NvHandle        hChannel; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_CHANNEL_PARAMS;

//
// UvmEnablePeerAccess
//
#define UVM_ENABLE_PEER_ACCESS                                       UVM_IOCTL_BASE(29)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_ENABLE_PEER_ACCESS_PARAMS;

//
// UvmDisablePeerAccess
//
#define UVM_DISABLE_PEER_ACCESS                                      UVM_IOCTL_BASE(30)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_DISABLE_PEER_ACCESS_PARAMS;

//
// UvmSetRangeGroup
//
#define UVM_SET_RANGE_GROUP                                           UVM_IOCTL_BASE(31)

typedef struct
{
    NvU64     rangeGroupId  NV_ALIGN_BYTES(8); // IN
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_SET_RANGE_GROUP_PARAMS;

//
// UvmMapExternalAllocation
//
#define UVM_MAP_EXTERNAL_ALLOCATION                                   UVM_IOCTL_BASE(33)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvU64                   offset                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS_V2];                 // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NvS32                   rmCtrlFd;                                          // IN
    NvU32                   hClient;                                           // IN
    NvU32                   hMemory;                                           // IN

    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;

//
// UvmFree
//
#define UVM_FREE                                                      UVM_IOCTL_BASE(34)
typedef struct
{
    NvU64     base      NV_ALIGN_BYTES(8); // IN
    NvU64     length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_FREE_PARAMS;

//
// UvmMemMap
//
#define UVM_MEM_MAP                                                   UVM_IOCTL_BASE(35)

typedef struct
{
    NvP64     regionBase   NV_ALIGN_BYTES(8); // IN
    NvU64     regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                       // OUT
} UVM_MEM_MAP_PARAMS;

//
// UvmDebugAccessMemory
//
#define UVM_DEBUG_ACCESS_MEMORY                                       UVM_IOCTL_BASE(36)

typedef struct
{
#ifdef __linux__
    NvS32               sessionIndex;                    // IN
#endif
    NvU64               baseAddress   NV_ALIGN_BYTES(8); // IN
    NvU64               sizeInBytes   NV_ALIGN_BYTES(8); // IN
    NvU32               accessType;                      // IN (UvmDebugAccessType)
    NvU64               buffer        NV_ALIGN_BYTES(8); // IN/OUT
    NvBool              isBitmaskSet;                    // OUT
    NvU64               bitmask       NV_ALIGN_BYTES(8); // IN/OUT
    NV_STATUS           rmStatus;                        // OUT
} UVM_DEBUG_ACCESS_MEMORY_PARAMS;

//
// UvmRegisterGpu
//
#define UVM_REGISTER_GPU                                              UVM_IOCTL_BASE(37)

typedef struct
{
    NvProcessorUuid gpu_uuid;    // IN/OUT
    NvBool          numaEnabled; // OUT
    NvS32           numaNodeId;  // OUT
    NvS32           rmCtrlFd;    // IN
    NvHandle        hClient;     // IN
    NvHandle        hSmcPartRef; // IN
    NV_STATUS       rmStatus;    // OUT
} UVM_REGISTER_GPU_PARAMS;

//
// UvmUnregisterGpu
//
#define UVM_UNREGISTER_GPU                                            UVM_IOCTL_BASE(38)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_PARAMS;

#define UVM_PAGEABLE_MEM_ACCESS                                       UVM_IOCTL_BASE(39)

typedef struct
{
    NvBool    pageableMemAccess; // OUT
    NV_STATUS rmStatus;          // OUT
} UVM_PAGEABLE_MEM_ACCESS_PARAMS;

//
// Due to limitations in how much we want to send per ioctl call, the numGroupIds
// member must be less than or equal to about 250. That's an upper limit.
//
// However, from a typical user-space driver's point of view (for example, the
// CUDA driver), a vast majority of the time, we expect there to be only one
// range group passed in. The second most common case is something like atmost 32
// range groups being passed in. The cases where there are more than 32 range
// groups are the most rare. So we might want to optimize the ioctls accordingly
// so that we don't always copy a 250 * sizeof(NvU64) sized array when there's
// only one or a few range groups.
//
// For that reason, UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL is set to 32.
//
// If the higher-level (uvm.h) call requires more range groups than
// this value, then multiple ioctl calls should be made.
//
#define UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL 32

//
// UvmPreventMigrationRangeGroups
//
#define UVM_PREVENT_MIGRATION_RANGE_GROUPS                            UVM_IOCTL_BASE(40)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                             // OUT
} UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS;

//
// UvmAllowMigrationRangeGroups
//
#define UVM_ALLOW_MIGRATION_RANGE_GROUPS                              UVM_IOCTL_BASE(41)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                             // OUT
} UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS;

//
// UvmSetPreferredLocation
//
#define UVM_SET_PREFERRED_LOCATION                                    UVM_IOCTL_BASE(42)

typedef struct
{
    NvU64           requestedBase      NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid preferredLocation;                    // IN
    NvS32           preferredCpuNumaNode;                 // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_SET_PREFERRED_LOCATION_PARAMS;

//
// UvmUnsetPreferredLocation
//
#define UVM_UNSET_PREFERRED_LOCATION                                  UVM_IOCTL_BASE(43)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_UNSET_PREFERRED_LOCATION_PARAMS;

//
// UvmEnableReadDuplication
//
#define UVM_ENABLE_READ_DUPLICATION                                   UVM_IOCTL_BASE(44)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_ENABLE_READ_DUPLICATION_PARAMS;

//
// UvmDisableReadDuplication
//
#define UVM_DISABLE_READ_DUPLICATION                                  UVM_IOCTL_BASE(45)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_DISABLE_READ_DUPLICATION_PARAMS;

//
// UvmSetAccessedBy
//
#define UVM_SET_ACCESSED_BY                                           UVM_IOCTL_BASE(46)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
    NvU64           length          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid accessedByUuid;                    // IN
    NV_STATUS       rmStatus;                          // OUT
} UVM_SET_ACCESSED_BY_PARAMS;

//
// UvmUnsetAccessedBy
//
#define UVM_UNSET_ACCESSED_BY                                         UVM_IOCTL_BASE(47)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
    NvU64           length          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid accessedByUuid;                    // IN
    NV_STATUS       rmStatus;                          // OUT
} UVM_UNSET_ACCESSED_BY_PARAMS;

// For managed allocations, UVM_MIGRATE implements the behavior described in
// UvmMigrate. If the input virtual range corresponds to system-allocated
// pageable memory, and the GPUs in the system support transparent access to
// pageable memory, the scheme is a bit more elaborate, potentially with
// several transitions betwen user and kernel spaces:
//
// 1) UVM_MIGRATE with the range base address and size. This will migrate
// anonymous vmas until:
//   a) It finds a file-backed vma or no GPUs are registered in the VA space
//      so no GPU can drive the copy. It will try to populate the vma using
//      get_user_pages and return NV_WARN_NOTHING_TO_DO.
//   b) It fails to allocate memory on the destination CPU node. It will return
//       NV_ERR_MORE_PROCESSING_REQUIRED.
//   c) It fails to populate pages directly on the destination GPU. It will try
//      to populate the vma using get_user_pages and return.
//   d) The full input range is migrated (or empty), this call will release
//      the semaphore before returning.
// 2) The user-mode needs to handle the following error codes:
//   a) NV_WARN_NOTHING_TO_DO: use move_pages to migrate pages for the VA
//      range corresponding to the vma that couldn't be migrated in kernel
//      mode. Then, it processes the remainder of the range, starting after
//      that vma.
//   b) NV_ERR_MORE_PROCESSING_REQUIRED: choose a different CPU NUMA node,
//      trying to enforce the NUMA policies of the thread and retry the
//      ioctl. If there are no more CPU NUMA nodes to try, try to populate
//      the remainder of the range anywhere using the UVM_POPULATE_PAGEABLE
//      ioctl.
//   c) NV_OK: success. This only guarantees that pages were populated, not
//      that they moved to the requested destination.
// 3) For cases 2.a) and 2.b) Goto 1
//
// If UVM_MIGRATE_FLAG_ASYNC is 0, the ioctl won't return until the migration is
// done and all mappings are updated, subject to the special rules for pageable
// memory described above. semaphoreAddress must be 0. semaphorePayload is
// ignored.
//
// If UVM_MIGRATE_FLAG_ASYNC is 1, the ioctl may return before the migration is
// complete. If semaphoreAddress is 0, semaphorePayload is ignored and no
// notification will be given on completion. If semaphoreAddress is non-zero
// and the returned error code is NV_OK, semaphorePayload will be written to
// semaphoreAddress once the migration is complete.
#define UVM_MIGRATE_FLAG_ASYNC              0x00000001

// When the migration destination is the CPU, skip the step which creates new
// virtual mappings on the CPU. Creating CPU mappings must wait for the
// migration to complete, so skipping this step allows the migration to be
// fully asynchronous. This flag is ignored for pageable migrations if the GPUs
// in the system support transparent access to pageable memory.
//
// The UVM driver must have builtin tests enabled for the API to use this flag.
#define UVM_MIGRATE_FLAG_SKIP_CPU_MAP       0x00000002

// By default UVM_MIGRATE returns an error if the destination UUID is a GPU
// without a registered GPU VA space. Setting this flag skips that check, so the
// destination GPU only needs to have been registered.
//
// This can be used in tests to trigger migrations of physical memory without
// the overhead of GPU PTE mappings.
//
// The UVM driver must have builtin tests enabled for the API to use this flag.
#define UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE    0x00000004

#define UVM_MIGRATE_FLAGS_TEST_ALL              (UVM_MIGRATE_FLAG_SKIP_CPU_MAP      | \
                                                 UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE)

#define UVM_MIGRATE_FLAGS_ALL                   (UVM_MIGRATE_FLAG_ASYNC | \
                                                 UVM_MIGRATE_FLAGS_TEST_ALL)

// If NV_ERR_INVALID_ARGUMENT is returned it is because cpuMemoryNode is not
// valid and the destination processor is the CPU. cpuMemoryNode is considered
// invalid if:
//      * it is less than -1,
//      * it is equal to or larger than the maximum number of nodes, or
//      * it corresponds to a registered GPU.
//      * it is not in the node_possible_map set of nodes,
//      * it does not have onlined memory
//
// For pageable migrations:
//
// In addition to the above, in the case of pageable memory, the
// cpuMemoryNode is considered invalid if it's -1.
//
// If NV_WARN_NOTHING_TO_DO is returned, user-space is responsible for
// completing the migration of the VA range described by userSpaceStart and
// userSpaceLength using move_pages.
//
// If NV_ERR_MORE_PROCESSING_REQUIRED is returned, user-space is responsible
// for re-trying with a different cpuNumaNode, starting at userSpaceStart.
//
#define UVM_MIGRATE                                                   UVM_IOCTL_BASE(51)
typedef struct
{
    NvU64           base               NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NvU32           flags;                                // IN
    NvU64           semaphoreAddress   NV_ALIGN_BYTES(8); // IN
    NvU32           semaphorePayload;                     // IN
    NvS32           cpuNumaNode;                          // IN
    NvU64           userSpaceStart     NV_ALIGN_BYTES(8); // OUT
    NvU64           userSpaceLength    NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_PARAMS;

#define UVM_MIGRATE_RANGE_GROUP                                       UVM_IOCTL_BASE(53)
typedef struct
{
    NvU64           rangeGroupId       NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_RANGE_GROUP_PARAMS;

//
// UvmEnableSystemWideAtomics
//
#define UVM_ENABLE_SYSTEM_WIDE_ATOMICS                                UVM_IOCTL_BASE(54)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS;

//
// UvmDisableSystemWideAtomics
//
#define UVM_DISABLE_SYSTEM_WIDE_ATOMICS                               UVM_IOCTL_BASE(55)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS;

//
// Initialize any tracker object such as a queue or counter
// UvmToolsCreateEventQueue, UvmToolsCreateProcessAggregateCounters,
// UvmToolsCreateProcessorCounters.
// Note that the order of structure elements has the version as the last field.
// This is used to tell whether the kernel supports V2 events or not because
// the V1 UVM_TOOLS_INIT_EVENT_TRACKER ioctl would not read or update that
// field but V2 will. This is needed because it is possible to create an event
// queue before CUDA is initialized which means UvmSetDriverVersion() hasn't
// been called yet and the kernel version is unknown.
//
#define UVM_TOOLS_INIT_EVENT_TRACKER                                  UVM_IOCTL_BASE(56)
typedef struct
{
    NvU64           queueBuffer        NV_ALIGN_BYTES(8); // IN
    NvU64           queueBufferSize    NV_ALIGN_BYTES(8); // IN
    NvU64           controlBuffer      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid processor;                            // IN
    NvU32           allProcessors;                        // IN
    NvU32           uvmFd;                                // IN
    NV_STATUS       rmStatus;                             // OUT
    NvU32           requestedVersion;                     // IN
    NvU32           grantedVersion;                       // OUT
} UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS;

//
// UvmToolsSetNotificationThreshold
//
#define UVM_TOOLS_SET_NOTIFICATION_THRESHOLD                          UVM_IOCTL_BASE(57)
typedef struct
{
    NvU32     notificationThreshold;                       // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS;

//
// UvmToolsEventQueueEnableEvents
//
#define UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS                           UVM_IOCTL_BASE(58)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS;

//
// UvmToolsEventQueueDisableEvents
//
#define UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS                          UVM_IOCTL_BASE(59)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS;

//
// UvmToolsEnableCounters
//
#define UVM_TOOLS_ENABLE_COUNTERS                                     UVM_IOCTL_BASE(60)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_ENABLE_COUNTERS_PARAMS;

//
// UvmToolsDisableCounters
//
#define UVM_TOOLS_DISABLE_COUNTERS                                    UVM_IOCTL_BASE(61)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_DISABLE_COUNTERS_PARAMS;

//
// UvmToolsReadProcessMemory
//
#define UVM_TOOLS_READ_PROCESS_MEMORY                                 UVM_IOCTL_BASE(62)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8); // IN
    NvU64     size                      NV_ALIGN_BYTES(8); // IN
    NvU64     targetVa                  NV_ALIGN_BYTES(8); // IN
    NvU64     bytesRead                 NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS;

//
// UvmToolsWriteProcessMemory
//
#define UVM_TOOLS_WRITE_PROCESS_MEMORY                                UVM_IOCTL_BASE(63)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8); // IN
    NvU64     size                      NV_ALIGN_BYTES(8); // IN
    NvU64     targetVa                  NV_ALIGN_BYTES(8); // IN
    NvU64     bytesWritten              NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS;

//
// UvmToolsGetProcessorUuidTable
// Note that tablePtr != 0 and count == 0 means that tablePtr is assumed to be
// an array of size UVM_MAX_PROCESSORS_V1 and that only UvmEventEntry_V1
// processor IDs (physical GPU UUIDs) will be reported.
// tablePtr == 0 and count == 0 can be used to query how many processors are
// present in order to dynamically allocate the correct size array since the
// total number of processors is returned in 'count'.
//
#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE                            UVM_IOCTL_BASE(64)
typedef struct
{
    NvU64     tablePtr                 NV_ALIGN_BYTES(8); // IN
    NvU32     count;                                      // IN/OUT
    NV_STATUS rmStatus;                                   // OUT
    NvU32     version;                                    // OUT
} UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS;


//
// UvmMapDynamicParallelismRegion
//
#define UVM_MAP_DYNAMIC_PARALLELISM_REGION                            UVM_IOCTL_BASE(65)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS;

//
// UvmUnmapExternal
//
#define UVM_UNMAP_EXTERNAL                                            UVM_IOCTL_BASE(66)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_UNMAP_EXTERNAL_PARAMS;


//
// UvmToolsFlushEvents
//
#define UVM_TOOLS_FLUSH_EVENTS                                        UVM_IOCTL_BASE(67)
typedef struct
{
    NV_STATUS rmStatus;                                   // OUT
} UVM_TOOLS_FLUSH_EVENTS_PARAMS;

//
// UvmAllocSemaphorePool
//
#define UVM_ALLOC_SEMAPHORE_POOL                                     UVM_IOCTL_BASE(68)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS_V2];                 // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_ALLOC_SEMAPHORE_POOL_PARAMS;

//
// UvmCleanUpZombieResources
//
#define UVM_CLEAN_UP_ZOMBIE_RESOURCES                                 UVM_IOCTL_BASE(69)
typedef struct
{
    NV_STATUS rmStatus;                    // OUT
} UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS;

//
// UvmIsPageableMemoryAccessSupportedOnGpu
//
#define UVM_PAGEABLE_MEM_ACCESS_ON_GPU                                UVM_IOCTL_BASE(70)

typedef struct
{
    NvProcessorUuid gpu_uuid;          // IN
    NvBool          pageableMemAccess; // OUT
    NV_STATUS       rmStatus;          // OUT
} UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS;

//
// UvmPopulatePageable
//
#define UVM_POPULATE_PAGEABLE                                         UVM_IOCTL_BASE(71)

// Allow population of managed ranges.
//
// The UVM driver must have builtin tests enabled for the API to use the
// following two flags.
#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED              0x00000001

// By default UVM_POPULATE_PAGEABLE returns an error if the destination vma
// does not have read permission. This flag skips that check.
#define UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK            0x00000002

#define UVM_POPULATE_PAGEABLE_FLAGS_TEST_ALL    (UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED | \
                                                 UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK)

#define UVM_POPULATE_PAGEABLE_FLAGS_ALL         UVM_POPULATE_PAGEABLE_FLAGS_TEST_ALL

typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NvU32           flags;                       // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_POPULATE_PAGEABLE_PARAMS;

//
// UvmValidateVaRange
//
#define UVM_VALIDATE_VA_RANGE                                         UVM_IOCTL_BASE(72)
typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_VALIDATE_VA_RANGE_PARAMS;

#define UVM_CREATE_EXTERNAL_RANGE                                     UVM_IOCTL_BASE(73)
typedef struct
{
    NvU64                  base                             NV_ALIGN_BYTES(8); // IN
    NvU64                  length                           NV_ALIGN_BYTES(8); // IN
    NV_STATUS              rmStatus;                                           // OUT
} UVM_CREATE_EXTERNAL_RANGE_PARAMS;

#define UVM_MAP_EXTERNAL_SPARSE                                       UVM_IOCTL_BASE(74)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_SPARSE_PARAMS;

//
// Used to initialise a secondary UVM file-descriptor which holds a
// reference on the memory map to prevent it being torn down without
// first notifying UVM. This is achieved by preventing mmap() calls on
// the secondary file-descriptor so that on process exit
// uvm_mm_release() will be called while the memory map is present
// such that UVM can cleanly shutdown the GPU by handling faults
// instead of cancelling them.
//
// This ioctl must be called after the primary file-descriptor has
// been initialised with the UVM_INITIALIZE ioctl. The primary FD
// should be passed in the uvmFd field and the UVM_MM_INITIALIZE ioctl
// will hold a reference on the primary FD. Therefore uvm_release() is
// guaranteed to be called after uvm_mm_release().
//
// Once this file-descriptor has been closed the UVM context is
// effectively dead and subsequent operations requiring a memory map
// will fail. Calling UVM_MM_INITIALIZE on a context that has already
// been initialized via any FD will return NV_ERR_INVALID_STATE.
//
// Calling this with a non-UVM file-descriptor in uvmFd will return
// NV_ERR_INVALID_ARGUMENT. Calling this on the same file-descriptor
// as UVM_INITIALIZE or more than once on the same FD will return
// NV_ERR_IN_USE.
//
// Not all platforms require this secondary file-descriptor. On those
// platforms NV_WARN_NOTHING_TO_DO will be returned and users may
// close the file-descriptor at anytime.
#define UVM_MM_INITIALIZE                                             UVM_IOCTL_BASE(75)
typedef struct
{
    NvS32                   uvmFd;    // IN
    NV_STATUS               rmStatus; // OUT
} UVM_MM_INITIALIZE_PARAMS;

//
// Temporary ioctls which should be removed before UVM 8 release
// Number backwards from 2047 - highest custom ioctl function number
// windows can handle.
//

//
// UvmIs8Supported
//
#define UVM_IS_8_SUPPORTED                                            UVM_IOCTL_BASE(2047)

typedef struct
{
    NvU32     is8Supported; // OUT
    NV_STATUS rmStatus;     // OUT
} UVM_IS_8_SUPPORTED_PARAMS;

//
// UvmLiveMigrationFinalize
// Finalizes migration, cleans up duplicate state
//
#define UVM_LIVE_MIGRATION_FINALIZE                                   UVM_IOCTL_BASE(101)

typedef struct
{
    NvU64     base          NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_LIVE_MIGRATION_FINALIZE_PARAMS;

//
// UvmLiveMigrationPrepareAll
// System-wide migration prepare - affects ALL processes (requires CAP_SYS_ADMIN)
// Called by migration agent to prepare entire system for VM migration
//
#define UVM_LIVE_MIGRATION_PREPARE_ALL                                UVM_IOCTL_BASE(102)

typedef struct
{
    NvU32     flags;                           // IN - Migration flags (for future use)
    NV_STATUS rmStatus;                        // OUT - Return status
} UVM_LIVE_MIGRATION_PREPARE_ALL_PARAMS;

//
// UvmLiveMigrationGetDirtyPages
// Get addresses of dirty pages (64KB granularity)
// Returns array of virtual addresses that have been written during migration
//
#define UVM_LIVE_MIGRATION_GET_DIRTY_PAGES                            UVM_IOCTL_BASE(104)

typedef struct
{
    NvU64     max_pages;                       // IN  - Maximum number of addresses to return
    NvU64     num_pages;                       // OUT - Actual number of dirty pages found
    NvU64    *dirty_addresses;                 // OUT - Array of 64KB-aligned addresses (user buffer)
    NV_STATUS rmStatus;                        // OUT - Return status
} UVM_LIVE_MIGRATION_GET_DIRTY_PAGES_PARAMS;

// ---------------------------------------------------------------------------
// Checkpoint ioctls (106)
// ---------------------------------------------------------------------------

// 106 – GET_VA_RANGES
// Return the base address and byte size of every UVM_VA_RANGE_TYPE_MANAGED
// allocation present in any VA space on the system.  Requires CAP_SYS_ADMIN.
#define UVM_LIVE_MIGRATION_GET_VA_RANGES                              UVM_IOCTL_BASE(106)

typedef struct
{
    NvU64    max_ranges   NV_ALIGN_BYTES(8); // IN  - capacity of caller arrays
    NvU64    num_ranges   NV_ALIGN_BYTES(8); // OUT - entries written
    NvU64   *base_addrs;                     // IN  - caller NvU64[max_ranges]
    NvU64   *range_sizes;                    // IN  - caller NvU64[max_ranges]
    NV_STATUS rmStatus;                      // OUT
} UVM_LIVE_MIGRATION_GET_VA_RANGES_PARAMS;


// 109 – READ_PAGES_RESIDENT
// Checkpoint a single managed VA range with per-page residency separation.
// No migration is performed: pages stay where they are.
// GPU pages are read via CE with contiguous coalescing (up to 2MB per
// transfer) for reduced push overhead.  In CC mode, CE encrypts and CPU
// decrypts each coalesced chunk — caller receives plaintext.
//
//   residency_map[i] encodes the location of page i (0-based, 4KB pages):
//     0  = CPU-resident  → data written densely to cpu_buf (plaintext)
//     1  = GPU-resident  → data written densely to gpu_buf (plaintext)
//     2  = absent/evicted → not captured; restore should treat as zero
//
// cpu_buf and gpu_buf must be large enough for worst-case (all pages on one
// side).  Use length bytes for each to be safe.
// residency_map must be at least (length / PAGE_SIZE) bytes.
//
// gpu_uuid: pass all-zeros to use the first registered GPU.
// Requires CAP_SYS_ADMIN.  App must be quiesced at AT_BOUNDARY.
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT                        UVM_IOCTL_BASE(109)

typedef struct
{
    NvU64           base          NV_ALIGN_BYTES(8); // IN  – range start (PAGE_SIZE multiple)
    NvU64           length        NV_ALIGN_BYTES(8); // IN  – byte count  (PAGE_SIZE multiple)
    NvU64           cpu_buf       NV_ALIGN_BYTES(8); // IN  – agent userspace buf for CPU pages
    NvU64           cpu_buf_size  NV_ALIGN_BYTES(8); // IN  – capacity of cpu_buf in bytes
    NvU64           gpu_buf       NV_ALIGN_BYTES(8); // IN  – agent userspace buf for GPU pages
    NvU64           gpu_buf_size  NV_ALIGN_BYTES(8); // IN  – capacity of gpu_buf in bytes
    NvU64           residency_map NV_ALIGN_BYTES(8); // IN  – agent NvU8[length/PAGE_SIZE]
    NvProcessorUuid gpu_uuid;                        // IN  – target GPU; {0} = first GPU
    NvU64           cpu_bytes_out NV_ALIGN_BYTES(8); // OUT – bytes written to cpu_buf
    NvU64           gpu_bytes_out NV_ALIGN_BYTES(8); // OUT – bytes written to gpu_buf
    NvU64           num_pages     NV_ALIGN_BYTES(8); // OUT – total 4KB pages in range
    NV_STATUS       rmStatus;                        // OUT
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_PARAMS;

// --------------------------------------------------------------------------
// 111 – READ_PAGES_RESIDENT_ENCRYPTED
// Same two-pass structure as 109, but GPU pages are returned as CE-encrypted
// ciphertext (no CPU decryption) with per-transfer crypto metadata.
// Contiguous GPU pages are coalesced into bulk CE transfers (up to 2MB).
// Each coalesced transfer produces one crypto_meta entry containing the
// IV, auth_tag, key_version, and transfer size.
//
// CPU-resident pages are returned as plaintext (agent encrypts with k3).
// num_transfers (OUT) = number of crypto_meta entries written.
//
// Non-CC mode: behaves identically to ioctl 109.
// Requires CAP_SYS_ADMIN.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED                UVM_IOCTL_BASE(111)

// Per-transfer encryption metadata for coalesced GPU page reads.
// Each entry covers a contiguous run of pages (up to DMA_BUFFER_SIZE = 2MB).
// The 'size' field indicates how many bytes this entry covers.
typedef struct __attribute__((packed))
{
    NvU64 size;                      // number of ciphertext bytes this entry covers
    NvU8  iv[12];                    // AES-GCM initialization vector
    NvU8  iv_fresh;                  // IV freshness flag
    NvU8  auth_tag[16];              // AES-GCM authentication tag
    NvU32 key_version;               // CSL key version at encryption time
} UVM_LIVE_MIGRATION_PAGE_CRYPTO_META;  // 41 bytes

typedef struct
{
    NvU64           base          NV_ALIGN_BYTES(8); // IN  – range start (PAGE_SIZE multiple)
    NvU64           length        NV_ALIGN_BYTES(8); // IN  – byte count  (PAGE_SIZE multiple)
    NvU64           cpu_buf       NV_ALIGN_BYTES(8); // IN  – agent buf for CPU pages (plaintext)
    NvU64           cpu_buf_size  NV_ALIGN_BYTES(8); // IN  – capacity of cpu_buf
    NvU64           gpu_buf       NV_ALIGN_BYTES(8); // IN  – agent buf for GPU pages (ciphertext)
    NvU64           gpu_buf_size  NV_ALIGN_BYTES(8); // IN  – capacity of gpu_buf
    NvU64           residency_map NV_ALIGN_BYTES(8); // IN  – agent NvU8[length/PAGE_SIZE]
    NvU64           crypto_meta   NV_ALIGN_BYTES(8); // IN  – agent buf for per-transfer metadata
    NvU64           crypto_meta_size NV_ALIGN_BYTES(8); // IN – capacity of crypto_meta
    NvProcessorUuid gpu_uuid;                        // IN  – target GPU; {0} = first GPU
    NvU64           cpu_bytes_out NV_ALIGN_BYTES(8); // OUT – bytes written to cpu_buf
    NvU64           gpu_bytes_out NV_ALIGN_BYTES(8); // OUT – bytes written to gpu_buf
    NvU64           num_pages     NV_ALIGN_BYTES(8); // OUT – total 4KB pages in range
    NvU64           num_transfers NV_ALIGN_BYTES(8); // OUT – number of crypto_meta entries
    NV_STATUS       rmStatus;                        // OUT
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS;

// --------------------------------------------------------------------------
// 112 – DECRYPT_ENCRYPTED_PAGES
// Decrypt checkpoint pages encrypted with the GPU→CPU channel key (k1).
// Takes ciphertext + per-page crypto metadata (IV, auth_tag, key_version),
// decrypts using the kernel's CSL context, and returns plaintext to userspace.
//
// The caller can then write the plaintext to GPU via cudaMemcpy or any
// other mechanism — this ioctl is purely a decryption service.
//
// Requires CAP_SYS_ADMIN.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES                      UVM_IOCTL_BASE(112)

typedef struct
{
    NvU64           cipher_buf    NV_ALIGN_BYTES(8); // IN  – ciphertext (k1-encrypted)
    NvU64           plain_buf     NV_ALIGN_BYTES(8); // OUT – plaintext output buffer
    NvU64           total_size    NV_ALIGN_BYTES(8); // IN  – total ciphertext bytes
    NvU64           crypto_meta   NV_ALIGN_BYTES(8); // IN  – per-transfer metadata array
    NvU64           num_transfers NV_ALIGN_BYTES(8); // IN  – number of metadata entries
    NvProcessorUuid gpu_uuid;                        // IN  – GPU whose k1 was used; {0} = first
    NV_STATUS       rmStatus;                        // OUT
} UVM_LIVE_MIGRATION_DECRYPT_ENCRYPTED_PAGES_PARAMS;

// --------------------------------------------------------------------------
// 113 – READ_PAGES_RESIDENT_ENCRYPTED_V2
// Same interface as ioctl 111, but uses double-buffered CE transfers
// to overlap CE encrypt with copy_to_user. Two pre-allocated DMA buffers
// ping-pong to eliminate per-transfer alloc/free and synchronous wait.
//
// Expected to achieve higher throughput than ioctl 111 for large transfers.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_V2              UVM_IOCTL_BASE(113)

typedef UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_PARAMS UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_V2_PARAMS;

// --------------------------------------------------------------------------
// 114 – READ_PAGES_RESIDENT_MT  (multi-threaded, plaintext)
// Same semantics as ioctl 109 but parallelizes PASS 2 across N kthread
// workers. Each worker owns its own pair of DMA buffers, its own CE channel
// (via uvm_push_begin on UVM_CHANNEL_TYPE_GPU_TO_CPU), and a contiguous
// slice of the coalesced run list. For each slice, the worker runs the same
// 2-buffer ping-pong as 109: CE encrypt → wait → CPU decrypt → copy_to_user.
// Workers share the caller's mm via kthread_use_mm so copy_to_user targets
// the caller's userspace buffer.
//
// Goal: remove the serial-CPU-decrypt bottleneck in ioctl 109 by spreading
// both CE submission and CPU decrypt across multiple cores and CE channels.
//
// num_threads: 0 = use default (4). Clamped to [1, 16] by the handler.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_MT                        UVM_IOCTL_BASE(114)

typedef struct
{
    NvU64           base          NV_ALIGN_BYTES(8); // IN  – range start
    NvU64           length        NV_ALIGN_BYTES(8); // IN  – byte count
    NvU64           cpu_buf       NV_ALIGN_BYTES(8); // IN  – agent buf for CPU-resident pages
    NvU64           cpu_buf_size  NV_ALIGN_BYTES(8); // IN
    NvU64           gpu_buf       NV_ALIGN_BYTES(8); // IN  – agent buf for GPU-resident pages (plaintext)
    NvU64           gpu_buf_size  NV_ALIGN_BYTES(8); // IN
    NvU64           residency_map NV_ALIGN_BYTES(8); // IN
    NvU32           num_threads;                     // IN  – 0 = default (4); max 16
    NvU32           _pad;
    NvProcessorUuid gpu_uuid;                        // IN
    NvU64           cpu_bytes_out NV_ALIGN_BYTES(8); // OUT
    NvU64           gpu_bytes_out NV_ALIGN_BYTES(8); // OUT
    NvU64           num_pages     NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                        // OUT
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_MT_PARAMS;

// --------------------------------------------------------------------------
// 115 – READ_PAGES_RESIDENT_ENCRYPTED_MT  (multi-threaded, ciphertext)
// Multi-threaded analogue of ioctl 113. Each worker owns a pair of DMA
// buffers + CE channel and processes a slice of the coalesced run list.
// Workers copy CE ciphertext directly to userspace (no in-kernel decrypt),
// and each worker writes its own slice of the crypto_meta array.
//
// num_threads: 0 = use default (4). Clamped to [1, 16] by the handler.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT              UVM_IOCTL_BASE(115)

typedef struct
{
    NvU64           base             NV_ALIGN_BYTES(8);
    NvU64           length           NV_ALIGN_BYTES(8);
    NvU64           cpu_buf          NV_ALIGN_BYTES(8);
    NvU64           cpu_buf_size     NV_ALIGN_BYTES(8);
    NvU64           gpu_buf          NV_ALIGN_BYTES(8);
    NvU64           gpu_buf_size     NV_ALIGN_BYTES(8);
    NvU64           residency_map    NV_ALIGN_BYTES(8);
    NvU64           crypto_meta      NV_ALIGN_BYTES(8);
    NvU64           crypto_meta_size NV_ALIGN_BYTES(8);
    NvU32           num_threads;
    NvU32           _pad;
    NvProcessorUuid gpu_uuid;
    NvU64           cpu_bytes_out    NV_ALIGN_BYTES(8);
    NvU64           gpu_bytes_out    NV_ALIGN_BYTES(8);
    NvU64           num_pages        NV_ALIGN_BYTES(8);
    NvU64           num_transfers    NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS;

// --------------------------------------------------------------------------
// 116 – READ_DIRTY_DELTA_ENCRYPTED_MT  (Tier3-fused dirty-delta, ciphertext)
// Same MT ciphertext pipeline as 115, but the kernel internally snapshots
// the live-migration dirty bitmap and includes ONLY pages in dirty 2MB
// va_blocks (migration_tracking_enabled && live_migration.dirty) — i.e. it
// fuses GET_DIRTY_PAGES + the per-region 115 loop into ONE ioctl so the
// kthread/DMA-buffer/CE-channel setup tax is paid once for the whole delta
// instead of once per dirty region.
//
// [base, base+length) bounds the VA window to scan (pass the full managed
// footprint, same as the coalesced precopy call). residency_map must cover
// length/PAGE_SIZE bytes. The app must be quiesced at AT_BOUNDARY so the
// dirty set is stable. Reuses the 115 params layout verbatim.
//
// num_threads: 0 = use default (4). Clamped to [1, 16] by the handler.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT                 UVM_IOCTL_BASE(116)

typedef UVM_LIVE_MIGRATION_READ_PAGES_RESIDENT_ENCRYPTED_MT_PARAMS
        UVM_LIVE_MIGRATION_READ_DIRTY_DELTA_ENCRYPTED_MT_PARAMS;

// --------------------------------------------------------------------------
// 117 - BENCH_DECOMPOSE   (instrumented FORK of the 114/115 MT pipeline)
//
// Debug/benchmark ioctl used to decompose CC-mode D2H bandwidth into its
// constituent stages. It is a deliberate copy of the 114/115 worker so the
// production paths (114/115/116) carry no timers and no extra branches.
// NEVER use this on the checkpoint path.
//
// Timed stages (accumulated per worker, then summed and max-reduced):
//   t_submit_ns  : uvm_push_begin .. uvm_push_end   (CE launch + IV/CSL)
//   t_iv_ns      : uvm_conf_computing_log_gpu_encryption (subset of submit)
//   t_ce_wait_ns : uvm_push_wait                    (CE encrypt + PCIe DMA)
//   t_decrypt_ns : uvm_conf_computing_cpu_decrypt   (AES-GCM + mem traffic)
//   t_copy_ns    : copy_to_user / staging memcpy    (push-pull + TME write)
//
// For an ADDITIVE breakdown pass BENCH_SERIAL: with ping-pong overlap on,
// the per-stage times legitimately exceed wall time and must not be summed.
// --------------------------------------------------------------------------
#define UVM_LIVE_MIGRATION_BENCH_DECOMPOSE                               UVM_IOCTL_BASE(117)

#define UVM_LIVE_MIGRATION_BENCH_SERIAL          0x1u  // no ping-pong overlap
#define UVM_LIVE_MIGRATION_BENCH_DECRYPT         0x2u  // run CPU decrypt (114 behavior)
#define UVM_LIVE_MIGRATION_BENCH_COPY_TO_SHARED  0x4u  // memcpy to shared DMA buf, not user
#define UVM_LIVE_MIGRATION_BENCH_SKIP_COPY       0x8u  // omit output copy entirely

typedef struct
{
    NvU64           base             NV_ALIGN_BYTES(8);  // IN
    NvU64           length           NV_ALIGN_BYTES(8);  // IN
    NvU64           cpu_buf          NV_ALIGN_BYTES(8);  // IN
    NvU64           cpu_buf_size     NV_ALIGN_BYTES(8);  // IN
    NvU64           gpu_buf          NV_ALIGN_BYTES(8);  // IN
    NvU64           gpu_buf_size     NV_ALIGN_BYTES(8);  // IN
    NvU64           residency_map    NV_ALIGN_BYTES(8);  // IN
    NvU32           num_threads;                         // IN  (0 = default 4)
    NvU32           dbg_flags;                           // IN  (BENCH_* above)
    NvProcessorUuid gpu_uuid;                            // IN

    NvU64           cpu_bytes_out    NV_ALIGN_BYTES(8);  // OUT
    NvU64           gpu_bytes_out    NV_ALIGN_BYTES(8);  // OUT
    NvU64           num_pages        NV_ALIGN_BYTES(8);  // OUT
    NvU64           num_transfers    NV_ALIGN_BYTES(8);  // OUT
    NvU32           threads_used;                        // OUT
    NvU32           _pad2;

    // Per-stage CPU time, summed across workers.
    NvU64           t_submit_ns      NV_ALIGN_BYTES(8);  // OUT
    NvU64           t_iv_ns          NV_ALIGN_BYTES(8);  // OUT
    NvU64           t_ce_wait_ns     NV_ALIGN_BYTES(8);  // OUT
    NvU64           t_decrypt_ns     NV_ALIGN_BYTES(8);  // OUT
    NvU64           t_copy_ns        NV_ALIGN_BYTES(8);  // OUT

    NvU64           t_worker_max_ns  NV_ALIGN_BYTES(8);  // OUT slowest worker
    NvU64           t_total_ns       NV_ALIGN_BYTES(8);  // OUT whole ioctl

    NV_STATUS       rmStatus;                            // OUT
} UVM_LIVE_MIGRATION_BENCH_DECOMPOSE_PARAMS;


#ifdef __cplusplus
}
#endif

#endif // _UVM_IOCTL_H
