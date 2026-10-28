// Minimal C API over ZeroMQ for the teleop server (imported by Swift as module CRTNet).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rtnet_socket_s* rtnet_socket;

enum {
    RTNET_PAIR = 0, RTNET_PUB = 1, RTNET_SUB = 2, RTNET_REQ = 3, RTNET_REP = 4,
    RTNET_DEALER = 5, RTNET_ROUTER = 6, RTNET_PULL = 7, RTNET_PUSH = 8,
};

/// Opens a socket of `type` on the process-wide context; NULL on failure (see rtnet_error).
rtnet_socket rtnet_open(int type);
void rtnet_close(rtnet_socket socket);
int rtnet_bind(rtnet_socket socket, const char* endpoint);
int rtnet_connect(rtnet_socket socket, const char* endpoint);
/// Integer socket options by ZeroMQ number (e.g. ZMQ_SNDHWM 23, ZMQ_RCVHWM 24, ZMQ_LINGER 17, ZMQ_CONFLATE 54).
int rtnet_set_int(rtnet_socket socket, int option, int value);
/// SUB: subscribe to messages starting with `prefix` (length 0: all).
int rtnet_subscribe(rtnet_socket socket, const void* prefix, size_t length);
/// Sends one message; `more`: further frames follow. Non-blocking: returns -2 when it would block.
int rtnet_send(rtnet_socket socket, const void* data, size_t length, int more, int nonblocking);
/// Receives one message frame into `buffer` (truncated to `capacity`); waits up to `timeout_ms` (-1: forever,
/// 0: poll). Returns the full frame length, -2 on timeout, -1 on error; `*more` = further frames follow.
int64_t rtnet_recv(rtnet_socket socket, void* buffer, size_t capacity, int timeout_ms, int* more);
/// Waits until one of `sockets` has input, up to `timeout_ms`; returns a bitmask of the ready ones (-1: error).
int rtnet_poll(rtnet_socket* sockets, int count, int timeout_ms);
const char* rtnet_error(void);
/// ZeroMQ version as major * 10000 + minor * 100 + patch.
int rtnet_version(void);

#ifdef __cplusplus
}
#endif
