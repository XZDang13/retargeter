// ZeroMQ wrapper for rtnet.h: one shared context, plain integer results.
#include "rtnet.h"

#include <zmq.h>

struct rtnet_socket_s { void* zmq; };

static void* context(void) {
    static void* ctx = 0;
    if (!ctx) ctx = zmq_ctx_new();
    return ctx;
}

static const int types[] = {ZMQ_PAIR, ZMQ_PUB, ZMQ_SUB, ZMQ_REQ, ZMQ_REP, ZMQ_DEALER, ZMQ_ROUTER, ZMQ_PULL, ZMQ_PUSH};

rtnet_socket rtnet_open(int type) {
    if (type < 0 || type > RTNET_PUSH) return 0;
    void* s = zmq_socket(context(), types[type]);
    if (!s) return 0;
    int linger = 0;
    zmq_setsockopt(s, ZMQ_LINGER, &linger, sizeof linger);
    static struct rtnet_socket_s pool[256];
    static int used = 0;
    // Sockets live for the process (a teleop server opens a handful); the pool avoids malloc in the C API.
    if (used >= 256) { zmq_close(s); return 0; }
    pool[used].zmq = s;
    return &pool[used++];
}

void rtnet_close(rtnet_socket socket) {
    if (socket && socket->zmq) { zmq_close(socket->zmq); socket->zmq = 0; }
}

int rtnet_bind(rtnet_socket socket, const char* endpoint) { return zmq_bind(socket->zmq, endpoint); }
int rtnet_connect(rtnet_socket socket, const char* endpoint) { return zmq_connect(socket->zmq, endpoint); }

int rtnet_set_int(rtnet_socket socket, int option, int value) {
    return zmq_setsockopt(socket->zmq, option, &value, sizeof value);
}

int rtnet_subscribe(rtnet_socket socket, const void* prefix, size_t length) {
    return zmq_setsockopt(socket->zmq, ZMQ_SUBSCRIBE, prefix, length);
}

int rtnet_send(rtnet_socket socket, const void* data, size_t length, int more, int nonblocking) {
    int flags = (more ? ZMQ_SNDMORE : 0) | (nonblocking ? ZMQ_DONTWAIT : 0);
    int r = zmq_send(socket->zmq, data, length, flags);
    if (r < 0 && zmq_errno() == EAGAIN) return -2;
    return r < 0 ? -1 : 0;
}

int64_t rtnet_recv(rtnet_socket socket, void* buffer, size_t capacity, int timeout_ms, int* more) {
    if (timeout_ms != -1) {
        zmq_pollitem_t item = {socket->zmq, 0, ZMQ_POLLIN, 0};
        int ready = zmq_poll(&item, 1, timeout_ms);
        if (ready < 0) return -1;
        if (ready == 0) return -2;
    }
    int r = zmq_recv(socket->zmq, buffer, capacity, 0);
    if (r < 0) return zmq_errno() == EAGAIN ? -2 : -1;
    if (more) {
        int m = 0;
        size_t size = sizeof m;
        zmq_getsockopt(socket->zmq, ZMQ_RCVMORE, &m, &size);
        *more = m;
    }
    return r;
}

int rtnet_poll(rtnet_socket* sockets, int count, int timeout_ms) {
    zmq_pollitem_t items[32];
    if (count > 32) count = 32;
    for (int i = 0; i < count; i++) {
        items[i].socket = sockets[i]->zmq;
        items[i].fd = 0;
        items[i].events = ZMQ_POLLIN;
        items[i].revents = 0;
    }
    int r = zmq_poll(items, count, timeout_ms);
    if (r < 0) return -1;
    int mask = 0;
    for (int i = 0; i < count; i++) if (items[i].revents & ZMQ_POLLIN) mask |= 1 << i;
    return mask;
}

const char* rtnet_error(void) { return zmq_strerror(zmq_errno()); }

int rtnet_version(void) {
    int major, minor, patch;
    zmq_version(&major, &minor, &patch);
    return major * 10000 + minor * 100 + patch;
}
