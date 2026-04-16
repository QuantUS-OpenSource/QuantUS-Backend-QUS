#include <Python.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32 // Windows-specific includes and definitions
#include <io.h>
#define open _open
#define close _close
#define lseek _lseek
#define read _read
typedef long ssize_t;
#else // Unix-like includes and definitions
#include <unistd.h>
#endif

int get_array_shape(long long num_clumps, char* fn, int offset_bytes){
    long long i = 0;
    char* bytes_read = calloc(256, 1);

    printf("%s\n", fn);

    int fd = open(fn, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(errno);
    }
    if (lseek(fd, offset_bytes, SEEK_SET) == -1) {
        perror("lseek");
        exit(errno);
    }

    while (i < num_clumps) {
        ssize_t num_bytes_read = read(fd, bytes_read, 32);
        if (num_bytes_read == -1) {
            perror("read");
            free(bytes_read);
            close(fd);
            exit(errno);
        }
        if (!num_bytes_read) {
            break;
        }
        ++i;    
    }
    return i;
}

int* get_partA(long long num_clumps, char* fn, int offset_bytes) {
    // equivalent to "partA = fread(fid, [12, numClumps], '12*ubit21',4);" in MATLAB

    int fd = open(fn, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(errno);
    }

    // get partA
    int* partA = calloc(12*num_clumps, sizeof(int));
    char* bytes_read = calloc(256, 1);
    if (lseek(fd, offset_bytes, SEEK_SET) == -1) {
        perror("lseek");
        exit(errno);
    }

    int i = 0, x = 0, j = 0, bits_left = 0;
    int bit_offset = 4;
    unsigned char first, second, third, mask, temp;
    char* full_num = calloc(4, sizeof(char));
    while (j < num_clumps) {
        if (!j || i == 31) {
            assert(bit_offset == 4);
            bit_offset = 8;
            ssize_t num_bytes_read = read(fd, bytes_read, 32);
            if (num_bytes_read == -1) {
                perror("read");
                free(bytes_read);
                free(full_num);
                close(fd);
                exit(errno);
            }
            if (!num_bytes_read) {
                printf("EOF reached at column %i\n", j); 
                break;
            }
            ++j; i = 0;
        }
        else {
            mask = (~0)<<(8-bit_offset);
            first = (bytes_read[i] & mask) >> (8-bit_offset);
            first |= (bytes_read[i+1] << (bit_offset));
            second = bytes_read[i+1];
            second = second >> (8 - bit_offset);
            third = bytes_read[i+2];
            second |= (bytes_read[i+2] << (bit_offset));
            third = bytes_read[i+2];
            third = third >> (8 - bit_offset);

            bits_left = 5 - bit_offset;
            if (bits_left > 0) {
                ++i;
                mask = ~((~0)<<bits_left);
                temp = mask & bytes_read[i+2];
                third |= temp << bit_offset;
                bit_offset = 8 - bits_left;
            }
            else if (bits_left < 0) {
                mask = ~(((uint32_t)~0)<<5);
                third &= mask;
                bit_offset = -1 * bits_left;
            }
            else {
                ++i;
                bit_offset = 8;
            }
            full_num[0] = first; full_num[1] = second; full_num[2] = third;
            partA[x] = *((int*)full_num);
            ++x;
            i += 2;
        }
    }

    free(bytes_read);
    free(full_num);
    close(fd);

    return partA;
}

static PyObject* py_get_partA(PyObject* self, PyObject* args) {
    long long num_clumps;
    const char* fn;
    int offset_bytes;

    if (!PyArg_ParseTuple(args, "Lsi", &num_clumps, &fn, &offset_bytes)) {
        return NULL;
    }

    int* partA = get_partA(num_clumps, (char*)fn, offset_bytes);
    PyObject* list = PyList_New(12 * num_clumps);
    for (int i = 0; i < 12 * num_clumps; i++) {
        PyList_SetItem(list, i, PyLong_FromLong(partA[i]));
    }
    free(partA);
    return list;
}

int* get_partB(long long num_clumps, char* fn, int offset_bytes) {
    // equivalent to "partB = fread(fid, [1, numClumps], '1*ubit4', 252);" in MATLAB

    int fd = open(fn, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(errno);
    }

    // get partB
    int* partB = calloc(num_clumps, sizeof(int));
    char* bytes_read = calloc(256, 1);
    if (lseek(fd, offset_bytes, SEEK_SET) == -1) {
        perror("lseek");
        exit(errno);
    }

    int x = 0, j = 0;
    unsigned char cur_num, mask;
    char* full_num = calloc(4, sizeof(char));
    mask = ~((uint32_t)(~0)<<4);
    while (j < num_clumps) {
        ssize_t num_bytes_read = read(fd, bytes_read, 32);
        if (num_bytes_read == -1) {
            perror("read");
            free(bytes_read);
            free(full_num);
            close(fd);
            exit(errno);
        }
        if (!num_bytes_read) {
            printf("EOF reached at column %i\n", j); 
            break;
        }
        ++j;
        cur_num = bytes_read[0];
        cur_num &= mask;
        partB[x] = (int)cur_num;
        ++x;
    }

    free(bytes_read);
    free(full_num);
    close(fd);

    return partB;
}

static PyObject* py_get_partB(PyObject* self, PyObject* args) {
    long long num_clumps;
    const char* fn;
    int offset_bytes;

    if (!PyArg_ParseTuple(args, "Lsi", &num_clumps, &fn, &offset_bytes)) {
        return NULL;
    }

    int* partB = get_partB(num_clumps, (char*)fn, offset_bytes);
    PyObject* list = PyList_New(num_clumps);
    for (int i = 0; i < num_clumps; i++) {
        PyList_SetItem(list, i, PyLong_FromLong(partB[i]));
    }
    free(partB);
    return list;
}

// Define methods
static PyMethodDef MyMethods[] = {
    {"getPartA", py_get_partA, METH_VARARGS, "Get part A data from file"},
    {"getPartB", py_get_partB, METH_VARARGS, "Get part B data from file"},
    {NULL, NULL, 0, NULL}
};

// Define module
static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "philipsRfParser",
    NULL,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_philipsRfParser(void) {
    return PyModule_Create(&mymodule);
}