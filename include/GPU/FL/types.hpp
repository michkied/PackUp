#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// Iterates cyclically over an array of keys
class CyclicIterator {
public:
    using value_type = unsigned int;
    using difference_type = unsigned int;
    using pointer = const unsigned int*;
    using reference = const unsigned int&;
    using iterator_category = thrust::random_access_host_iterator_tag;

    __host__ __device__
        CyclicIterator(const unsigned int* keys, unsigned int size, unsigned int iteration = 0)
        : _keys(keys), _size(size), _iteration(iteration) {}

    __host__ __device__
        CyclicIterator(unsigned int size, unsigned int iteration = 0)
        : _keys(nullptr), _size(size), _iteration(iteration) {
    }

    __host__ __device__
        value_type operator*() const {
		if (_keys == nullptr) {
			return _iteration % _size;
		}
        return _keys[_iteration % _size];
    }

    __host__ __device__
        value_type operator[](difference_type i) const {
		if (_keys == nullptr) {
			return (_iteration + i) % _size;
		}
        return _keys[(i + _iteration) % _size];
    }

    __host__ __device__
        CyclicIterator& operator++() {
        ++_iteration;
        return *this;
    }

    __host__ __device__
        CyclicIterator operator++(int) {
        CyclicIterator temp = *this;
        ++(*this);
        return temp;
    }

    // Arithmetic operations
    __host__ __device__
        CyclicIterator operator+(difference_type n) const {
        return CyclicIterator(_keys, _size, _iteration + n);
    }

    __host__ __device__
        CyclicIterator& operator+=(difference_type n) {
        _iteration += n;
        return *this;
    }

    __host__ __device__
        CyclicIterator operator-(difference_type n) const {
        return CyclicIterator(_keys, _size, _iteration - n);
    }

    __host__ __device__
        difference_type operator-(const CyclicIterator& other) const {
        return _iteration - other._iteration;
    }

    // Comparison operators
    __host__ __device__
        bool operator==(const CyclicIterator& other) const {
        return _iteration == other._iteration;
    }

    __host__ __device__
        bool operator!=(const CyclicIterator& other) const {
        return !(*this == other);
    }

private:
    const unsigned int* _keys;
    unsigned int _size;
	unsigned int _iteration;
};


// Allows the usage of thrust reduction while keeping track of other connected values
struct DivisionWrapper {
    unsigned int removed_zeros = 0;
    unsigned int seg_size = 0;
    unsigned int insig_zeros = 0;

    __host__ __device__ DivisionWrapper(unsigned int removed_zeros, unsigned int seg_size, unsigned int insig_zeros)
		: removed_zeros(removed_zeros), seg_size(seg_size), insig_zeros(insig_zeros) {
	}

    __host__ __device__ DivisionWrapper() {}

	__host__ __device__ DivisionWrapper operator+(const DivisionWrapper& other) const {
		return DivisionWrapper(removed_zeros + other.removed_zeros, seg_size + other.seg_size, insig_zeros + other.insig_zeros);
	}

	__host__ __device__ bool operator>(const DivisionWrapper& other) const {
		return removed_zeros > other.removed_zeros;
	}

	__host__ __device__ bool operator<(const DivisionWrapper& other) const {
		return removed_zeros < other.removed_zeros;
	}

	__host__ __device__ bool operator==(const DivisionWrapper& other) const {
		return removed_zeros == other.removed_zeros;
	}

	__host__ __device__ bool operator>=(const DivisionWrapper& other) const {
		return removed_zeros >= other.removed_zeros;
	}

	__host__ __device__ bool operator<=(const DivisionWrapper& other) const {
		return removed_zeros <= other.removed_zeros;
	}

	__host__ __device__ bool operator!=(const DivisionWrapper& other) const {
		return removed_zeros != other.removed_zeros;
	}
};