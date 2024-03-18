// AvxPlayground.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <immintrin.h>
#include <assert.h>
#include <random>
#include <chrono>
#include <bitset>
#include <fstream>
#include <sstream>

#include "PermutationTable.h"

#define DIV256 sizeof(__m256i) // 32 bytes (32 * 8 = 256 bits)
#define DIV512 sizeof(__m512i)  // 64 bytes (64 * 8 = 512 bits)

using regI = __m256i;

using namespace std;

uint64_t addNonAVX(uint8_t* a, uint8_t* b, uint8_t* c, unsigned size) {
    auto ts = chrono::system_clock::now();

    for (unsigned i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }

    auto te = chrono::system_clock::now();
    return chrono::duration_cast<chrono::microseconds>(te - ts).count();
}

uint64_t addAVX256(uint8_t* a, uint8_t* b, uint8_t* c, unsigned size) {
    assert(size % DIV512 == 0);

    auto ts = chrono::system_clock::now();

    for (unsigned i = 0; i < size; i += 32) {
        __m256i av = _mm256_loadu_epi8((__m256i*)(a + i));
        __m256i bv = _mm256_loadu_epi8((__m256i*)(b + i));

        __m256i cv = _mm256_adds_epu8(av, bv);

        _mm256_storeu_epi8((__m256i*)(c + i), cv);
    }

    auto te = chrono::system_clock::now();
    return chrono::duration_cast<chrono::microseconds>(te - ts).count();
}

uint64_t addAVX512(uint8_t* a, uint8_t* b, uint8_t* c, unsigned size) {
    assert(size % DIV512 == 0);

    auto ts = chrono::system_clock::now();

    for (unsigned i = 0; i < size; i += 64) {
        __m512i av = _mm512_loadu_epi8((__m512i*)(a + i));
        __m512i bv = _mm512_loadu_epi8((__m512i*)(b + i));

        __m512i cv = _mm512_adds_epu8(av, bv);

        _mm512_storeu_epi8((__m512i*)(c + i), cv);
    }

    auto te = chrono::system_clock::now();
    return chrono::duration_cast<chrono::microseconds>(te - ts).count();
}

void print_epi32(__m256i v) {
    alignas(32) int values[8];
   _mm256_store_si256((__m256i*) values, v);

   for (unsigned i = 0; i < 8; ++i) {
       cout << std::bitset<32>(values[i]) << " (" << values[i] << ") ";
       cout << endl;
   }
   cout << endl;
}

void print_ps(__m256 v) {
    alignas(32) float values[8];
    _mm256_store_ps(values, v);

    for (unsigned i = 0; i < 8; ++i) {
        // Interpret the bits of the float as an unsigned int for bitset
        unsigned int asInt;
        std::memcpy(&asInt, &values[i], sizeof(float));
        std::cout << std::bitset<32>(asInt) << " (" << values[i] << ") ";
    }
    std::cout << std::endl;
}

void partitionBlock(int* dataPtr, __m256i P, int* writeLeft, int* writeRight) {
    // load data into register
    __m256i data = _mm256_loadu_epi32((__m256i*)dataPtr);
    // compare greater than 
    __m256i gtVec = _mm256_cmpgt_epi32(data, P);
    print_epi32(gtVec);
    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(gtVec));
    cout << bitset<8>(mask) << endl;

}

void createInt32PermutationTable() {
    std::ofstream headerFile("PermutationTable.h");
    if (!headerFile.is_open()) {
        std::cerr << "Failed to open header file for writing.\n";
        return;
    }

    headerFile << "#ifndef PERMUTATION_TABLE_H\n";
    headerFile << "#define PERMUTATION_TABLE_H\n\n";
    headerFile << "constexpr int permTable[256][8] = {\n";

    for (unsigned mask = 0; mask < 256; ++mask) {
        std::stringstream ss;
        ss << "   // " << mask << " => 0b";
        int data[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
        int left = 0, right = 0;

        int numRight = _mm_popcnt_u32(mask);
        int numLeft = 8 - numRight;

        for (auto b = 0; b < 8; ++b) {
            if (((mask >> b) & 1) == 0) {
                if (left < numLeft) {
                    data[left++] = b;
                    ss << "0";
                }                    
            } else {
                if (right < numRight)
                {
                    data[numLeft + right++] = b;
                    ss << "1";
                }
            }
        }

        for (unsigned i = 0; i < 8; ++i)
            assert(data[i] != -1);

        headerFile << "    { ";
        for (unsigned i = 0; i < 8; ++i) {
            headerFile << data[i];
            if (i < 7) headerFile << ", ";
        }
        if (mask < 255)
            headerFile << " }," << ss.str() << "\n";
        else
            headerFile << " }" << ss.str() << "\n";
    }

    headerFile << "};\n\n";
    headerFile << "#endif // PERMUTATION_TABLE_H\n";

    headerFile.close();
    std::cout << "Permutation table generated in PermutationTable.h\n";
}

int main() {

    // populate vector
    __m256i vecI = _mm256_set_epi32(0, -1, 2, 3, 4, 5, 6, 7);
    __m256 vecS = _mm256_castsi256_ps(vecI);
    //__m256 vec = _mm256_set_ps(0.1, -1.1, -2, -3, -4, -5, -6, -7); // gibt keine methode für int
    //print_ps(vec);

    //int mask = _mm256_movemask_ps(vecS);
    //cout << bitset<8>(mask) << " Value: " << mask << endl;
    
    size_t size = 8;
    int data[8]; // stack
    //memset(data, 0, size * sizeof(int)); // memset sets bytes not array elements. Thus * sizeof(int)

    for (unsigned i = 0; i < 8; ++i) {
        data[i] = 2 * i;
    }
    data[7] -= 8;

    regI P = _mm256_set1_epi32(7);

    int* wL = data;
    int* wR = data;

    //partitionBlock(data, P, wL, wR);
    //createInt32PermutationTable();



    // test perm table
    auto dataVec = _mm256_set_epi32(1, 1, 1, 1, 1, 8, 9, 10);
    auto mask = _mm256_movemask_ps(
                    _mm256_castsi256_ps(
                        _mm256_cmpgt_epi32(dataVec, P)
                    ));
    
    std::cout << std::bitset<32>(mask) << endl;

    const int* PermTablePtr = permTable[0];

    
    auto permVec = _mm256_loadu_epi32((__m256i*)PermTablePtr + mask);
    print_epi32(permVec);

    return 0;
}

