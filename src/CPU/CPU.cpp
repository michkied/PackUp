#include <cstring>
#include <vector>

#include "CPU/CPU.hpp"

namespace CPU 
{
    void fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter) {}

    void fixed_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) {}

    void run_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter)
    {
        unsigned int symbol_size = parameter;
        unsigned int partition_size = 255;

        std::vector<unsigned char> array = std::vector<unsigned char>();
        std::vector<unsigned char> data = std::vector<unsigned char>();
        int count = 1;
        std::vector<unsigned char> last_symbol = std::vector<unsigned char>();
        std::vector<unsigned char> this_symbol = std::vector<unsigned char>();

        for (long unsigned int byte = 0; byte < input_size; byte++)
        {
            if (last_symbol.size() < symbol_size)
            {
                last_symbol.push_back(input[byte]);
                continue;
            }

            if (this_symbol.size() < symbol_size)
            {
                this_symbol.push_back(input[byte]);
                continue;
            }

            if (last_symbol != this_symbol) {
                if (count != 0) {
                    array.push_back(count);
                    for (unsigned char& symbol : last_symbol)
                    {
                        data.push_back(symbol);
                    }
                }

                std::copy(this_symbol.begin(), this_symbol.end(), last_symbol.begin());
                this_symbol.clear();
                this_symbol.push_back(input[byte]);
                count = 1;
                continue;
            }

            count++;
            if (count == partition_size)
            {
                array.push_back(count);
                for (unsigned char& symbol : last_symbol)
                {
                    data.push_back(symbol);
                }
                count = 0;
            }
            this_symbol.clear();
			this_symbol.push_back(input[byte]);
        }

		if (last_symbol == this_symbol)
		{
			count++;
			array.push_back(count);
			for (unsigned char& symbol : last_symbol)
			{
				data.push_back(symbol);
			}
		}
		else
		{
			if (count != 0)
			{
                array.push_back(count);
                for (unsigned char& symbol : last_symbol)
                {
                    data.push_back(symbol);
                }
			}

			if (this_symbol.size() == symbol_size)
			{
                array.push_back(1);
			}

            for (unsigned char& symbol : this_symbol)
            {
                data.push_back(symbol);
            }
		}

		output_size = array.size() + data.size() + 5;
		unsigned int array_size = array.size();
		output = new unsigned char[output_size];
		output[0] = symbol_size;
        std::memcpy(output+1, &array_size, 4);
		std::memcpy(output + 5, array.data(), array.size());
		std::memcpy(output + 5 + array.size(), data.data(), data.size());
    }

    void run_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size)
    {
        unsigned int header_size = 5;
        unsigned int symbol_size = input[0];
        unsigned int array_size;
        std::memcpy(&array_size, input + 1, 4);
        unsigned int remaining_symbols = input_size - header_size - array_size - array_size * symbol_size;

        output_size = 0;
        output = new unsigned char[symbol_size * array_size * 255];
        for (unsigned int symbol_index = 0; symbol_index < array_size; ++symbol_index)
        {
            for (int rep = 0; rep < input[header_size + symbol_index]; ++rep)
            {
                for (int byte = 0; byte < symbol_size; ++byte)
                {
                    output[output_size] = input[header_size + array_size + symbol_index * symbol_size + byte];
                    output_size++;
                }
            }
        }
        while (remaining_symbols > 0)
        {
            output[output_size++] = input[input_size - remaining_symbols--];
        }
    }
}

